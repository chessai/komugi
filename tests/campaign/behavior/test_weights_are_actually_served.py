"""The unowned link: a checkpoint that was written is a checkpoint that is served.

A campaign loop is only a loop because iteration 1 plays with iteration 0's
weights. Every other test in this suite would pass unchanged if it did not: the
shards would still be written, the checkpoints would still be numbered, the
journal would still fill up, and the whole thing would be a very expensive way
to generate random-play data. Nothing on either side of the seam can catch that
-- ``train/`` proves a checkpoint was written, ``selfplay/`` proves a net was
served, and the sentence joining them ("the same net") lives in a docstring.

So this module proves it mechanically, at the one place the claim is
falsifiable: :func:`komugi.campaign.factory.serving_net` is the module-level
callable a *spawned server process* imports and calls (design D2/F18 -- a model
never crosses the process boundary), and :data:`CHECKPOINT_ENV` is the only
channel through which the parent can tell it which weights to build. So:

1. set ``KOMUGI_CAMPAIGN_CKPT`` to iteration 0's checkpoint,
2. call ``serving_net()`` exactly as the server would,
3. assert the result is **bit-identical** to what
   :func:`komugi.train.checkpoint.load_checkpoint` reads out of that file, and
   **different** from a fresh ``serving_net()`` with the variable unset.

Step 3's two halves are two different failures. Without the "bit-identical"
half, a factory that loaded *some* checkpoint (the newest, a stale one, a
half-trained one) would pass. Without the "different from fresh" half, a
factory that silently ignored the variable and returned a fresh net would pass
whenever the comparison was loose enough -- which is why the checkpoint being
served is isolated into a directory of its own first: pointed at the campaign
root, "load the newest" and "load the one I asked for" are indistinguishable.
"""

from __future__ import annotations

import contextlib
import math
import os
import shutil
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import pytest
import torch

from komugi.campaign.boundary import CAMPAIGN_MARKER, LOSS_TAIL_MEDIAN
from komugi.campaign.factory import new_net
from komugi.selfplay.orchestrate import require_module_level_factory
from komugi.telemetry.journal import read_records
from komugi.telemetry.liveness import fingerprint_series_complete
from komugi.telemetry.records import (
    EMA,
    SELF_PLAY,
    TRAIN,
    IterationMetrics,
    NetFingerprint,
)
from komugi.train.checkpoint import (
    Checkpoint,
    list_checkpoints,
    load_checkpoint,
    load_checkpoint_file,
)
from komugi.train.digest import canonical_digest, parameter_keys
from komugi.train.step import TrainStep

import campaign_rig as rig

camp = pytest.importorskip("komugi.campaign.loop")

if not hasattr(camp, "run_campaign"):
    pytest.skip(
        "komugi.campaign.loop exports no run_campaign yet (KOM-031)",
        allow_module_level=True,
    )


@contextlib.contextmanager
def _checkpoint_env(value: str | None) -> Iterator[None]:
    """Run the block with ``CHECKPOINT_ENV`` set to `value` (or unset).

    ``None`` means *absent*, not empty: "the variable is not set" is the
    first-iteration case the factory has to handle by building a fresh net, and
    an empty string is a different input that a factory is free to treat as an
    error. The previous value is restored either way, because a leaked
    environment variable would silently change what every later test in this
    interpreter serves.
    """
    previous = os.environ.get(rig.CHECKPOINT_ENV)
    if value is None:
        os.environ.pop(rig.CHECKPOINT_ENV, None)
    else:
        os.environ[rig.CHECKPOINT_ENV] = value
    try:
        with rig.small_net_env():
            yield
    finally:
        if previous is None:
            os.environ.pop(rig.CHECKPOINT_ENV, None)
        else:
            os.environ[rig.CHECKPOINT_ENV] = previous


def _serving_net(value: str | None) -> Any:
    """Build a net the way a spawned server would, under `value`.

    ``eval()`` on the way out so a later forward pass is a function of the
    weights alone: the trunk has BatchNorm, and comparing two nets' logits in
    training mode would compare batch statistics as well.
    """
    with _checkpoint_env(value):
        net = rig.campaign_factory().serving_net()
    net.eval()
    return net


def _observation() -> torch.Tensor:
    """One deterministic observation-shaped batch, ``(1, 128, 9, 9)``.

    Not zeros: a zero input can mask a weight difference outright (every
    convolution of zero is its bias), which would make the logits comparison
    below pass for two completely different nets.
    """
    total = rig.IN_PLANES * rig.BOARD_H * rig.BOARD_W
    ramp = torch.linspace(-1.0, 1.0, total, dtype=torch.float32)
    return ramp.reshape(1, rig.IN_PLANES, rig.BOARD_H, rig.BOARD_W)


@pytest.fixture(scope="module")
def campaign() -> rig.SharedCampaign:
    """The shared two-iteration campaign."""
    return rig.shared_campaign()


@pytest.fixture(scope="module")
def iteration_zero_checkpoint(
    campaign: rig.SharedCampaign, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    """Iteration 0's checkpoint, copied into a directory holding nothing else.

    The isolation is the experiment's control. The campaign root holds two
    checkpoints, so a factory that resolves ``CHECKPOINT_ENV`` as "a directory,
    load the newest" and one that resolves it as "the exact file I named" would
    return *different* nets there and this module could not tell which contract
    it had confirmed. With one file in the directory, both readings resolve to
    iteration 0 and the assertion is about the weights rather than about the
    spelling.
    """
    campaign.results()  # surface a failed campaign here rather than as a KeyError
    refs = rig.checkpoint_refs(campaign.root)
    assert len(refs) >= 2, (
        f"want at least two checkpoints under {campaign.root}, found "
        f"{[ref.path.name for ref in refs]}"
    )
    served = Path(tmp_path_factory.mktemp("served"))
    shutil.copy2(refs[0].path, served / refs[0].path.name)
    return served / refs[0].path.name


@pytest.fixture(scope="module")
def loaded(iteration_zero_checkpoint: Path) -> Checkpoint:
    """The same checkpoint read through :mod:`komugi.train.checkpoint`.

    The independent half of the comparison: this path verifies the digest in
    the filename before unpickling, so a checkpoint that got here at all is one
    whose bytes are intact, and the tensors it yields are the reference the
    served net is held to.
    """
    checkpoint = load_checkpoint(iteration_zero_checkpoint.parent)
    assert checkpoint is not None, (
        f"komugi.train.checkpoint cannot read {iteration_zero_checkpoint}"
    )
    return checkpoint


@pytest.fixture(scope="module")
def served(iteration_zero_checkpoint: Path) -> tuple[str, Any]:
    """``serving_net()`` under the checkpoint, and the spelling that worked.

    The contract pins the *variable*, not whether its value names the
    checkpoint file or the directory holding it. Both are tried, in that order,
    and the first that builds a net wins; the isolation above makes the two
    resolve to the same checkpoint, so accepting either pins the behaviour
    without over-pinning the spelling. Failing both is a hard failure with
    every error attached -- that is a factory that cannot be told what to
    serve.
    """
    attempts: list[str] = []
    for candidate in (
        str(iteration_zero_checkpoint),
        str(iteration_zero_checkpoint.parent),
    ):
        try:
            return candidate, _serving_net(candidate)
        except Exception as exc:  # noqa: BLE001 -- every failure is reported below
            attempts.append(f"{rig.CHECKPOINT_ENV}={candidate!r}: {exc!r}")
    pytest.fail(
        "serving_net() could not be pointed at iteration 0's checkpoint by any "
        "spelling of " + rig.CHECKPOINT_ENV + ":\n  " + "\n  ".join(attempts)
    )


@pytest.fixture(scope="module")
def fresh() -> Any:
    """``serving_net()`` with the variable unset -- iteration 0's own net."""
    return _serving_net(None)


def test_the_checkpoint_variable_is_the_pinned_one() -> None:
    """``factory.CHECKPOINT_ENV`` is ``KOMUGI_CAMPAIGN_CKPT``.

    A constant, asserted, because it is an interface between two *processes*: a
    rename that updated the factory and not the loop (or a future operator
    script) would not fail anything -- self-play would carry on with fresh
    weights, and the campaign would produce a beautifully monotone set of
    checkpoints that no game was ever played with.
    """
    factory = rig.campaign_factory()
    assert factory.CHECKPOINT_ENV == rig.CHECKPOINT_ENV, (
        f"factory.CHECKPOINT_ENV is {factory.CHECKPOINT_ENV!r}, want "
        f"{rig.CHECKPOINT_ENV!r}"
    )


def test_serving_net_is_something_a_spawned_server_can_import() -> None:
    """It is module-level, per design D2/F18, judged by the orchestrator's own rule.

    Checked with :func:`~komugi.selfplay.orchestrate.require_module_level_factory`
    rather than by re-deriving the rule here: that function is what
    ``run_selfplay`` applies to whatever the campaign hands it, so this test
    fails exactly when the real run would -- but at the seam, in one sentence,
    instead of as a pickling error inside a child process.
    """
    require_module_level_factory(rig.campaign_factory().serving_net)


def test_the_served_net_is_the_checkpoint_bit_for_bit(
    served: tuple[str, Any], loaded: Checkpoint
) -> None:
    """Every parameter and buffer served equals the checkpoint's, exactly.

    ``torch.equal``, not a tolerance: the claim is that these are the same
    weights, and the interesting failure -- serving a net one training
    iteration stale, or one that was re-initialised and then partially loaded
    -- is exactly the kind that a tolerance forgives.
    """
    spelling, net = served
    state: dict[str, torch.Tensor] = net.state_dict()
    assert set(state) == set(loaded.model_state), (
        f"served net's state_dict keys differ from the checkpoint's "
        f"(served via {spelling!r}); "
        f"only-served={sorted(set(state) - set(loaded.model_state))}, "
        f"only-checkpoint={sorted(set(loaded.model_state) - set(state))}"
    )
    mismatched = [
        key
        for key in state
        if not torch.equal(state[key].cpu(), loaded.model_state[key].cpu())
    ]
    assert not mismatched, (
        f"served via {spelling!r}, but {len(mismatched)} tensor(s) differ from "
        f"{loaded.path.name}: {mismatched[:8]}"
    )


def test_the_served_net_is_not_a_fresh_one(served: tuple[str, Any], fresh: Any) -> None:
    """Unset and set must not produce the same weights.

    The control for the test above: a factory that ignored the environment
    entirely would still be "bit-identical to the checkpoint" if the checkpoint
    happened to hold a freshly initialised net. It does not -- iteration 0
    trained it -- so a served net equal to a fresh one means the variable was
    read by nobody.
    """
    spelling, net = served
    assert not rig.state_dicts_match(net.state_dict(), fresh.state_dict()), (
        f"serving_net() returned the same weights with {rig.CHECKPOINT_ENV} set "
        f"to {spelling!r} as with it unset: the checkpoint is being ignored"
    )


def test_the_two_nets_predict_differently(served: tuple[str, Any], fresh: Any) -> None:
    """The weight difference reaches the output, on a real observation shape.

    A state-dict comparison can be satisfied by a tensor nothing reads (a
    stale buffer, a head that is never evaluated). This is the same claim made
    where it matters: feed both nets the same ``(1, 128, 9, 9)`` batch and the
    policy the search would act on is not the same policy.
    """
    _, net = served
    observation = _observation()
    with torch.no_grad():
        served_out = net(observation)
        fresh_out = fresh(observation)
    assert not torch.equal(served_out.policy_logits, fresh_out.policy_logits), (
        "the served net and a fresh net produce identical policy logits: "
        "whatever was loaded, it is not affecting what self-play plays"
    )
    assert torch.isfinite(served_out.policy_logits).all(), (
        "the served net produced non-finite policy logits"
    )


def test_serving_is_repeatable(served: tuple[str, Any]) -> None:
    """Two servers built from one checkpoint serve the same net.

    Design D2 builds one net per *incarnation*, and a restarted shard mid-run
    must not silently change strength. Repeatability here is what makes that
    true by construction rather than by luck -- and it is the property that
    lets ``serial_debug`` claim a bit-reproducible replay of a production game.
    """
    spelling, net = served
    again = _serving_net(spelling)
    assert rig.state_dicts_match(net.state_dict(), again.state_dict()), (
        f"two serving_net() calls under {rig.CHECKPOINT_ENV}={spelling!r} "
        f"produced different weights"
    )


# ===========================================================================
# KOM-022 §2/F4: the trainer resumes from <root>/ckpt while self-play serves
# <root>/ema. This is the most dangerous edit in that ticket, and rev 1 of its
# design had no test that would catch it.
# ===========================================================================


def test_trainer_resumes_from_train_while_selfplay_serves_ema(
    campaign: rig.SharedCampaign,
) -> None:
    """Acceptance #28: two bindings, not one repointed one.

    ``run_campaign``'s single ``latest`` binding was used three times -- to set
    ``CHECKPOINT_ENV``, to load the training net, and to load the optimiser
    state. Repointing it at ``<root>/ema`` makes **the trainer resume from EMA
    weights**, and only half of that is loud: an implementer who guards the
    optimiser load (an entirely natural reading of "an EMA has no optimiser")
    is left with the training net loading smoothed weights every iteration.
    Training then descends from a smoothed net and re-smooths it, the recursion
    the band gate is derived from no longer describes the system, and
    **nothing raises**.

    Asserted from the journal's own fingerprints rather than from the live
    environment, because ``run_campaign`` restores ``CHECKPOINT_ENV`` in a
    ``finally``: after iteration ``i``, the ``self_play`` digest is the *EMA's*
    and the ``train`` digest is the *checkpoint directory's*, and the two are
    different nets.
    """
    campaign.results()
    root = campaign.root
    loop = rig.campaign_loop()

    ema_refs = list_checkpoints(loop.ema_dir(root))
    ckpt_refs = list_checkpoints(loop.checkpoint_dir(root))
    assert ema_refs, f"no EMA checkpoint under {loop.ema_dir(root)}"
    assert ckpt_refs, f"no train checkpoint under {loop.checkpoint_dir(root)}"
    assert ema_refs[0].step == ckpt_refs[0].step, (
        "the EMA and the trainer are at different steps: the EMA is published "
        "after its source, at the same step, so this means one of them did not "
        "happen"
    )

    fingerprints = read_records(
        loop.journal_path(root), NetFingerprint, missing_ok=True
    ).records
    assert fingerprints, "the campaign wrote no NetFingerprint records at all"

    by_role: dict[tuple[str, int], str] = {
        (entry.role, entry.iteration): entry.sha256 for entry in fingerprints
    }
    iterations = sorted({entry.iteration for entry in fingerprints})
    latest = iterations[-1]
    assert latest >= 1, (
        "the cross-role invariant starts at iteration 1, which is the first "
        "iteration at which 'did the loop point self-play at the EMA at all' "
        "is answerable; this campaign ran only iteration 0"
    )

    with rig.small_net_env():
        keys = parameter_keys(new_net())

    ema_state = load_checkpoint(loop.ema_dir(root))
    train_state = load_checkpoint(loop.checkpoint_dir(root))
    assert ema_state is not None and train_state is not None

    ema_digest = canonical_digest(ema_state.model_state, keys).sha256
    train_digest = canonical_digest(train_state.model_state, keys).sha256

    # (a) what self-play served for iteration `latest` is the *previous* EMA...
    assert by_role[(SELF_PLAY, latest)] == by_role[(EMA, latest - 1)], (
        "the served fingerprint is not the previous iteration's EMA"
    )
    # ... and it is not the trainer's checkpoint for that iteration.
    assert by_role[(SELF_PLAY, latest)] != by_role[(TRAIN, latest)]

    # (b) the trainer's published net is <root>/ckpt's newest, and the EMA is
    #     a different net -- so nothing trained *from* the smoothed weights.
    assert by_role[(TRAIN, latest)] == train_digest
    assert by_role[(EMA, latest)] == ema_digest
    assert train_digest != ema_digest, (
        "the trainer's newest checkpoint and the EMA are bit-identical after "
        "smoothing has begun: the two bindings are one"
    )
    assert ema_state.optim_state == {}, (
        "the EMA file carries optimiser state; nothing may resume from it"
    )


OBSERVED_ITERATIONS: Final[int] = 3
"""Iterations the observed campaign runs, and three is the smallest that works.

``EMA_0 := theta_0`` **bit-identically** (§2), so at iteration 1 the newest
train checkpoint and the newest EMA are the same net and "the trainer did not
start from the EMA" is not yet a distinguishable claim. Smoothing first bites
at iteration 2, which is therefore the first iteration at which mutant M12 --
the trainer loading ``serving.model_state`` -- is observable at all.
"""


@dataclass(frozen=True)
class _TrainStepObservation:
    """What was true at the moment one iteration's ``TrainStep`` was built."""

    iteration: int
    net_sha256: str
    """The digest of the net **handed to the trainer**, at construction."""
    ckpt_sha256: str
    """The newest ``<root>/ckpt`` checkpoint at that moment; ``""`` if none."""
    ema_sha256: str
    """The newest ``<root>/ema`` checkpoint at that moment; ``""`` if none."""
    checkpoint_env: str
    """``os.environ[CHECKPOINT_ENV]`` as the spawned servers would read it."""
    served_path: str
    """The path of the newest EMA -- what the loop should have published."""


@dataclass(frozen=True)
class _ObservedCampaign:
    """A campaign run with a recording ``TrainStep``, and what it recorded."""

    root: Path
    run: rig.WalledRun
    observations: tuple[_TrainStepObservation, ...]


@pytest.fixture(scope="module")
def observed(tmp_path_factory: pytest.TempPathFactory) -> _ObservedCampaign:
    """Its own campaign, because the observation has to be installed first.

    The shared campaign has already run by the time any test can patch
    anything, and what acceptance #28(b) is about happens *inside* an iteration
    -- so this pays for a second (three-iteration) campaign rather than
    asserting over the artefacts of one that was never watched.

    A real :class:`~komugi.train.step.TrainStep` subclass, not a stand-in:
    ``training_iteration`` checks ``step.net is net`` and the campaign has to
    actually train, so the only thing added is one digest taken at
    construction.
    """
    root = Path(tmp_path_factory.mktemp("observed"))
    loop = rig.campaign_loop()
    seen: list[_TrainStepObservation] = []

    class _RecordingTrainStep(TrainStep):
        """Digest the net the loop handed the trainer, then be a TrainStep."""

        def __init__(self, net: torch.nn.Module, *args: Any, **kwargs: Any) -> None:
            """Record the net, the two series and the environment, then delegate."""
            keys = parameter_keys(net)
            train = load_checkpoint(loop.checkpoint_dir(root))
            ema = load_checkpoint(loop.ema_dir(root))
            seen.append(
                _TrainStepObservation(
                    iteration=len(seen),
                    net_sha256=canonical_digest(net.state_dict(), keys).sha256,
                    ckpt_sha256=(
                        canonical_digest(train.model_state, keys).sha256
                        if train is not None
                        else ""
                    ),
                    ema_sha256=(
                        canonical_digest(ema.model_state, keys).sha256
                        if ema is not None
                        else ""
                    ),
                    checkpoint_env=os.environ.get(rig.CHECKPOINT_ENV, ""),
                    served_path=str(ema.path) if ema is not None else "",
                )
            )
            super().__init__(net, *args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(loop, "TrainStep", _RecordingTrainStep)
        run = rig.walled_run_campaign(
            root,
            iterations=OBSERVED_ITERATIONS,
            wall_s=rig.CAMPAIGN_WALL_S,
            label="observed",
        )
    return _ObservedCampaign(root=root, run=run, observations=tuple(seen))


def test_the_trainer_starts_each_iteration_from_the_ckpt_series(
    observed: _ObservedCampaign,
) -> None:
    """Acceptance #28(b), asserted where it can fail: on the trainer's own net.

    Everything the suite had was over the journal's fingerprints, and a trainer
    that resumed **from the EMA** satisfies all of it: ``self_play(i)`` is still
    ``ema(i-1)``, the published ``train`` digest is still whatever the trainer
    published, every series still advances, and nothing raises. Mutant M12 --
    ``net.load_state_dict(serving.model_state)`` with the optimiser load left on
    ``resume``, which is precisely the "implementer who guards the optimiser
    load" the design names -- was predicted to survive the entire suite.

    So this observes the net at the one place the claim is falsifiable: the
    module the loop hands to ``TrainStep``. For every iteration after the first
    its digest must be the newest ``<root>/ckpt`` checkpoint's, and from
    iteration 2 (where ``EMA_1 != theta_1``) it must **not** be the EMA's.

    Under M12 the assertion that fires is
    ``assert entry.net_sha256 == entry.ckpt_sha256``, at **iteration 2**: the
    trainer is handed ``EMA_1``, whose parameters differ from ``theta_1``'s.
    It cannot fire at iteration 1 -- ``EMA_0`` and ``theta_0`` are the same
    parameters by construction, which is why this campaign runs three
    iterations rather than two -- and the sibling assertion
    ``assert entry.net_sha256 != entry.ema_sha256`` fires at iteration 2 too,
    naming the failure the other way round.
    """
    observed.run.unwrap()
    entries = observed.observations
    assert len(entries) == OBSERVED_ITERATIONS, (
        f"the recording TrainStep was built {len(entries)} time(s) for "
        f"{OBSERVED_ITERATIONS} iterations: the loop did not rebuild the "
        f"trainer from a checkpoint every iteration, or the patch was not in "
        f"force"
    )

    assert entries[0].ckpt_sha256 == "", (
        "iteration 0 resumed from a checkpoint; there is none to resume from"
    )

    for entry in entries[1:]:
        assert entry.ckpt_sha256, f"iteration {entry.iteration} had no checkpoint"
        assert entry.net_sha256 == entry.ckpt_sha256, (
            f"iteration {entry.iteration} handed the trainer a net whose "
            f"parameters are not the newest train checkpoint's "
            f"({entry.net_sha256[:16]} vs {entry.ckpt_sha256[:16]}): the "
            f"trainer is descending from weights it did not publish"
        )

    smoothed = list(entries[2:])
    assert smoothed, "no iteration ran at which the EMA differs from the trainer"
    for entry in smoothed:
        # Non-vacuity first: from iteration 2 the two series really are two
        # different nets, so the inequality below is a claim and not a
        # tautology about EMA_0 := theta_0.
        assert entry.ckpt_sha256 != entry.ema_sha256, (
            f"at iteration {entry.iteration} the newest checkpoint and the "
            f"newest EMA are the same net, so 'the trainer did not start from "
            f"the EMA' cannot be observed here"
        )
        assert entry.net_sha256 != entry.ema_sha256, (
            f"iteration {entry.iteration} started its trainer from the EMA: "
            f"training then descends from a smoothed net and re-smooths it, "
            f"the recursion the band gate is derived from no longer describes "
            f"the system, and nothing raises"
        )


def test_the_environment_names_the_net_the_fingerprint_digests(
    observed: _ObservedCampaign,
) -> None:
    """Review F10 / mutant M13: the ``self_play`` digest is of the served file.

    The fingerprint used to be computed from the loop's local ``served``
    string rather than from ``CHECKPOINT_ENV``, so a loop that set the variable
    from another binding -- M13 sets it from ``resume`` -- wrote a journal
    naming an EMA no game was ever played with, and **no gate noticed**. The
    parent cannot see what a child actually loaded (§3a says so), but it can be
    made to digest the value it really exported, and that is the difference
    between "the two agree" and "the two are the same object".

    Two halves, both observed inside the run: the variable equals the newest
    EMA's path at the moment the iteration's trainer is built, and the
    ``self_play`` fingerprint in the journal is the digest of the file that
    variable named.

    Under M13 the first assertion, ``entry.checkpoint_env == entry.served_path``,
    fires at iteration 1: the environment holds ``<root>/ckpt``'s newest file
    and the EMA's path is under ``<root>/ema``.
    """
    observed.run.unwrap()
    root = observed.root
    loop = rig.campaign_loop()
    entries = observed.observations

    assert entries[0].checkpoint_env == "", (
        "iteration 0 exported a checkpoint; its workers must initialise freshly"
    )

    with rig.small_net_env():
        keys = parameter_keys(new_net())

    fingerprints = read_records(
        loop.journal_path(root), NetFingerprint, missing_ok=True
    ).records
    by_role = {
        (record.role, record.iteration): record.sha256 for record in fingerprints
    }

    for entry in entries[1:]:
        assert entry.checkpoint_env == entry.served_path, (
            f"iteration {entry.iteration} exported "
            f"{entry.checkpoint_env!r} while the newest EMA was "
            f"{entry.served_path!r}: the workers played a net the journal does "
            f"not describe"
        )
        assert str(loop.ema_dir(root)) in entry.checkpoint_env
        assert str(loop.checkpoint_dir(root)) not in entry.checkpoint_env

        served_file = load_checkpoint_file(Path(entry.checkpoint_env))
        assert served_file is not None, (
            f"{entry.checkpoint_env} does not verify: what self-play played "
            f"cannot be fingerprinted"
        )
        assert (
            canonical_digest(served_file.model_state, keys).sha256
            == by_role[(SELF_PLAY, entry.iteration)]
        ), (
            f"iteration {entry.iteration}'s self_play fingerprint is not the "
            f"digest of the file {rig.CHECKPOINT_ENV} named"
        )


def test_the_campaign_writes_the_records_its_gates_read(
    campaign: rig.SharedCampaign,
) -> None:
    """Acceptances #23 and #31, on a real run rather than a planted root.

    Two ``IterationMetrics`` per iteration -- the trainer's and the campaign's,
    told apart by an explicit marker the trainer can never write -- and
    ``3N - 1`` fingerprints with no gaps.
    """
    results = campaign.results()
    root = campaign.root
    loop = rig.campaign_loop()
    journal = loop.journal_path(root)
    count = len(results)

    metrics = read_records(journal, IterationMetrics, missing_ok=True)
    assert metrics.skipped == 0, f"{metrics.skipped} unreadable journal line(s)"
    marked = [
        record
        for record in metrics.records
        if record.extras.get(CAMPAIGN_MARKER, 0.0) == 1.0
    ]
    assert len(marked) == count
    assert len(metrics.records) == 2 * count, (
        "the journal does not hold exactly the trainer's record and the "
        "campaign's for every iteration"
    )
    for record in marked:
        assert LOSS_TAIL_MEDIAN in record.extras
        assert "bn_digest_lo" in record.extras
        assert "bn_digest_advanced" in record.extras
        assert all(math.isfinite(value) for value in record.extras.values())

    fingerprints = read_records(journal, NetFingerprint, missing_ok=True).records
    assert len(fingerprints) == 3 * count - 1
    verdict = fingerprint_series_complete(fingerprints, count - 1)
    assert verdict.ok, verdict.reason
