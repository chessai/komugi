"""The EMA's arithmetic, its resume rule, and the digest that must be able to fail.

KOM-022 §2 and §3a. Three of the claims here are load-bearing and the rest
fence them:

* the recursion is the one the band gate's closed form is derived from
  (acceptance #1). If the published series is not
  ``EMA_i = d*EMA_{i-1} + (1-d)*theta_i`` then ``r* = (d/(1-d))**2`` describes
  a system that is not running, and §3b's band is calibrated to nothing;
* the *averaged* set and the *digested* set are deliberately different
  (acceptances #2, #27). The BatchNorm running statistics must be averaged --
  a net whose convolutions average four generations but whose normalisation is
  one generation's statistics is not any of the four -- and must **not** be
  digested, because ``net.train()`` advances them in the forward pass on a net
  that is not learning at all;
* the digest is therefore able to say "this trainer did not learn", which is
  the statement rev 1's and rev 2's field sets could not make.

**Acceptance #27's fixture, and why it is built this way.** The non-learning
series is a *published checkpoint series* -- built straight through
:func:`~komugi.train.checkpoint.save_checkpoint`, with no ``TrainStep``, no
``StepConfig``, no optimiser and no learning rate anywhere. Rev 3 of the design
specified the same property with an ``lr = 0.0`` trainer, and this repo refuses
to construct one: ``StepConfig.__post_init__`` and
``CampaignConfig.__post_init__`` both raise, and the guard is *correct* -- its
docstring states this ticket's own failure mode. So the fixture is the
trainer's **output** rather than the trainer: parameters bit-identical from
file to file (the optimiser is their only writer and both of AdamW's
parameter-writing terms scale by ``lr``), ``running_mean`` / ``running_var``
perturbed and ``num_batches_tracked`` advanced at every step (the forward pass
moves those whatever the optimiser does).

That fixture is *red* on rev 1's whole-state-dict digest and on rev 2's
float-only one -- both hash the perturbed running statistics, so the equality
below fails -- and green only on the parameters-only digest. It is not rev 1's
condemned #6 vacuity, where the buffers were identical too and the test could
not tell the three field sets apart: here they differ by construction.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Final

import pytest
import torch

from komugi.campaign.factory import NET_ENV, SMALL_NET, new_net
from komugi.train.checkpoint import (
    list_checkpoints,
    load_checkpoint_file,
    require_matching_plane_count,
    save_checkpoint,
)
from komugi.train.digest import (
    bn_digest,
    canonical_digest,
    digest_lo,
    parameter_keys,
)
from komugi.train.ema import (
    EMA_DECAY,
    EmaError,
    ema_meta,
    ema_state,
    publish_ema,
    reconcile_ema,
)

STEPS_PER_ITERATION: Final[int] = 8
"""Small, and only ever used as a step *stride*: nothing here runs a step."""

WEIGHT: Final[str] = "block.weight"
BIAS: Final[str] = "block.bias"
RUNNING_MEAN: Final[str] = "block.bn.running_mean"
RUNNING_VAR: Final[str] = "block.bn.running_var"
COUNTER: Final[str] = "block.bn.num_batches_tracked"

SYNTHETIC_PARAMS: Final[tuple[str, ...]] = (BIAS, WEIGHT)
"""Sorted, as :func:`~komugi.train.digest.parameter_keys` returns them."""


# ---------------------------------------------------------------------------
# a synthetic net-shaped state dict: two parameters, two float buffers, one
# integer counter. Small enough to publish six of, and it exercises every
# branch of the partition the real net exercises.
# ---------------------------------------------------------------------------


def _theta(index: int) -> dict[str, torch.Tensor]:
    """Return a state dict standing in for ``theta_<index>``.

    Every entry moves with `index`, so an averaging bug that dropped a key --
    or copied one it should have interpolated -- shows up as a wrong number
    rather than as a wrong shape.
    """
    value = float(index)
    return {
        WEIGHT: torch.full((2, 3), value),
        BIAS: torch.full((3,), value * 2.0),
        RUNNING_MEAN: torch.full((3,), value * 0.5),
        RUNNING_VAR: torch.full((3,), 1.0 + value),
        COUNTER: torch.tensor(index * STEPS_PER_ITERATION, dtype=torch.int64),
    }


def _publish_series(directory: Path, count: int) -> None:
    """Publish ``theta_0 .. theta_<count-1>`` as real checkpoints."""
    for index in range(count):
        save_checkpoint(
            directory,
            (index + 1) * STEPS_PER_ITERATION,
            _theta(index),
            {},
            {"iteration": index, "role": "train"},
        )


def _publish_one(directory: Path, index: int) -> Path:
    """Publish ``theta_<index>`` alone, at the step iteration `index` files at."""
    return save_checkpoint(
        directory,
        (index + 1) * STEPS_PER_ITERATION,
        _theta(index),
        {},
        {"iteration": index, "role": "train"},
    )


def _grow_series(checkpoints: Path, ema: Path, count: int) -> tuple[int, ...]:
    """Publish `count` iterations, reconciling the EMA after **each** one.

    One reconcile per publication, because that is what the loop does and
    because §2's resume rule *refuses* the alternative: an empty EMA directory
    beside more than one train checkpoint is a pre-KOM-022 root, and
    backfilling it would manufacture an EMA history that never served a game.
    """
    published: list[int] = []
    for index in range(count):
        _publish_one(checkpoints, index)
        published.extend(
            reconcile_ema(
                checkpoints,
                ema,
                SYNTHETIC_PARAMS,
                steps_per_iteration=STEPS_PER_ITERATION,
            )
        )
    return tuple(published)


def _newest_state(directory: Path) -> dict[str, torch.Tensor]:
    """Return the newest checkpoint's weights, failing if there is none."""
    refs = list_checkpoints(directory)
    assert refs, f"{directory} holds no checkpoint"
    loaded = load_checkpoint_file(refs[0].path)
    assert loaded is not None, f"{refs[0].path} does not verify"
    return loaded.model_state


# ---------------------------------------------------------------------------
# A. the recursion (#1) and the partition (#2)
# ---------------------------------------------------------------------------


def test_recursion_matches_closed_form(tmp_path: Path) -> None:
    """Acceptance #1: ``EMA_5`` is the recursion's closed form, elementwise.

    ``EMA_0 := theta_0`` and ``EMA_i = d*EMA_{i-1} + (1-d)*theta_i``, so after
    five updates ``EMA_5 = d**5 * theta_0 + sum_k (1-d) d**k theta_{5-k}``.
    This is the identity §3b's ``r* = (d/(1-d))**2`` is derived from: if the
    published series is not this one, the band is calibrated to a system that
    is not running.
    """
    checkpoints = tmp_path / "ckpt"
    ema = tmp_path / "ema"

    published = _grow_series(checkpoints, ema, 6)

    assert published == tuple(
        (index + 1) * STEPS_PER_ITERATION for index in range(6)
    ), "every train checkpoint should have produced exactly one EMA update"

    decay = EMA_DECAY
    expected = decay**5 * _theta(0)[WEIGHT]
    for k in range(5):
        expected = expected + (1.0 - decay) * decay**k * _theta(5 - k)[WEIGHT]

    got = _newest_state(ema)[WEIGHT]
    assert torch.allclose(got, expected, atol=1e-6), (
        f"EMA_5 weight is {got.flatten()[0].item()}, the closed form says "
        f"{expected.flatten()[0].item()}"
    )


def test_buffers_are_averaged_and_counters_copied(tmp_path: Path) -> None:
    """Acceptance #2: running statistics are averaged, counters are copied.

    The classic EMA bug is averaging the parameters and leaving the buffers
    stale, and its mirror image is averaging the *counter* -- an
    exponentially-averaged counter counts nothing. So: ``running_mean`` and
    ``running_var`` must differ from ``theta_i``'s (they were smoothed), and
    ``num_batches_tracked`` must equal it **exactly**.
    """
    checkpoints = tmp_path / "ckpt"
    ema = tmp_path / "ema"
    _grow_series(checkpoints, ema, 4)

    averaged = _newest_state(ema)
    newest_theta = _theta(3)

    assert not torch.equal(averaged[RUNNING_MEAN], newest_theta[RUNNING_MEAN]), (
        "running_mean was copied rather than averaged: a net whose convolutions "
        "average four generations, normalised by one generation's statistics, "
        "is not any of the four"
    )
    assert not torch.equal(averaged[RUNNING_VAR], newest_theta[RUNNING_VAR])
    assert torch.equal(averaged[COUNTER], newest_theta[COUNTER]), (
        "num_batches_tracked was averaged; an exponentially-averaged counter "
        "is not a counter"
    )
    assert averaged[COUNTER].dtype == torch.int64


def test_the_ema_update_does_not_alias_its_inputs() -> None:
    """§3b guard 3's bug, at the arithmetic: an EMA must not *be* the trainer.

    An update that bound ``theta``'s tensor by reference would agree with the
    trainer exactly on every probe position, which is what the ``KL == 0.0``
    aliasing check pauses on. Cheaper to refuse here than to detect there.
    """
    previous = _theta(0)
    theta = _theta(1)
    updated = ema_state(previous, theta)

    for key, tensor in updated.items():
        assert tensor.data_ptr() != theta[key].data_ptr(), f"{key} aliases theta"
        assert tensor.data_ptr() != previous[key].data_ptr(), (
            f"{key} aliases the previous EMA"
        )
    assert torch.equal(previous[WEIGHT], _theta(0)[WEIGHT]), (
        "the update mutated its own input in place"
    )


def test_the_ema_arithmetic_runs_on_cpu_float32() -> None:
    """§2, pinned: the update is CPU arithmetic, whatever the caller holds.

    The unconditional half of acceptance #34. Production runs
    ``device='cuda'``, and the recomputability the crash story rests on is only
    bit-reproducible if the arithmetic is the same arithmetic on both devices;
    reading the inputs back through the loader (which maps to the CPU) is what
    delivers that, and this pins the output side of it.
    """
    updated = ema_state(_theta(0), _theta(1))

    assert all(tensor.device.type == "cpu" for tensor in updated.values())
    assert updated[WEIGHT].dtype == torch.float32


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the cuda half of #34 needs a GPU"
)
def test_the_ema_arithmetic_is_the_same_on_cuda_and_cpu() -> None:
    """Acceptance #34's **skipped** half: the update is device-independent.

    Production runs ``device='cuda'``, and the crash story rests on the EMA
    being *recomputable*: a resumed campaign replays the missed updates and
    must land on the same bytes the interrupted one would have published. That
    is only true if the arithmetic does not depend on where the trainer's
    tensors happened to live, and no test asserted it -- there was no
    ``torch.cuda`` reference anywhere under ``tests/campaign`` or
    ``tests/train`` (review F7).

    ``ema_state`` moves both inputs to CPU float32 before it interpolates, so
    the claim is bit-equality rather than a tolerance. A tolerance here would
    pass for an implementation that interpolated on the device and copied the
    result back, which is the version that makes a GPU replay produce different
    bytes from a CPU one.
    """
    previous = _theta(0)
    theta = _theta(1)
    on_cpu = ema_state(previous, theta)
    on_cuda = ema_state(
        {key: value.to("cuda") for key, value in previous.items()},
        {key: value.to("cuda") for key, value in theta.items()},
    )

    assert set(on_cuda) == set(on_cpu)
    for key, value in on_cuda.items():
        assert value.device.type == "cpu", (
            f"{key} came back on {value.device}: the published EMA must be CPU "
            f"tensors whatever the caller held"
        )
        assert torch.equal(value, on_cpu[key]), (
            f"{key} differs between a cuda-resident and a cpu-resident input: "
            f"a resumed campaign would replay to different bytes"
        )


def test_a_key_mismatch_is_refused_rather_than_unioned() -> None:
    """Two different nets must not be averaged into a third that is neither."""
    theta = _theta(1)
    del theta[BIAS]
    with pytest.raises(ValueError, match="disagree on their keys"):
        ema_state(_theta(0), theta)


@pytest.mark.parametrize("decay", [-0.1, 1.0, 1.5])
def test_a_decay_outside_the_unit_interval_is_refused(decay: float) -> None:
    """``d = 1`` is a frozen EMA and ``d > 1`` is divergence; both are refused."""
    with pytest.raises(ValueError, match="decay"):
        ema_state(_theta(0), _theta(1), decay)


# ---------------------------------------------------------------------------
# B. the resume rule (#4, #5) and the fresh-roots-only refusal
# ---------------------------------------------------------------------------


def test_ema_step_tracks_ckpt_step(tmp_path: Path) -> None:
    """Acceptance #5: after any completed iteration the two newest steps agree."""
    checkpoints = tmp_path / "ckpt"
    ema = tmp_path / "ema"
    for index in range(3):
        _publish_one(checkpoints, index)
        reconcile_ema(
            checkpoints,
            ema,
            SYNTHETIC_PARAMS,
            steps_per_iteration=STEPS_PER_ITERATION,
        )
        assert list_checkpoints(ema)[0].step == list_checkpoints(checkpoints)[0].step


def test_a_replay_after_a_kill_reproduces_the_uninterrupted_bytes(
    tmp_path: Path,
) -> None:
    """Acceptance #3's arithmetic half: the EMA is recomputable, not atomic.

    A kill between the train checkpoint's publication and the EMA's leaves the
    EMA behind by k; replaying k updates in ascending step order from train
    checkpoints that all coexist must land on the *same digest* as a run that
    was never interrupted. The kill itself belongs to the behavioural suite;
    what is pinned here is that the replay is deterministic arithmetic rather
    than a best effort.
    """
    interrupted = tmp_path / "killed"
    uninterrupted = tmp_path / "clean"

    # The uninterrupted run: an EMA update after every publication.
    for count in range(1, 5):
        _publish_series(uninterrupted / "ckpt", count)
        reconcile_ema(
            uninterrupted / "ckpt",
            uninterrupted / "ema",
            SYNTHETIC_PARAMS,
            steps_per_iteration=STEPS_PER_ITERATION,
        )

    # The killed run: iteration 0 published its EMA, then the process died
    # between each later train checkpoint and its EMA, so the EMA is three
    # updates behind on the resume and replays all three at once.
    _publish_series(interrupted / "ckpt", 1)
    reconcile_ema(
        interrupted / "ckpt",
        interrupted / "ema",
        SYNTHETIC_PARAMS,
        steps_per_iteration=STEPS_PER_ITERATION,
    )
    _publish_series(interrupted / "ckpt", 4)
    replayed_steps = reconcile_ema(
        interrupted / "ckpt",
        interrupted / "ema",
        SYNTHETIC_PARAMS,
        steps_per_iteration=STEPS_PER_ITERATION,
    )
    assert len(replayed_steps) == 3, "the resume did not replay what the kill cost"

    replayed = canonical_digest(
        _newest_state(interrupted / "ema"), SYNTHETIC_PARAMS
    )
    steady = canonical_digest(
        _newest_state(uninterrupted / "ema"), SYNTHETIC_PARAMS
    )
    assert replayed.sha256 == steady.sha256


def test_reconcile_is_idempotent(tmp_path: Path) -> None:
    """Running the replay twice publishes nothing the second time."""
    checkpoints = tmp_path / "ckpt"
    ema = tmp_path / "ema"

    first = _grow_series(checkpoints, ema, 3)
    second = reconcile_ema(
        checkpoints, ema, SYNTHETIC_PARAMS, steps_per_iteration=STEPS_PER_ITERATION
    )

    assert len(first) == 3
    assert second == ()


def test_ema_ahead_is_refused(tmp_path: Path) -> None:
    """Acceptance #4: an EMA newer than the trainer means the invariant broke.

    The EMA is published *after* its source, so "EMA ahead" cannot happen in
    one campaign. It can happen when two campaigns share a root, which is the
    disclosed gap §11.2 names -- and starting anyway would average over a
    history this process did not produce.
    """
    checkpoints = tmp_path / "ckpt"
    ema = tmp_path / "ema"
    _publish_series(checkpoints, 1)
    publish_ema(
        ema,
        99 * STEPS_PER_ITERATION,
        _theta(0),
        ema_meta(iteration=98, source_step=0, train_sha256="x", ema_sha256="y"),
    )

    with pytest.raises(EmaError, match="ahead of the trainer"):
        reconcile_ema(
            checkpoints,
            ema,
            SYNTHETIC_PARAMS,
            steps_per_iteration=STEPS_PER_ITERATION,
        )


def test_an_existing_root_is_refused_rather_than_backfilled(tmp_path: Path) -> None:
    """§2's fresh-roots-only rule: backfilling would fabricate a liveness record.

    An empty EMA directory beside *more than one* train checkpoint is a
    pre-KOM-022 root. Replaying it would publish an EMA history that never
    served a single game, and the fingerprints written beside it would be a
    liveness record of iterations nothing observed -- the one artefact this
    ticket must never manufacture.
    """
    checkpoints = tmp_path / "ckpt"
    _publish_series(checkpoints, 3)

    with pytest.raises(EmaError, match="pre-KOM-022 root"):
        reconcile_ema(
            checkpoints,
            tmp_path / "ema",
            SYNTHETIC_PARAMS,
            steps_per_iteration=STEPS_PER_ITERATION,
        )


def test_one_checkpoint_and_an_empty_ema_is_a_fresh_root(tmp_path: Path) -> None:
    """The boundary of the rule above: exactly one checkpoint is iteration 0."""
    checkpoints = tmp_path / "ckpt"
    ema = tmp_path / "ema"
    _publish_series(checkpoints, 1)

    published = reconcile_ema(
        checkpoints, ema, SYNTHETIC_PARAMS, steps_per_iteration=STEPS_PER_ITERATION
    )

    assert published == (STEPS_PER_ITERATION,)
    assert torch.equal(_newest_state(ema)[WEIGHT], _theta(0)[WEIGHT]), (
        "EMA_0 must be theta_0 exactly, so iteration 1 serves bit-identical "
        "weights to the pre-EMA campaign's and smoothing first bites at 2"
    )


def test_the_published_ema_carries_both_digests_in_its_meta(tmp_path: Path) -> None:
    """§2's ``meta``: the source's parameter digest and the EMA's, side by side."""
    checkpoints = tmp_path / "ckpt"
    ema = tmp_path / "ema"
    _grow_series(checkpoints, ema, 2)

    newest = load_checkpoint_file(list_checkpoints(ema)[0].path)
    assert newest is not None
    assert newest.meta["role"] == "ema"
    assert newest.meta["decay"] == EMA_DECAY
    assert newest.meta["train_sha256"] == canonical_digest(
        _theta(1), SYNTHETIC_PARAMS
    ).sha256
    assert newest.meta["ema_sha256"] == canonical_digest(
        newest.model_state, SYNTHETIC_PARAMS
    ).sha256
    assert newest.optim_state == {}, "an EMA has no optimiser to resume from"


def test_a_published_ema_is_loadable_by_the_net_that_serves_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """KOM-002a: the file self-play serves must survive the plane-count guard.

    :func:`~komugi.train.ema.ema_meta` names five fields and none of them is
    ``in_planes``, so before the codec stamped the plane map every EMA this
    module published was refused by
    :func:`~komugi.train.checkpoint.require_matching_plane_count` -- the guard
    firing on a file written seconds earlier by this very process. That is not
    a cosmetic mismatch: the EMA is precisely the checkpoint
    ``factory.serving_net`` loads, so the refusal happened inside a spawned
    inference server and reached the campaign as ``inference server for
    incarnation 0 exited with code 1``.

    Asserted through the real load site rather than on the meta dict, because
    the meta is only interesting for what the loader does with it.
    """
    monkeypatch.setenv(NET_ENV, SMALL_NET)
    net = new_net()
    checkpoints = tmp_path / "ckpt"
    ema = tmp_path / "ema"
    _publish_series(checkpoints, 1)
    reconcile_ema(
        checkpoints, ema, SYNTHETIC_PARAMS, steps_per_iteration=STEPS_PER_ITERATION
    )

    served = load_checkpoint_file(list_checkpoints(ema)[0].path)
    assert served is not None
    require_matching_plane_count(served, net.config.trunk.in_planes)
    assert served.meta["role"] == "ema", "and it is still an EMA, not a rewrite"


# ---------------------------------------------------------------------------
# C. acceptance #27 -- the digest cannot move without learning
# ---------------------------------------------------------------------------


def _non_learning_series(
    directory: Path, net: torch.nn.Module, iterations: int
) -> list[Path]:
    """Publish what a **non-learning trainer** publishes, with no trainer.

    From one real net built by the campaign's own factory: snapshot
    ``state_dict()`` once, then publish `iterations` checkpoints in which
    **every tensor named by ``named_parameters()`` is the same tensor, byte for
    byte, in every file**, while at each ``i`` every ``running_mean`` /
    ``running_var`` is perturbed by ``i * 1e-3`` and every
    ``num_batches_tracked`` is advanced by ``STEPS_PER_ITERATION``.

    That is exactly the artefact the trainer's own contract describes: the
    optimiser is the parameters' only writer and both of its parameter-writing
    terms scale by ``lr``, while ``net.train()``'s **forward** advances the
    BatchNorm state whatever the optimiser does. No ``StepConfig``, no
    optimiser, no learning rate -- so no guard is relaxed, monkeypatched, or
    reached around.
    """
    snapshot = {key: value.detach().clone() for key, value in net.state_dict().items()}
    params = set(parameter_keys(net))
    paths: list[Path] = []
    for index in range(iterations):
        state: dict[str, torch.Tensor] = {}
        for key, value in snapshot.items():
            if key in params:
                # The same tensor object, deliberately: byte-for-byte identical
                # parameters are what "did not learn" means.
                state[key] = value
            elif value.is_floating_point():
                state[key] = value + float(index) * 1e-3
            else:
                state[key] = value + index * STEPS_PER_ITERATION
        paths.append(
            save_checkpoint(
                directory,
                (index + 1) * STEPS_PER_ITERATION,
                state,
                {},
                {"iteration": index, "role": "train"},
            )
        )
    return paths


def test_a_non_learning_trainer_does_not_advance_the_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Acceptance #27: the no-op direction, on the artefact the gate reads.

    Two published checkpoints of a non-learning trainer. **(a)**
    ``canonical_digest`` is *equal* across them -- an equality, not a
    tolerance, because the parameter bytes are identical by construction;
    **(b)** ``bn_digest`` differs, because the BatchNorm statistics moved;
    **(c)** ``num_batches_tracked`` advanced; **(d)** the two *filename*
    digests differ, because the checkpoint name covers ``meta_json`` and the
    meta carries ``iteration``.

    Red on rev 1's whole-state-dict field set and on rev 2's float-only one --
    both digest the perturbed ``running_*``, so (a) fails under either -- and
    green only on the parameters-only digest. (d) is the §1 trap: it is exactly
    why the filename digest is not the liveness hash.
    """
    monkeypatch.setenv(NET_ENV, SMALL_NET)
    net = new_net()
    params = parameter_keys(net)
    directory = tmp_path / "ckpt"

    published = _non_learning_series(directory, net, 2)
    first = load_checkpoint_file(published[0])
    second = load_checkpoint_file(published[1])
    assert first is not None and second is not None

    # (a) the liveness digest did not move.
    assert (
        canonical_digest(first.model_state, params).sha256
        == canonical_digest(second.model_state, params).sha256
    ), (
        "the parameters-only digest advanced across two checkpoints whose "
        "parameters are byte-identical: the hash gate cannot fail, which is "
        "the defect rev 3 narrowed the field set to fix"
    )
    assert (
        canonical_digest(first.model_state, params).scalars
        == canonical_digest(second.model_state, params).scalars
    )

    # (b) the BatchNorm series did move -- the two series are independent.
    assert (
        bn_digest(first.model_state, params).sha256
        != bn_digest(second.model_state, params).sha256
    ), "the fixture did not actually perturb the running statistics"

    # (c) the counters advanced, which is what a train-mode forward does.
    counters = [
        key for key in first.model_state if key.endswith("num_batches_tracked")
    ]
    assert counters, "the net grew no BatchNorm counter; the fixture is vacuous"
    for key in counters:
        assert int(second.model_state[key].item()) == int(
            first.model_state[key].item()
        ) + STEPS_PER_ITERATION

    # (d) and the filename digests differ anyway, which is the §1 trap.
    assert published[0].name != published[1].name
    names = {ref.step: ref.digest for ref in list_checkpoints(directory)}
    assert names[STEPS_PER_ITERATION] != names[2 * STEPS_PER_ITERATION], (
        "the checkpoint filename digest covers meta_json, which covers "
        "iteration, so it advances for a completely frozen net -- using it as "
        "the liveness hash is the shortcut §3 refuses"
    )


def test_the_flat_series_produces_a_flat_ema_series(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The EMA of a flat-parameter series is flat too, by the arithmetic.

    ``EMA_i``'s parameters are a convex combination of parameters that are all
    the same tensor, so they are that tensor. This is what makes acceptance
    #41's *whole* fingerprint series flat -- ``train``, ``ema`` and
    ``self_play`` alike -- from one construction.
    """
    monkeypatch.setenv(NET_ENV, SMALL_NET)
    net = new_net()
    params = parameter_keys(net)
    checkpoints = tmp_path / "ckpt"
    ema = tmp_path / "ema"

    # One reconcile per publication, exactly as the loop does it: an empty EMA
    # directory beside more than one train checkpoint is the pre-KOM-022 root
    # the resume rule refuses, and refusing it is the point.
    for count in range(1, 4):
        _non_learning_series(checkpoints, net, count)
        reconcile_ema(
            checkpoints, ema, params, steps_per_iteration=STEPS_PER_ITERATION
        )

    digests = set()
    bn_digests = set()
    for ref in list_checkpoints(ema):
        loaded = load_checkpoint_file(ref.path)
        assert loaded is not None
        digests.add(canonical_digest(loaded.model_state, params).sha256)
        bn_digests.add(bn_digest(loaded.model_state, params).sha256)

    assert len(digests) == 1, "the EMA's parameters moved without the trainer's"
    assert len(bn_digests) == 3, (
        "the EMA's BatchNorm statistics did not move, so the two series would "
        "not be observably independent on this root"
    )


# ---------------------------------------------------------------------------
# D. the digest's own contract
# ---------------------------------------------------------------------------


def test_a_state_dict_missing_a_parameter_raises(tmp_path: Path) -> None:
    """Never digest a subset: a narrowed gate is worse than a failing one."""
    state = _theta(0)
    del state[WEIGHT]
    with pytest.raises(KeyError, match="missing"):
        canonical_digest(state, SYNTHETIC_PARAMS)


def test_the_digest_ignores_key_order() -> None:
    """Dict ordering is not a contract; the walk sorts, so the hash is stable."""
    forward = _theta(0)
    reversed_order = {key: forward[key] for key in reversed(list(forward))}

    assert (
        canonical_digest(forward, SYNTHETIC_PARAMS).sha256
        == canonical_digest(reversed_order, SYNTHETIC_PARAMS).sha256
    )


def test_the_digest_separates_name_dtype_and_shape() -> None:
    """A reshape that preserves the buffer must not preserve the digest."""
    base = _theta(0)
    reshaped = dict(base)
    reshaped[WEIGHT] = base[WEIGHT].reshape(3, 2)

    assert (
        canonical_digest(base, SYNTHETIC_PARAMS).sha256
        != canonical_digest(reshaped, SYNTHETIC_PARAMS).sha256
    )


def test_the_param_count_is_over_the_restricted_set() -> None:
    """``param_count`` counts parameters, not counters and not statistics."""
    digest = canonical_digest(_theta(0), SYNTHETIC_PARAMS)
    assert digest.scalars == 2 * 3 + 3


def test_bn_digest_holds_the_float_buffers_and_no_parameter() -> None:
    """The separate series: floating-point non-parameters, nothing else."""
    buffers = bn_digest(_theta(0), SYNTHETIC_PARAMS)
    assert buffers.scalars == 3 + 3, (
        "bn_digest should cover running_mean and running_var and neither the "
        "parameters nor the int64 counter"
    )

    moved = _theta(0)
    moved[RUNNING_VAR] = moved[RUNNING_VAR] + 1.0
    assert bn_digest(moved, SYNTHETIC_PARAMS).sha256 != buffers.sha256

    frozen_buffers = _theta(0)
    frozen_buffers[WEIGHT] = frozen_buffers[WEIGHT] + 1.0
    assert bn_digest(frozen_buffers, SYNTHETIC_PARAMS).sha256 == buffers.sha256, (
        "a parameter change moved the BatchNorm series, so the two series are "
        "not independent"
    )


def test_digest_lo_is_a_lossless_finite_float() -> None:
    """52 bits survive the trip through ``extras``, which must be finite floats."""
    value = digest_lo("f" * 64)
    assert math.isfinite(value)
    assert value == float(int(value)), "the low bits did not survive as an integer"
    assert value < 2.0**53
    assert digest_lo("0" * 64) == 0.0


# ---------------------------------------------------------------------------
# F. the KOM-036 arm on the SERVED artefact (#31, the EMA half)
# ---------------------------------------------------------------------------

ARM_KEY: Final[str] = "fast_policy_weight"
"""The meta key both writers stamp, so an arm cannot be misattributed."""

_BASE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "iteration",
        "role",
        "decay",
        "source_step",
        "train_sha256",
        "ema_sha256",
    }
)
"""``ema_meta``'s six pre-KOM-036 keys, which the arm joins and never replaces."""


def _publish_armed(directory: Path, index: int, arm: float | None) -> Path:
    """Publish ``theta_<index>`` as a trainer checkpoint carrying `arm`.

    ``None`` writes the pre-stamp meta, which is what a checkpoint from before
    KOM-036 looks like on disk.
    """
    meta: dict[str, object] = {"iteration": index, "role": "train"}
    if arm is not None:
        meta[ARM_KEY] = arm
    return save_checkpoint(
        directory, (index + 1) * STEPS_PER_ITERATION, _theta(index), {}, meta
    )


@pytest.mark.parametrize("arm", [0.0, 0.125, 1.0])
def test_the_ema_meta_round_trips_the_arm(tmp_path: Path, arm: float) -> None:
    """#31, EMA half: ``ema_meta`` carries the arm, and the codec keeps it.

    Red before the field existed: ``ema_meta`` was a fixed six-key dict, so
    every *served* checkpoint was unattributable while the trainer-only node
    went green -- which is precisely the misattribution review M12 named.

    Arm ``0.0`` is in the sweep because it is falsy: ``if fast_policy_weight:``
    would drop the discard arm and keep the other two, and would do it silently.
    """
    meta = ema_meta(
        iteration=3,
        source_step=4 * STEPS_PER_ITERATION,
        train_sha256="a" * 64,
        ema_sha256="b" * 64,
        fast_policy_weight=arm,
    )
    assert set(meta) == _BASE_KEYS | {ARM_KEY}, (
        "the arm must JOIN the existing keys, not replace or reorder them"
    )
    assert meta[ARM_KEY] == arm

    published = publish_ema(tmp_path, 4 * STEPS_PER_ITERATION, _theta(4), meta)
    assert published.exists()
    loaded = load_checkpoint_file(published)
    assert loaded is not None, "the published EMA does not verify"
    assert loaded.meta[ARM_KEY] == arm, (
        "the arm did not survive the checkpoint codec, so a served net copied "
        "out of a run directory still cannot name the arm that produced it"
    )


def test_an_absent_arm_is_omitted_rather_than_stamped_as_none() -> None:
    """Absent means the key is missing, never present-and-``None``.

    A reader has to be able to tell "no arm recorded" from "an arm recorded as
    nothing", and the default exists only so the three pre-existing call sites
    keep working -- not so that they start claiming an arm.
    """
    meta = ema_meta(
        iteration=0, source_step=0, train_sha256="x" * 64, ema_sha256="y" * 64
    )
    assert set(meta) == _BASE_KEYS
    assert ARM_KEY not in meta


@pytest.mark.parametrize("arm", [0.0, 0.125, 1.0])
def test_reconcile_stamps_the_source_checkpoints_own_arm(
    tmp_path: Path, arm: float
) -> None:
    """The trainer's stamp and the EMA's are provably the SAME value.

    ``reconcile_ema`` is ``ema_meta``/``publish_ema``'s only production caller
    and it has no ``StepConfig`` and no ``CampaignConfig`` in scope, so the arm
    is read off the source trainer checkpoint's own ``meta`` -- the same way the
    iteration is. That is what makes the two stamps one claim rather than two
    that could disagree; threading a second copy down would reintroduce exactly
    the misattribution the stamp exists to prevent.
    """
    checkpoints = tmp_path / "ckpt"
    ema = tmp_path / "ema"
    _publish_armed(checkpoints, 0, arm)

    published = reconcile_ema(
        checkpoints, ema, SYNTHETIC_PARAMS, steps_per_iteration=STEPS_PER_ITERATION
    )
    assert published == (STEPS_PER_ITERATION,)

    refs = list_checkpoints(ema)
    assert refs
    loaded = load_checkpoint_file(refs[0].path)
    assert loaded is not None
    assert loaded.meta[ARM_KEY] == arm


def test_an_unstamped_source_publishes_an_unstamped_ema(tmp_path: Path) -> None:
    """Absent stays absent across the reconcile, too.

    A checkpoint written before the stamp existed carries no key, and the EMA
    rebuilt from it must not invent one -- which is the non-vacuity half of the
    node above: it shows the stamp is read from the source rather than
    defaulted to something.
    """
    checkpoints = tmp_path / "ckpt"
    ema = tmp_path / "ema"
    _publish_armed(checkpoints, 0, None)

    assert reconcile_ema(
        checkpoints, ema, SYNTHETIC_PARAMS, steps_per_iteration=STEPS_PER_ITERATION
    ) == (STEPS_PER_ITERATION,)

    refs = list_checkpoints(ema)
    assert refs
    loaded = load_checkpoint_file(refs[0].path)
    assert loaded is not None
    assert ARM_KEY not in loaded.meta
