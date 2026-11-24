"""The campaign loop's in-process half: config, factory, step counter, resume.

Nothing here spawns a process or plays a game -- the behavioural suite owns
:func:`~komugi.selfplay.orchestrate.run_selfplay` and everything that follows
from it. What is left is the part of a campaign that can be wrong while every
component is right, which is most of what the wave-10 review recorded as
obligations:

* **the global checkpoint step** (review F5). Two training iterations into one
  directory used to publish under the same step number, and
  :func:`~komugi.train.checkpoint.save_checkpoint`'s same-step supersede then
  *deleted* the older one -- one file survived a whole campaign and rollback
  was impossible. ``test_two_iterations_into_one_directory_coexist`` fails on
  the pre-``global_step_base`` trainer.
* **resume detection**, which is what makes a crash cost one iteration:
  :func:`~komugi.campaign.loop.resume_point` must find the newest checkpoint
  and name the iteration *after* it.
* **the telemetry wire-in** (KOM-025). Which record the boundary reads the
  trainer's ``extras`` back off -- the marker is the discriminator, and the
  *last* record for a re-run iteration is the one describing disk -- and the
  EMA advance gate's decision boundary, both from literals, with no campaign
  anywhere near them.
* **the factory's environment channel**. The orchestrator refuses a closure
  (design D2/F18), so "which checkpoint does this iteration serve" travels
  through :data:`~komugi.campaign.factory.CHECKPOINT_ENV`. If that link is
  broken, self-play silently serves freshly initialised weights forever and
  the campaign's loss curve looks like a hard problem rather than a bug.

The nets are the smallest real :class:`~komugi.net.model.KomugiNet` the thing
under test admits: ``_tiny_net`` for the trainer (4 planes, 81 actions, the
``tests/train`` pattern) and the factory's own b2c16 for the factory, whose
seam shapes are the production ones on purpose.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import os
import textwrap
from pathlib import Path
from collections.abc import Mapping
from typing import Final

import pytest
import torch
from torch import nn

from komugi.campaign.factory import (
    CHECKPOINT_ENV,
    CONFIG,
    DEVICE_ENV,
    NET_ENV,
    SMALL_CONFIG,
    SMALL_NET,
    net_config,
    serving_net,
    small_serving_net,
)
from komugi.campaign import boundary
from komugi.campaign import loop as campaign_loop
from komugi.campaign.game_stats import GameStats, game_stats
from komugi.campaign import pinned
from komugi.campaign.pause import Mute
from komugi.campaign.telemetry import selfplay_journal_path
from komugi.telemetry.journal import read_records, write_record
from komugi.telemetry.records import IterationMetrics
from komugi.campaign.loop import (
    CampaignConfig,
    WindowSource,
    campaign_batch_fit,
    checkpoint_dir,
    iteration_seed,
    resume_point,
    shard_root,
    window_refs,
)
from komugi.config import SelfPlayConfig
from komugi.engine.game import Game
from komugi.engine.observation import observation_size
from komugi.modes import MODE_FULL
from komugi.selfplay.inference import InferenceConfig
from komugi.net import HeadsConfig, KomugiNet, NetConfig, NetOutput, TrunkConfig
from komugi.selfplay.records import ConcatSource
from komugi.terminations import TERMINATION_DECISIVE
from komugi.train.checkpoint import list_checkpoints, load_checkpoint, save_checkpoint
from komugi.train.digest import parameter_keys
from komugi.train import losses as train_losses
from komugi.train.losses import LossAssembler
from komugi.train.step import (
    TOTAL_KEY,
    StepConfig,
    TrainStep,
    _METRIC_PREFIX,
    training_iteration,
)
from komugi.train.window import IterationRef, RecencyWindow, SampleSource, WindowConfig
from komugi.train.wiring import P0A_RECIPE, P0B_RECIPE

_PLANES: Final[int] = 4
_ACTIONS: Final[int] = 81
_ROWS: Final[int] = 6
_STEPS: Final[int] = 2


def _tiny_net(seed: int = 3) -> KomugiNet:
    """A one-block, 81-action net -- the ``tests/train`` trainer fixture."""
    trunk = TrunkConfig(
        in_planes=_PLANES, channels=8, blocks=1, gpool_every=3, seed=seed
    )
    heads = HeadsConfig(channels=8, num_actions=_ACTIONS, policy_planes=1, seed=seed)
    return KomugiNet(NetConfig(trunk=trunk, heads=heads))


def _sample(row: int) -> dict[str, torch.Tensor]:
    """One schema-shaped row, a deterministic function of `row`."""
    generator = torch.Generator().manual_seed(500 + row)
    policy = torch.rand(_ACTIONS, generator=generator) + 0.05
    wdl = torch.zeros(3)
    wdl[row % 3] = 1.0
    return {
        "observation": torch.rand(
            (_PLANES, 9, 9), generator=generator, dtype=torch.float32
        ),
        "policy_target": policy / policy.sum(),
        "wdl_target": wdl,
        # `TrainStep.run` requires the fast-row stratum column at every arm
        # (KOM-036 §1.5); all-full rows keep the multiplier at exactly ones.
        "search_mode": torch.tensor(MODE_FULL, dtype=torch.int64),
        # ... and the cap-row stratum on the same terms (KOM-160); all-decisive
        # rows keep that multiplier at exactly ones too.
        "termination": torch.tensor(TERMINATION_DECISIVE, dtype=torch.int64),
        # ... and the two columns the moves-left cap-row weight reconstructs a
        # game's length from (KOM-159 D4), required on the same terms again.
        # Decisive rows are never censored, whatever the plies say.
        "ply": torch.tensor(row, dtype=torch.int64),
        "mlh_target": torch.tensor(0.25, dtype=torch.float32),
    }


class _ListSource:
    """An in-memory iteration: the smallest thing satisfying ``SampleSource``."""

    def __init__(self, values: tuple[int, ...]) -> None:
        self._values = values

    def __len__(self) -> int:
        """Return the number of samples held."""
        return len(self._values)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return a one-field sample carrying its own identifying value."""
        return {"observation": torch.tensor(float(self._values[index]))}


def _step(net: KomugiNet) -> TrainStep:
    """Bind `net` to the P0a recipe and a two-row step."""
    return TrainStep(
        net,
        LossAssembler(P0A_RECIPE),
        StepConfig(batch_size=2, lr=1e-3, seed=1),
    )


def _campaign(
    root: Path,
    *,
    iterations: int = 3,
    run_seed: int = 11,
    steps_per_iteration: int = 4,
    batch_size: int = 2,
    lr: float = 1e-3,
    workers: int = 1,
    games_per_worker: int = 1,
    device: str = "cpu",
) -> CampaignConfig:
    """A minimal valid campaign rooted at `root`, one field at a time varied."""
    return CampaignConfig(
        iterations=iterations,
        run_seed=run_seed,
        root=str(root),
        steps_per_iteration=steps_per_iteration,
        batch_size=batch_size,
        lr=lr,
        workers=workers,
        games_per_worker=games_per_worker,
        device=device,
    )


# --------------------------------------------------------------------------
# CampaignConfig
# --------------------------------------------------------------------------


def test_a_minimal_campaign_config_is_accepted(tmp_path: Path) -> None:
    """The defaults that matter: CPU, and every count as given."""
    campaign = _campaign(tmp_path)
    assert campaign.device == "cpu"
    assert campaign.steps_per_iteration == 4
    assert campaign.root == str(tmp_path)


@pytest.mark.parametrize(
    "field",
    ["iterations", "steps_per_iteration", "batch_size", "workers", "games_per_worker"],
)
def test_a_non_positive_count_is_rejected(tmp_path: Path, field: str) -> None:
    """Every count is a count of work; zero of any of them is not a campaign."""
    counts = {
        "iterations": 3,
        "steps_per_iteration": 4,
        "batch_size": 2,
        "workers": 1,
        "games_per_worker": 1,
    }
    counts[field] = 0
    with pytest.raises(ValueError, match=field):
        _campaign(
            tmp_path,
            iterations=counts["iterations"],
            steps_per_iteration=counts["steps_per_iteration"],
            batch_size=counts["batch_size"],
            workers=counts["workers"],
            games_per_worker=counts["games_per_worker"],
        )


def test_a_non_positive_learning_rate_is_rejected(tmp_path: Path) -> None:
    """A zero lr is a run that trains nothing while looking like one that does."""
    with pytest.raises(ValueError, match="lr"):
        _campaign(tmp_path, lr=0.0)


def test_a_nan_learning_rate_is_rejected(tmp_path: Path) -> None:
    """``not lr > 0`` rather than ``lr <= 0``, so NaN cannot slip through."""
    with pytest.raises(ValueError, match="lr"):
        _campaign(tmp_path, lr=float("nan"))


@pytest.mark.parametrize("lr", [0.0, -1e-3, -1.0])
def test_zero_lr_campaign_config_is_refused(tmp_path: Path, lr: float) -> None:
    """KOM-022 acceptance #47: the second of the two guards, one level up.

    ``CampaignConfig.__post_init__`` raises before any directory is created,
    and ``loop.py`` re-validates through ``StepConfig`` besides -- so a
    non-learning campaign cannot be built by weakening only one of them.

    The acceptance exists to pin a guard KOM-022 must **not** relax: rev 3 of
    its design built the non-learning fixture for the hash gate out of
    ``lr = 0.0``, and rev 4 replaced that fixture with a published checkpoint
    series precisely so that nobody reaches for these two lines instead.
    """
    root = tmp_path / "never-created"
    with pytest.raises(ValueError, match="lr"):
        _campaign(root, lr=lr)
    assert not root.exists(), "the config touched the filesystem before validating"


def test_a_negative_seed_is_rejected(tmp_path: Path) -> None:
    """``SeedSequence`` rejects negative entropy; say so here, not in a child."""
    with pytest.raises(ValueError, match="run_seed"):
        _campaign(tmp_path, run_seed=-1)


def test_an_empty_root_is_rejected() -> None:
    """The campaign owns its root directory; it must have one."""
    with pytest.raises(ValueError, match="root"):
        CampaignConfig(
            iterations=1,
            run_seed=0,
            root="",
            steps_per_iteration=1,
            batch_size=1,
            lr=1e-3,
            workers=1,
            games_per_worker=1,
        )


def test_a_device_outside_the_pair_is_rejected(tmp_path: Path) -> None:
    """'gpu', 'cuda:0' and 'CPU' are all ways to reach the wrong device late."""
    with pytest.raises(ValueError, match="device"):
        _campaign(tmp_path, device="gpu")


def test_cuda_is_a_legal_device(tmp_path: Path) -> None:
    """Validation is a spelling check, not an availability check.

    The desktop run sets ``cuda`` and CI has no GPU: refusing the value on a
    CPU box would make the config unwritable on the machine it is written on.
    """
    assert _campaign(tmp_path, device="cuda").device == "cuda"


# --------------------------------------------------------------------------
# the cross-config batch-fit invariant (review F8, design D11')
# --------------------------------------------------------------------------


def _selfplay(*, workers: int = 1, games_in_flight: int = 1) -> SelfPlayConfig:
    """The cheapest self-play config that carries a lane count."""
    return SelfPlayConfig(
        workers=workers, process_batch_size=1, games_in_flight=games_in_flight
    )


def _seam(max_batch: int) -> InferenceConfig:
    """A seam config whose only interesting field is ``max_batch``."""
    return InferenceConfig(
        in_planes=_PLANES, num_actions=_ACTIONS, max_batch=max_batch
    )


def test_a_batch_ceiling_below_the_slot_count_is_refused(tmp_path: Path) -> None:
    """4 workers x 2 lanes is 8 slots; a 4-row ceiling caps every forward.

    Silently -- ``_row_ceiling`` takes a min and nothing complains -- for the
    whole campaign, which is the failure mode review F8 recorded: the second
    lane buys queueing instead of width and the run is slower than the same box
    was before interleaving was switched on.
    """
    with pytest.raises(ValueError) as refusal:
        campaign_batch_fit(
            _campaign(tmp_path, workers=4),
            _selfplay(workers=4, games_in_flight=2),
            _seam(max_batch=4),
        )
    message = str(refusal.value)
    # All three fields, and the product they imply, so the reader knows which
    # one to move.
    for fragment in ("max_batch=4", "workers=4", "games_in_flight=2", "8"):
        assert fragment in message, message


def test_a_batch_ceiling_that_exactly_holds_the_slots_is_accepted(
    tmp_path: Path,
) -> None:
    """Equality is the boundary: 8 slots need 8 rows, not 9."""
    campaign_batch_fit(
        _campaign(tmp_path, workers=4),
        _selfplay(workers=4, games_in_flight=2),
        _seam(max_batch=8),
    )


def test_one_lane_is_still_measured_against_the_worker_count(
    tmp_path: Path,
) -> None:
    """The check predates lanes as an idea: at one lane the product is workers."""
    with pytest.raises(ValueError, match="max_batch=2"):
        campaign_batch_fit(
            _campaign(tmp_path, workers=3), _selfplay(workers=3), _seam(max_batch=2)
        )


def test_run_campaign_refuses_a_capped_seam_before_it_runs_anything(
    tmp_path: Path,
) -> None:
    """The enforcement point is ``run_campaign``, not only the seam function.

    A campaign is the long unattended run nobody watches, so the refusal must
    happen before the first iteration spawns a single process -- pinned here by
    the shard tree that never appears.
    """
    with pytest.raises(ValueError, match="max_batch"):
        campaign_loop.run_campaign(
            _campaign(tmp_path, iterations=1, workers=4),
            _selfplay(workers=4, games_in_flight=2),
            inference=_seam(max_batch=4),
        )
    assert not shard_root(tmp_path, 0).exists()


# --------------------------------------------------------------------------
# per-iteration seeds
# --------------------------------------------------------------------------


def test_every_iteration_gets_its_own_seed() -> None:
    """No two iterations of a campaign play the same games.

    A reused seed would hand every iteration the identical Gumbel stream, so
    the campaign would re-explore one set of trajectories while the net
    changed underneath it -- slow learning that looks like a hard game.
    """
    seeds = [iteration_seed(11, iteration) for iteration in range(8)]
    assert len(set(seeds)) == len(seeds)
    assert all(seed >= 0 for seed in seeds)


def test_iteration_seeds_are_a_pure_function_of_the_pair() -> None:
    """A resumed campaign must derive the seeds the crashed one would have."""
    assert iteration_seed(11, 3) == iteration_seed(11, 3)
    assert iteration_seed(12, 3) != iteration_seed(11, 3)


# --------------------------------------------------------------------------
# the serving factory
# --------------------------------------------------------------------------


def test_the_default_shape_is_the_b10c128_campaign_net(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without :data:`NET_ENV` the factory builds the design's §2.1 shape."""
    monkeypatch.delenv(NET_ENV, raising=False)
    assert net_config() is CONFIG
    assert CONFIG.trunk.channels == 128
    assert CONFIG.trunk.blocks == 10
    assert CONFIG.trunk.in_planes == observation_size()[0] == 128
    assert CONFIG.heads.num_actions == 37341


def test_the_small_shape_keeps_the_seam_and_shrinks_the_trunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """b2c16, and the observation/action shapes a real worker would send."""
    monkeypatch.setenv(NET_ENV, SMALL_NET)
    monkeypatch.delenv(CHECKPOINT_ENV, raising=False)
    assert net_config() is SMALL_CONFIG

    net = serving_net()
    assert isinstance(net, KomugiNet)
    assert net.config.trunk.channels == 16
    assert net.config.trunk.blocks == 2
    assert net.config.trunk.in_planes == observation_size()[0] == 128
    assert net.config.heads.num_actions == 37341
    # Self-play is inference: a serving net left in training mode would move
    # its BatchNorm statistics with whatever batch the server assembled.
    assert not net.training


def test_the_small_factory_is_the_same_net_by_another_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``small_serving_net`` is module-level too, so it is usable as a factory."""
    monkeypatch.delenv(NET_ENV, raising=False)
    monkeypatch.delenv(CHECKPOINT_ENV, raising=False)
    net = small_serving_net()
    assert isinstance(net, KomugiNet)
    assert net.config is SMALL_CONFIG
    assert small_serving_net.__module__ == "komugi.campaign.factory"
    assert "<locals>" not in small_serving_net.__qualname__


def _perturbed_small_net() -> KomugiNet:
    """A b2c16 net whose weights are *not* the fresh initialisation."""
    net = KomugiNet(SMALL_CONFIG)
    with torch.no_grad():
        for parameter in net.parameters():
            parameter.add_(0.5)
    return net


def test_the_factory_serves_the_checkpoint_the_env_names(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The link a campaign iteration depends on: env in, trained weights out.

    Broken, this fails silently -- every iteration would serve a freshly
    initialised net and the campaign would never improve.
    """
    saved = _perturbed_small_net()
    directory = tmp_path / "ckpt"
    path = save_checkpoint(directory, 8, saved.state_dict(), {}, {"iteration": 1})

    monkeypatch.setenv(NET_ENV, SMALL_NET)
    monkeypatch.setenv(CHECKPOINT_ENV, str(path))
    loaded = serving_net()

    expected = dict(saved.named_parameters())
    for name, parameter in loaded.named_parameters():
        assert torch.equal(parameter, expected[name]), f"{name} was not restored"


def test_the_factory_accepts_the_checkpoint_directory_too(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A directory names its newest verified checkpoint; a file names its own.

    Both spellings resolve through
    :func:`~komugi.train.checkpoint.load_checkpoint`, which is the only code
    allowed to decide what "loadable" means.
    """
    saved = _perturbed_small_net()
    directory = tmp_path / "ckpt"
    save_checkpoint(directory, 8, saved.state_dict(), {}, {})

    monkeypatch.setenv(NET_ENV, SMALL_NET)
    monkeypatch.setenv(CHECKPOINT_ENV, str(directory))
    loaded = serving_net()

    expected = dict(saved.named_parameters())
    for name, parameter in loaded.named_parameters():
        assert torch.equal(parameter, expected[name])


def test_a_pinned_checkpoint_file_is_served_exactly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Pinning an older file must not silently serve a newer sibling.

    The rollback contract the per-iteration coexisting checkpoints exist for:
    the wave-11 union review set :data:`CHECKPOINT_ENV` to a step-8 file in a
    directory that also held a step-16 checkpoint and the factory served the
    step-16 weights (finding F4) -- a pin that upgrades itself. With a single
    checkpoint per directory this is invisible, which is why the tests above
    could not see it.
    """
    older = _perturbed_small_net()
    directory = tmp_path / "ckpt"
    older_path = save_checkpoint(directory, 8, older.state_dict(), {}, {"iteration": 0})
    newer = KomugiNet(SMALL_CONFIG)
    save_checkpoint(directory, 16, newer.state_dict(), {}, {"iteration": 1})

    monkeypatch.setenv(NET_ENV, SMALL_NET)
    monkeypatch.setenv(CHECKPOINT_ENV, str(older_path))
    loaded = serving_net()

    expected = dict(older.named_parameters())
    for name, parameter in loaded.named_parameters():
        assert torch.equal(parameter, expected[name]), (
            f"{name}: the factory served the newer checkpoint although the "
            f"environment pinned {older_path.name}"
        )


def test_without_the_env_the_factory_initialises_freshly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Unset means iteration 0, and iteration 0's weights are not the saved ones."""
    saved = _perturbed_small_net()
    save_checkpoint(tmp_path / "ckpt", 8, saved.state_dict(), {}, {})

    monkeypatch.setenv(NET_ENV, SMALL_NET)
    monkeypatch.delenv(CHECKPOINT_ENV, raising=False)
    fresh = serving_net()

    expected = dict(saved.named_parameters())
    assert any(
        not torch.equal(parameter, expected[name])
        for name, parameter in fresh.named_parameters()
    ), "a fresh factory returned the checkpoint's weights"


def test_an_empty_checkpoint_env_is_also_a_fresh_init(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty, not just unset: the loop spells iteration 0 as ``''``."""
    monkeypatch.setenv(NET_ENV, SMALL_NET)
    monkeypatch.setenv(CHECKPOINT_ENV, "")
    assert isinstance(serving_net(), KomugiNet)


def test_an_unloadable_checkpoint_env_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Loudly, because serving random weights for an hour is the silent version."""
    monkeypatch.setenv(NET_ENV, SMALL_NET)
    monkeypatch.setenv(CHECKPOINT_ENV, str(tmp_path / "nothing" / "here.pt"))
    with pytest.raises(FileNotFoundError, match=CHECKPOINT_ENV):
        serving_net()


def test_the_device_env_defaults_to_the_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """No device literal fires unless the environment asks for one."""
    monkeypatch.setenv(NET_ENV, SMALL_NET)
    monkeypatch.delenv(CHECKPOINT_ENV, raising=False)
    monkeypatch.delenv(DEVICE_ENV, raising=False)
    net = serving_net()
    assert all(parameter.device.type == "cpu" for parameter in net.parameters())


# --------------------------------------------------------------------------
# TrainStep moves the batch to the net's device
# --------------------------------------------------------------------------


class _DeviceProbe(nn.Module):
    """A net whose forward refuses a batch that is not on its own device."""

    def __init__(self) -> None:
        """Hold one parameter, and a record of the devices forward has seen."""
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))
        self.seen: list[torch.device] = []

    def forward(self, x: torch.Tensor) -> NetOutput:
        """Return eight heads' worth of logits derived from `x`.

        The staged four are derived from ``pooled`` like the rest rather than
        being fresh zeros, so they stay on the net's own device -- which is the
        single property this probe exists to observe.
        """
        assert x.device == self.scale.device, (
            f"TrainStep handed a {x.device} batch to a net on "
            f"{self.scale.device}: the batch must follow the net"
        )
        self.seen.append(x.device)
        rows = x.shape[0]
        pooled = x.reshape(rows, -1).sum(dim=-1, keepdim=True) * self.scale
        return NetOutput(
            policy_logits=pooled.expand(rows, _ACTIONS),
            soft_policy_logits=pooled.expand(rows, _ACTIONS),
            wdl_logits=pooled.expand(rows, 3),
            moves_left=pooled.reshape(rows),
            uncertainty=pooled.reshape(rows),
            ownership_logits=pooled.unsqueeze(-1).expand(rows, 81, 3),
            short_term_value=pooled.expand(rows, 3),
            draft_outcome_logits=pooled.expand(rows, 3),
        )


def test_the_step_hands_the_batch_to_the_nets_own_device() -> None:
    """The GPU rule, exercised where CI can exercise it: on the CPU.

    ``.to(device)`` is the identity when the tensor is already there, so this
    cannot distinguish "moved" from "not moved" on a CPU-only box -- what it
    pins is the *rule*: the device comes from the net's parameters and the
    batch is what follows. The probe asserts the agreement from inside the
    forward, so the day a net is built on ``cuda`` (the desktop run, where
    :data:`DEVICE_ENV` is set) an unmoved batch fails here rather than as a
    ``RuntimeError`` deep in a convolution.
    """
    probe = _DeviceProbe()
    step = TrainStep(
        probe, LossAssembler(P0A_RECIPE), StepConfig(batch_size=2, lr=1e-3)
    )
    metrics = step.run(
        {
            "observation": torch.rand((2, _PLANES, 9, 9)),
            "policy_target": torch.full((2, _ACTIONS), 1.0 / _ACTIONS),
            "wdl_target": torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            "search_mode": torch.zeros(2, dtype=torch.int64),
            "termination": torch.zeros(2, dtype=torch.int64),
            # The two columns the moves-left cap-row weight reads (KOM-159 D4),
            # demanded on the same terms as search_mode and termination.
            "ply": torch.zeros(2, dtype=torch.int64),
            "mlh_target": torch.full((2,), 0.25, dtype=torch.float32),
        }
    )
    assert probe.seen == [probe.scale.device]
    assert TOTAL_KEY in metrics


# --------------------------------------------------------------------------
# the global checkpoint step (wave-10 review F5)
# --------------------------------------------------------------------------


def test_two_iterations_into_one_directory_coexist(tmp_path: Path) -> None:
    """The F5 regression: iteration 1 must not annihilate iteration 0.

    Before ``global_step_base``, both calls published under step ``_STEPS``
    and ``save_checkpoint``'s same-step supersede unlinked the first file --
    so exactly one checkpoint ever existed in a campaign directory and
    rollback to a previous generation was impossible. This fails on that
    trainer with one file where two are expected.
    """
    net = _tiny_net()
    step = _step(net)
    rows = [_sample(row) for row in range(_ROWS)]
    directory = tmp_path / "ckpt"

    for iteration in range(2):
        training_iteration(
            net,
            rows,
            step,
            steps=_STEPS,
            rng_seed=iteration,
            checkpoint_dir=directory,
            iteration=iteration,
            global_step_base=iteration * _STEPS,
        )

    refs = list_checkpoints(directory)
    assert [ref.step for ref in refs] == [2 * _STEPS, _STEPS], (
        "the two iterations did not produce two coexisting checkpoints at "
        f"monotone steps; the directory holds {[ref.path.name for ref in refs]}"
    )
    assert all(ref.path.exists() for ref in refs)


def test_the_checkpoint_step_is_the_base_plus_this_calls_steps(tmp_path: Path) -> None:
    """Explicit arithmetic, so the resume division inverts it."""
    net = _tiny_net()
    directory = tmp_path / "ckpt"
    training_iteration(
        net,
        [_sample(row) for row in range(_ROWS)],
        _step(net),
        steps=_STEPS,
        rng_seed=0,
        checkpoint_dir=directory,
        iteration=5,
        global_step_base=5 * _STEPS,
    )
    refs = list_checkpoints(directory)
    assert [ref.step for ref in refs] == [6 * _STEPS]


def test_a_negative_global_step_base_is_rejected(tmp_path: Path) -> None:
    """A negative base would file a checkpoint below one already published."""
    net = _tiny_net()
    with pytest.raises(ValueError, match="global_step_base"):
        training_iteration(
            net,
            [_sample(row) for row in range(_ROWS)],
            _step(net),
            steps=_STEPS,
            rng_seed=0,
            checkpoint_dir=tmp_path,
            global_step_base=-1,
        )


def test_the_default_base_preserves_the_single_iteration_numbering(
    tmp_path: Path,
) -> None:
    """Every existing caller keeps the numbering it already has."""
    net = _tiny_net()
    directory = tmp_path / "ckpt"
    training_iteration(
        net,
        [_sample(row) for row in range(_ROWS)],
        _step(net),
        steps=_STEPS,
        rng_seed=0,
        checkpoint_dir=directory,
    )
    assert [ref.step for ref in list_checkpoints(directory)] == [_STEPS]


# --------------------------------------------------------------------------
# the resume round trip (wave-10 review, untested-but-promised #1 / L15)
# --------------------------------------------------------------------------


def _train(
    net: KomugiNet,
    step: TrainStep,
    rows: list[dict[str, torch.Tensor]],
    rng_seed: int,
    directory: Path | None = None,
) -> None:
    """Run one two-step iteration, optionally checkpointing it."""
    training_iteration(
        net, rows, step, steps=_STEPS, rng_seed=rng_seed, checkpoint_dir=directory
    )


def _resumed(
    directory: Path, rows: list[dict[str, torch.Tensor]], *, with_moments: bool
) -> KomugiNet:
    """Rebuild from the checkpoint in `directory` and train two more steps."""
    checkpoint = load_checkpoint(directory)
    assert checkpoint is not None
    net = _tiny_net()
    net.load_state_dict(checkpoint.model_state)
    step = _step(net)
    if with_moments:
        step.optimizer.load_state_dict(checkpoint.optim_state)
    _train(net, step, rows, rng_seed=2)
    return net


def test_a_resumed_run_continues_exactly_where_it_stopped(tmp_path: Path) -> None:
    """The L15-shaped gap: the *sequence*, not each half in isolation.

    Both halves were already green -- the optimiser state is saved and
    non-empty, the model state round-trips bit-equal -- and neither says
    anything about a run that stops, reloads and continues. Here it does: a
    net trained 2 + 2 steps across a checkpoint round trip must reach the
    **bit-identical** parameters of one trained 2 + 2 steps in memory.

    Bit-identical rather than approximate because every input to the second
    half is a pure function of what the checkpoint carries: same batches (the
    seeded sampling stream), same weights, same moments, and AdamW's update is
    a closed form of them.
    """
    rows = [_sample(row) for row in range(_ROWS)]
    directory = tmp_path / "ckpt"

    uninterrupted = _tiny_net()
    step = _step(uninterrupted)
    _train(uninterrupted, step, rows, rng_seed=1)
    _train(uninterrupted, step, rows, rng_seed=2)

    checkpointed = _tiny_net()
    _train(checkpointed, _step(checkpointed), rows, rng_seed=1, directory=directory)
    resumed = _resumed(directory, rows, with_moments=True)

    expected = dict(uninterrupted.named_parameters())
    for name, parameter in resumed.named_parameters():
        assert torch.equal(parameter, expected[name]), (
            f"{name} diverged across the resume: a run that reloads is not the "
            f"run it claims to be continuing"
        )


def test_a_resume_without_the_moments_is_a_different_run(tmp_path: Path) -> None:
    """The control, and why the optimiser state is half of a checkpoint.

    Restoring the weights alone produces a run that looks perfectly ordinary
    -- ordinary loss, ordinary curve -- and takes mis-scaled steps until
    AdamW's moment estimates warm up again, at *every* iteration boundary of a
    campaign. If this test ever passed, the one above would be pinning
    nothing.
    """
    rows = [_sample(row) for row in range(_ROWS)]
    directory = tmp_path / "ckpt"
    net = _tiny_net()
    _train(net, _step(net), rows, rng_seed=1, directory=directory)

    with_moments = dict(_resumed(directory, rows, with_moments=True).named_parameters())
    without = _resumed(directory, rows, with_moments=False)
    assert any(
        not torch.equal(parameter, with_moments[name])
        for name, parameter in without.named_parameters()
    ), "dropping the optimiser state changed nothing; the moments are not travelling"


# --------------------------------------------------------------------------
# resume detection
# --------------------------------------------------------------------------


def test_an_empty_root_resumes_at_iteration_zero(tmp_path: Path) -> None:
    """No checkpoint is a fresh campaign, not an error."""
    point = resume_point(_campaign(tmp_path))
    assert point.iteration == 0
    assert point.checkpoint is None


def test_resume_picks_the_highest_step_and_continues_after_it(
    tmp_path: Path,
) -> None:
    """Three iterations on disk; the campaign continues at the fourth."""
    campaign = _campaign(tmp_path)
    directory = checkpoint_dir(tmp_path)
    net = _tiny_net()
    for iteration in range(3):
        save_checkpoint(
            directory,
            (iteration + 1) * campaign.steps_per_iteration,
            net.state_dict(),
            {},
            {"iteration": iteration},
        )

    point = resume_point(campaign)
    assert point.checkpoint is not None
    assert point.checkpoint.step == 3 * campaign.steps_per_iteration
    assert point.iteration == 3


def test_resume_falls_back_to_the_step_when_the_meta_says_nothing(
    tmp_path: Path,
) -> None:
    """Step ``(i + 1) * steps_per_iteration`` inverts to "start at ``i + 1``"."""
    campaign = _campaign(tmp_path)
    save_checkpoint(
        checkpoint_dir(tmp_path),
        3 * campaign.steps_per_iteration,
        _tiny_net().state_dict(),
        {},
        {},
    )
    assert resume_point(campaign).iteration == 3


# --------------------------------------------------------------------------
# WindowSource
# --------------------------------------------------------------------------


def _source(values: tuple[int, ...]) -> SampleSource:
    """Return a ``SampleSource`` whose rows report `values`."""
    return _ListSource(values)


def _window() -> RecencyWindow:
    """Two iterations: three old rows (0-2) and five new ones (10-14)."""
    refs: list[IterationRef] = [
        IterationRef(iteration=0, source=_source((0, 1, 2))),
        IterationRef(iteration=1, source=_source((10, 11, 12, 13, 14))),
    ]
    return RecencyWindow(refs, WindowConfig(), seed=0)


def test_the_window_source_reaches_every_row_of_every_member() -> None:
    """No sample is unreachable, which a naive quota would silently make it.

    The index map is monotone, so a member whose block is *shorter* than its
    row count would be sampled at a stride -- 200 of 1000 rows, forever, with
    nothing to see. The scale is chosen to prevent exactly that.
    """
    source = WindowSource(_window())
    seen = {int(source[index]["observation"].item()) for index in range(len(source))}
    assert seen == {0, 1, 2, 10, 11, 12, 13, 14}


def test_the_window_source_blocks_follow_the_recency_weights() -> None:
    """The tilt is in the index map, so uniform sampling reproduces it.

    Default decay 0.8 over two iterations gives the newer one 1/1.8 of the
    draws and the older 0.8/1.8, which is 5 and 4 of nine virtual slots here.
    """
    window = _window()
    source = WindowSource(window)
    assert source.quotas == (5, 4)
    assert len(source) == 9
    newest_share = source.quotas[0] / len(source)
    assert newest_share == pytest.approx(window.weights[0], abs=0.02)


def test_the_window_source_rejects_an_index_outside_its_range() -> None:
    """A silent wrap would train on the wrong iteration."""
    source = WindowSource(_window())
    with pytest.raises(IndexError):
        source[len(source)]
    assert int(source[-1]["observation"].item()) in {0, 1, 2}


def test_window_refs_retains_only_what_the_window_can_use(tmp_path: Path) -> None:
    """Old iterations get unretained readers; the newest ``size`` are retained.

    The KOM-033 review (F1) measured the alternative: a production iteration
    is ~1.5-2.2 GB decompressed (200x its on-disk size), and ``window_refs``
    built retained readers for every iteration the campaign had ever run --
    the resident set grew by GBs per iteration for data the size-5 window
    provably never samples. Eligibility stays the window's decision (every
    non-empty iteration still gets a ref); only the *retention* is scoped.
    """
    from komugi.campaign.loop import shard_root, window_refs
    from komugi.train.shards import FieldSpec, Schema, ShardWriter

    schema = Schema(fields=(FieldSpec(name="x", dtype="float32", shape=(2,)),))
    iterations = 7
    for iteration in range(iterations):
        worker_dir = shard_root(tmp_path, iteration) / "worker_0_r0"
        worker_dir.mkdir(parents=True)
        with ShardWriter(worker_dir, schema, samples_per_shard=4) as writer:
            for row in range(4):
                writer.append({"x": torch.full((2,), float(iteration * 10 + row))})

    refs = window_refs(tmp_path, iterations - 1)  # default retain_last = size 5

    assert [ref.iteration for ref in refs] == list(range(iterations))
    retention = [
        all(reader.retains_arrays for reader in ref.source.sources)  # type: ignore[attr-defined]
        for ref in refs
    ]
    assert retention == [False, False, True, True, True, True, True], (
        f"retention by iteration was {retention}: only the newest "
        f"{WindowConfig().size} iterations may hold decompressed arrays"
    )


# --------------------------------------------------------------------------
# window_refs' construction bound (KOM-041 / KOM-032 review F7)
# --------------------------------------------------------------------------


def _write_iteration(root: Path, iteration: int, rows: int) -> None:
    """Give iteration `iteration` a worker directory holding `rows` samples.

    ``rows=0`` writes the *empty* case the campaign really produces -- a
    worker directory a wedged incarnation created and never filled -- rather
    than a missing tree, which the caller expresses by not calling this at all.
    """
    from komugi.train.shards import FieldSpec, Schema, ShardWriter

    worker_dir = shard_root(root, iteration) / "worker_0_r0"
    worker_dir.mkdir(parents=True)
    if rows == 0:
        return
    schema = Schema(fields=(FieldSpec(name="x", dtype="float32", shape=(2,)),))
    with ShardWriter(worker_dir, schema, samples_per_shard=4) as writer:
        for row in range(rows):
            writer.append({"x": torch.full((2,), float(iteration * 10 + row))})


def _members(refs: tuple[IterationRef, ...]) -> list[tuple[int, int]]:
    """Return the window over `refs` as ``(iteration, rows)``, newest first."""
    window = RecencyWindow(refs, WindowConfig(), seed=0)
    return [(ref.iteration, len(ref.source)) for ref in window.members]


def test_the_bounded_walk_builds_the_same_window_as_the_unbounded_one(
    tmp_path: Path,
) -> None:
    """The equivalence the bound rests on, over a history with holes in it.

    ``RecencyWindow.__init__`` sorts descending and keeps the first ``size``
    refs holding a sample, and ``window_refs`` only ever returns non-empty
    refs -- so everything past newest-first position ``size`` is unreachable
    no matter what it holds, and never constructing it cannot change the
    window. Iterations 3 (an empty worker directory) and 7 (no directory at
    all) are here because the two empties are the case where "position" and
    "iteration number" come apart.
    """
    through = 9
    for iteration in range(through + 1):
        if iteration == 7:
            continue  # missing tree: the iteration was never played
        _write_iteration(tmp_path, iteration, 0 if iteration == 3 else 2 + iteration)

    size = WindowConfig().size
    unbounded = window_refs(tmp_path, through)
    bounded = window_refs(tmp_path, through, stop_after=size)

    assert len(unbounded) == 8, "the rig must offer more history than the window"
    assert [ref.iteration for ref in bounded] == [4, 5, 6, 8, 9], (
        "the bounded walk must return the newest `size` non-empty iterations, "
        "still ascending, with the two empties skipped rather than counted"
    )
    assert _members(bounded) == _members(unbounded), (
        "bounding construction at the consuming window's size changed the "
        "window's membership or its per-member row counts"
    )


def _spy_on_iteration_source(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Record which iterations ``window_refs`` actually builds a source for."""
    calls: list[int] = []
    real = campaign_loop.iteration_source

    def counted(
        root: str | os.PathLike[str], iteration: int, *, retain_arrays: bool = True
    ) -> ConcatSource:
        calls.append(iteration)
        return real(root, iteration, retain_arrays=retain_arrays)

    monkeypatch.setattr(campaign_loop, "iteration_source", counted)
    return calls


def test_the_bound_stops_the_walk_rather_than_filtering_its_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The point of the ticket: cost, which the returned tuple cannot show.

    Each :func:`~komugi.campaign.loop.iteration_source` call globs the
    iteration's worker directories and scans every shard's header, so an
    unbounded walk costs one scan per iteration the campaign has *ever* run,
    on every training iteration -- quadratic over a campaign (KOM-032 review
    F7). A result-level assertion passes just as happily on a walk that builds
    all twelve and throws nine away, so the spy is the test.
    """
    through = 11
    for iteration in range(through + 1):
        _write_iteration(tmp_path, iteration, 3)

    calls = _spy_on_iteration_source(monkeypatch)
    refs = window_refs(tmp_path, through, stop_after=3)
    assert [ref.iteration for ref in refs] == [9, 10, 11]
    assert calls == [11, 10, 9], (
        f"the walk touched {calls}: with three non-empty iterations found it "
        f"must stop, not carry on down to iteration 0"
    )

    calls.clear()
    assert len(window_refs(tmp_path, through)) == 12
    assert calls == list(range(through, -1, -1)), (
        f"stop_after=None touched {calls}: the unbounded walk is the "
        f"behaviour every existing caller still gets"
    )


def test_empty_iterations_do_not_consume_the_bound(tmp_path: Path) -> None:
    """The budget counts iterations *with data*, as the window's ``size`` does.

    Counting directories instead would let a run of wedged incarnations --
    exactly the situation that leaves empty shard trees -- shrink the window
    below its configured size, and it would do so silently.
    """
    through = 9
    for iteration in range(through + 1):
        _write_iteration(tmp_path, iteration, 0 if iteration in {6, 8} else 4)

    refs = window_refs(tmp_path, through, stop_after=3)
    assert [ref.iteration for ref in refs] == [5, 7, 9]
    assert all(len(ref.source) == 4 for ref in refs)


def test_a_stop_after_below_one_is_rejected(tmp_path: Path) -> None:
    """Zero is an empty window, and the window's own error names the symptom."""
    _write_iteration(tmp_path, 0, 2)
    with pytest.raises(ValueError, match="stop_after"):
        window_refs(tmp_path, 0, stop_after=0)


def test_a_bound_past_the_history_returns_what_exists(tmp_path: Path) -> None:
    """stop_after beyond the data mirrors an under-full window, not an error."""
    for iteration in range(2):
        _write_iteration(tmp_path, iteration, 2)
    refs = window_refs(tmp_path, 1, stop_after=5)
    assert [ref.iteration for ref in refs] == [0, 1]


def test_the_campaign_call_shape_takes_both_bounds_from_one_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The KOM-041 equivalence hangs on run_campaign passing the SAME size as
    ``retain_last`` and ``stop_after``: :func:`campaign_window_refs` is the one
    place that call shape lives, and this pin is what notices if either bound
    drifts from the config the window itself is built with (impl review F1 --
    the mutant `stop_after=1` at the call site survived every other test)."""
    seen: dict[str, object] = {}

    def spy(root: object, through: int, **kwargs: object) -> tuple[()]:
        seen.update(kwargs, through=through)
        return ()

    monkeypatch.setattr(campaign_loop, "window_refs", spy)
    config = WindowConfig(size=3)
    campaign_loop.campaign_window_refs(tmp_path, 7, config)
    assert seen == {"retain_last": 3, "stop_after": 3, "through": 7}, (
        f"campaign_window_refs must pass config.size as BOTH bounds; saw {seen}"
    )


# --------------------------------------------------------------------------
# KOM-002 §5.3 -- the campaign's own two load sites (review F3, F4)
# --------------------------------------------------------------------------

_PMV1_PLANES: Final[int] = 118
"""The plane count KOM-002 broke: 118, from KOM-027's map."""


def _publish_stale(directory: Path, meta: Mapping[str, object]) -> None:
    """Publish a verifying step-1 checkpoint of the small net under `meta`.

    The weights are the campaign net's own, so ``load_state_dict`` would
    SUCCEED if it were reached: every refusal below is therefore a statement
    about ordering -- the guard ran before the load -- and not a shape mismatch
    surfacing from underneath it.
    """
    save_checkpoint(directory, 1, KomugiNet(SMALL_CONFIG).state_dict(), {}, meta)


def _publish_at(directory: Path, step: int) -> None:
    """Publish a verifying checkpoint of the small net at `step` under `directory`.

    The EMA advance gate reads the *step numbering* and nothing else, so the
    weights are irrelevant to it -- but they are a real net's, published
    through the real codec, because a file the loader would skip is not a
    published EMA and would make the gate's refusal a statement about
    corruption rather than about staleness.
    """
    save_checkpoint(
        directory,
        step,
        KomugiNet(SMALL_CONFIG).state_dict(),
        {},
        {"iteration": step, "role": "ema"},
    )


def test_the_probe_loader_refuses_a_stale_plane_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review F3: the EMA state the KL probe measures is guarded at the source.

    ``_newest_ema_state`` is the last point on this path where the *checkpoint*
    still exists -- ``_loaded_net`` receives a bare state dict, by which time
    the meta recording which plane map those weights were trained under is
    gone. An unguarded stale EMA takes the campaign down inside torch, at an
    iteration boundary, while it is writing telemetry, with a shape traceback
    that names neither the plane map nor the ticket.

    **Deleting ``require_matching_plane_count`` from
    :func:`komugi.campaign.loop._newest_ema_state` fires
    ``pytest.raises(ValueError)`` below**, and the loader hands the probe
    weights from a map the encoder no longer speaks.
    """
    monkeypatch.setenv(NET_ENV, SMALL_NET)
    ema = tmp_path / "ema"
    _publish_stale(ema, {"in_planes": _PMV1_PLANES, "plane_map_version": 1})

    with pytest.raises(ValueError) as raised:
        campaign_loop._newest_ema_state(ema)
    message = str(raised.value)
    assert str(_PMV1_PLANES) in message
    assert str(observation_size()[0]) in message
    assert "KOM-002" in message

    # Non-vacuity: the guard refuses STALE files, not every file. A checkpoint
    # today's codec stamped passes and the weights come back.
    current = tmp_path / "ema-current"
    _publish_stale(current, {"iteration": 1})
    state = campaign_loop._newest_ema_state(current)
    assert "trunk.stem.weight" in state


def test_the_resume_path_refuses_a_stale_plane_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review F4: the campaign-resume load site, asserted rather than read.

    The resume is the site with the worst failure mode of the three: the
    checkpoint that carries a whole campaign's training forward. What made it
    untested was placement -- the guard sat inline in ``run_campaign``, behind
    a self-play iteration -- so :func:`komugi.campaign.loop._net_for_resume`
    holds the guard and the load together, and ``run_campaign`` calls it.

    **Deleting ``require_matching_plane_count`` from ``_net_for_resume`` fires
    ``pytest.raises(ValueError)`` below**: the stale weights load, the campaign
    resumes, and the net reads a board encoded under a map it was never trained
    on.
    """
    monkeypatch.setenv(NET_ENV, SMALL_NET)
    directory = checkpoint_dir(tmp_path)
    _publish_stale(directory, {"in_planes": _PMV1_PLANES, "plane_map_version": 1})

    stale = load_checkpoint(directory)
    assert stale is not None, "the checkpoint itself verifies; only its map is old"

    with pytest.raises(ValueError) as raised:
        campaign_loop._net_for_resume(stale)
    message = str(raised.value)
    assert str(_PMV1_PLANES) in message
    assert "KOM-002" in message

    # Non-vacuity, both ways: no checkpoint is a fresh campaign, and a current
    # checkpoint resumes with its weights actually in the net.
    fresh = campaign_loop._net_for_resume(None)
    assert fresh.config.trunk.in_planes == observation_size()[0]

    current_dir = checkpoint_dir(tmp_path / "current")
    _publish_stale(current_dir, {"iteration": 1})
    current = load_checkpoint(current_dir)
    assert current is not None
    resumed = campaign_loop._net_for_resume(current)
    assert torch.equal(
        resumed.state_dict()["trunk.stem.weight"],
        current.model_state["trunk.stem.weight"],
    )


# --------------------------------------------------------------------------
# KOM-025: the trainer's extras cross the boundary, and the EMA advance gate
# --------------------------------------------------------------------------

_DECILE_KEYS: Final[tuple[str, ...]] = (
    "loss_total_first_decile",
    "loss_total_last_decile",
    "loss_total_step_std",
)
"""KOM-005b's staging statistic, spelled as the trainer writes it.

Restated here rather than imported from ``komugi.train.step`` (which builds
them from a private prefix) for the reason ``campaign_rig`` restates
``CHECKPOINT_ENV``: these key names *are* the interface between the trainer
that computes the statistic and the stage gate that reads it off the campaign
record, and a rename that broke it would otherwise be invisible.
"""


def _trainer_record(iteration: int, **extras: float) -> IterationMetrics:
    """A record shaped like :func:`~komugi.train.step.training_iteration`'s.

    Zeroed game statistics and no campaign marker -- the trainer never meets a
    ``RunSummary`` and never writes the marker, which is exactly what makes the
    marker the discriminator.
    """
    fields: dict[str, float] = {
        "loss_total": 1.5,
        "steps": 8.0,
        "batch_size": 16.0,
    }
    fields.update({key: 0.5 for key in _DECILE_KEYS})
    fields.update(extras)
    return IterationMetrics(
        iteration=iteration,
        games=0,
        draw_rate=0.0,
        avg_game_len=0.0,
        policy_entropy=0.0,
        extras=fields,
    )


def _campaign_record(iteration: int, **extras: float) -> IterationMetrics:
    """A record shaped like the one the campaign writes at a boundary."""
    fields: dict[str, float] = {boundary.CAMPAIGN_MARKER: 1.0}
    fields.update(extras)
    return IterationMetrics(
        iteration=iteration,
        games=3,
        draw_rate=0.0,
        avg_game_len=0.0,
        policy_entropy=0.0,
        extras=fields,
    )


def test_the_trainers_decile_keys_are_read_back_off_the_journal(
    tmp_path: Path,
) -> None:
    """KOM-025: the staging statistic reaches the campaign's own record.

    KOM-005b computes ``loss_<name>_first_decile`` / ``_last_decile`` /
    ``_step_std`` inside ``training_iteration``, over the step history only
    that function holds, and journals them on *its* record. Every
    campaign-level reader filters on
    :data:`~komugi.campaign.boundary.CAMPAIGN_MARKER`, so before this the
    statistic every stage-1-to-3 gate reads was on a record those readers skip.
    """
    journal = campaign_loop.journal_path(tmp_path)
    write_record(journal, _trainer_record(0))

    extras = campaign_loop.trainer_extras(journal, 0)

    for key in _DECILE_KEYS:
        assert extras.get(key) == 0.5, (
            f"{key} did not cross from the trainer's record to the campaign's; "
            f"got {sorted(extras)}"
        )
    assert extras["loss_total"] == 1.5
    assert extras["steps"] == 8.0


def test_the_campaigns_own_record_is_never_mistaken_for_the_trainers(
    tmp_path: Path,
) -> None:
    """The marker is the discriminator, not ``games == 0``.

    An iteration whose self-play produced no games still trains on the older
    window and still writes a campaign record, so a ``games``-based
    discriminator fails precisely during a self-play outage. Here the campaign
    record is the *only* one for iteration 1, and reading it back as "the
    trainer's" would feed the boundary its own extras.
    """
    journal = campaign_loop.journal_path(tmp_path)
    write_record(journal, _campaign_record(1, some_boundary_measurement=9.0))

    assert campaign_loop.trainer_extras(journal, 1) == {}, (
        "the campaign's own record was read back as the trainer's"
    )


def test_only_this_iterations_trainer_record_is_read(tmp_path: Path) -> None:
    """Iteration 2's boundary must not carry iteration 1's losses."""
    journal = campaign_loop.journal_path(tmp_path)
    write_record(journal, _trainer_record(1, loss_total=11.0))
    write_record(journal, _trainer_record(2, loss_total=22.0))

    assert campaign_loop.trainer_extras(journal, 2)["loss_total"] == 22.0
    assert campaign_loop.trainer_extras(journal, 1)["loss_total"] == 11.0


def test_a_re_run_iterations_last_trainer_record_wins(tmp_path: Path) -> None:
    """The crash-before-checkpoint schedule appends; the newest describes disk.

    An iteration that died after its shards were sealed and before its
    checkpoint was published is re-run into a cleaned tree, and the journal is
    append-only, so both attempts' records survive under one iteration number.
    The surviving attempt is the later line -- the same adjudication
    :func:`~komugi.campaign.boundary.latest_fingerprints` makes for the
    fingerprint series.
    """
    journal = campaign_loop.journal_path(tmp_path)
    write_record(journal, _trainer_record(3, loss_total=99.0))
    write_record(journal, _trainer_record(3, loss_total=4.0))

    assert campaign_loop.trainer_extras(journal, 3)["loss_total"] == 4.0


def test_a_missing_trainer_record_costs_the_boundary_nothing(
    tmp_path: Path,
) -> None:
    """No journal, or no record for this iteration: an empty mapping, not a raise.

    The campaign record is still written, carrying everything the boundary
    itself measured. A boundary that raised here would turn a missing
    diagnostic into a dead campaign.
    """
    absent = campaign_loop.journal_path(tmp_path)
    assert campaign_loop.trainer_extras(absent, 0) == {}

    write_record(absent, _trainer_record(0))
    assert campaign_loop.trainer_extras(absent, 7) == {}


def test_the_campaign_logs_are_the_two_a_relaunch_must_not_lose(
    tmp_path: Path,
) -> None:
    """Both logs, so a relaunch guard cannot save one and expose the other."""
    assert campaign_loop.campaign_logs(tmp_path) == (
        campaign_loop.journal_path(tmp_path),
        selfplay_journal_path(tmp_path),
    )


def test_the_ema_advance_gate_passes_when_the_publish_kept_up(
    tmp_path: Path,
) -> None:
    """KOM-022 §2 read off the numbering: ``EMA_i`` is filed at ``theta_i``'s step."""
    ema = campaign_loop.ema_dir(tmp_path)
    _publish_at(ema, 8)

    verdict = campaign_loop.ema_publish_verdict(ema, 8)

    assert verdict.ok, verdict.reason
    assert "8" in verdict.reason


def test_a_stalled_ema_publish_fails_the_gate_at_its_own_boundary(
    tmp_path: Path,
) -> None:
    """The whole point of the step gate: it does not need a second iteration.

    The run-length hash gate catches a stalled EMA too, but only once the same
    digest has been recorded *twice* -- a freeze is a statement about change.
    So a publish that silently no-ops at iteration ``i`` buys a further whole
    iteration of self-play, training and checkpointing against the stale net
    before anything stops. This is the same invariant on two integers,
    available immediately.

    **Deleting the step comparison makes this node pass vacuously**, which is
    why the reason string is asserted to name both steps.
    """
    ema = campaign_loop.ema_dir(tmp_path)
    _publish_at(ema, 8)

    verdict = campaign_loop.ema_publish_verdict(ema, 16)

    assert not verdict.ok, "a one-generation-stale EMA passed the advance gate"
    assert "8" in verdict.reason and "16" in verdict.reason, (
        f"the alert must name both steps for an operator to act on it: "
        f"{verdict.reason!r}"
    )


def test_an_empty_ema_directory_fails_the_gate(tmp_path: Path) -> None:
    """A gate that cannot see its input has not verified anything."""
    verdict = campaign_loop.ema_publish_verdict(campaign_loop.ema_dir(tmp_path), 8)

    assert not verdict.ok
    assert "no EMA checkpoint" in verdict.reason


def test_the_gate_reads_the_newest_publish_not_the_oldest(tmp_path: Path) -> None:
    """``list_checkpoints`` is newest-first; the gate must agree with it.

    A campaign has one EMA file per iteration, so reading the wrong end of the
    listing would fail every healthy boundary after the first -- and pass every
    stalled one on a fresh root, which is the dangerous direction.
    """
    ema = campaign_loop.ema_dir(tmp_path)
    _publish_at(ema, 8)
    _publish_at(ema, 16)

    assert campaign_loop.ema_publish_verdict(ema, 16).ok
    assert not campaign_loop.ema_publish_verdict(ema, 8).ok


# --------------------------------------------------------------------------
# KOM-025 wire-in review F8 -- the owed step is a FILE's, not an expression's
# --------------------------------------------------------------------------


def test_the_owed_step_is_read_off_the_train_checkpoint(tmp_path: Path) -> None:
    """F8: the gate's expectation comes from disk, not from the arithmetic.

    ``EMA_i`` is published at ``theta_i``'s step, so the step the EMA owes is a
    property of the newest file under ``<root>/ckpt``. Deriving it a second
    time from ``(i + 1) * steps_per_iteration`` makes an **unmutable** gate
    whose two sides can agree because they were computed the same way -- and
    the dangerous direction is the silent one: a root whose checkpoints are not
    where the expression says would be gated against a step that exists
    nowhere on disk.

    Here the train checkpoint is at 12 while the arithmetic predicts 16, which
    is the disagreement the whole finding is about.
    """
    checkpoints = checkpoint_dir(tmp_path)
    _publish_at(checkpoints, 12)

    assert campaign_loop.owed_ema_step(checkpoints, 16) == 12, (
        "the owed step was taken from the arithmetic rather than from the "
        "newest train checkpoint on disk"
    )


def test_an_ema_at_the_predicted_step_still_fails_when_the_file_disagrees(
    tmp_path: Path,
) -> None:
    """And the refusal names the file's step, keeping the prediction as evidence.

    An EMA published at the *predicted* step over a root whose train checkpoint
    is elsewhere is exactly the state a re-derived expectation cannot see. The
    message must carry both numbers: the one it judged on (12) and the one the
    boundary's own arithmetic expected (16), because a disagreement between
    them is itself the diagnosis.
    """
    checkpoints = checkpoint_dir(tmp_path)
    ema = campaign_loop.ema_dir(tmp_path)
    _publish_at(checkpoints, 12)
    _publish_at(ema, 16)

    verdict = campaign_loop.ema_publish_verdict(
        ema, campaign_loop.owed_ema_step(checkpoints, 16), predicted_step=16
    )

    assert not verdict.ok, (
        "an EMA at the step the arithmetic predicted passed a gate whose "
        "train checkpoint is at another step"
    )
    assert "12" in verdict.reason and "16" in verdict.reason, (
        f"the alert must name the step the train checkpoint is at and the one "
        f"the boundary predicted: {verdict.reason!r}"
    )


def test_with_no_train_checkpoint_the_arithmetic_is_the_fallback(
    tmp_path: Path,
) -> None:
    """Non-vacuity: an empty ``ckpt`` directory must not gate against step 0.

    Reading the expectation off a file that does not exist has to degrade to
    the prediction rather than to zero -- otherwise the first boundary of any
    root whose checkpoint write is still in flight would fail an unmutable
    gate.
    """
    assert campaign_loop.owed_ema_step(checkpoint_dir(tmp_path), 16) == 16


def test_the_published_ema_step_is_the_newest_files_own(tmp_path: Path) -> None:
    """F6: what :data:`EMA_PUBLISHED_STEP` is stamped from, and its empty case."""
    ema = campaign_loop.ema_dir(tmp_path)
    assert campaign_loop.published_ema_step(ema) is None

    _publish_at(ema, 8)
    _publish_at(ema, 16)

    assert campaign_loop.published_ema_step(ema) == 16.0


# --------------------------------------------------------------------------
# KOM-025 wire-in review F3/F4 -- the merge itself, at a real boundary
# --------------------------------------------------------------------------

_BOUNDARY_STEPS: Final[int] = 8
"""``steps_per_iteration`` for the planted boundary below: iteration 0 owes an
EMA at step 8."""

_SENTINEL: Final[float] = 12345.5
"""A value the loop's own inputs cannot produce.

The step history the boundary is handed below is ``{1.0, 0.9}``, so every
statistic derivable from it -- a decile mean, a tail median, a population
standard deviation -- lies inside ``[0.0, 1.0]``. A key carrying this number on
the campaign record therefore *came from the trainer's record on disk*, which
is the claim an equality against a faithfully re-derived statistic cannot make
(KOM-025 wire-in review, F3 / mutant M10).
"""

_BOUNDARY_OWNED: Final[frozenset[str]] = frozenset(
    {
        boundary.CAMPAIGN_MARKER,
        boundary.LOSS_TAIL_MEDIAN,
        boundary.SPIKE_FLAG,
        boundary.BAND_VIOLATION,
        boundary.FLAT_TRAINER_FLAG,
        boundary.MUTED_FLAG,
        boundary.BN_DIGEST_LO,
        boundary.BN_DIGEST_ADVANCED,
        boundary.GAUNTLET_BREACH,
        *boundary.GAUNTLET_RAN,
        campaign_loop.EMA_PUBLISHED_STEP,
        pinned.EMA_PUBLISH_STALLED,
        campaign_loop.DRAWS,
        campaign_loop.TURN_CAP_GAMES,
        campaign_loop.CAP_RATE,
        campaign_loop.MAX_PLIES,
        campaign_loop.EARLY_SHU_RATE,
        campaign_loop.MEDIAN_PLACEMENTS_W,
        campaign_loop.MEDIAN_PLACEMENTS_B,
        campaign_loop.CAP_P_MEAN,
        campaign_loop.CAP_P_GE_03_SHARE,
        campaign_loop.CAP_ADJUDICATED_GAMES,
        campaign_loop.LOSS_RECIPE_IS_MLH,
        campaign_loop.MLH_UTILITY_LAMBDA,
        "rows",
    }
)
"""Every extras key the **boundary** writes and owns *exclusively*.

The merge documents its precedence as "the trainer's extras go in first, so
every key the boundary measured overrides them" (``loop.py``), and that rule is
only observable where the two namespaces meet. They are disjoint today -- which
is what makes the precedence untested rather than harmless -- and
``loss_tail_median`` / ``loss_spike`` already sit *inside* the ``loss_`` prefix
the trainer owns, one loss-head name away from a real collision.

:data:`~komugi.campaign.loop.MLH_UTILITY_LAMBDA` is **in** this set, and its
membership is the enforced half of KOM-171: KOM-159a landed with that key
stamped by ``training_iteration`` as well -- which made the disjointness claim
above false, and was bought off by narrowing this set -- and the follow-up made
the boundary the single writer (KOM-159a finisher review F5). The key belongs
to the boundary because it describes the *games*: it is read off the self-play
config the loop owns, and is not a property of an optimiser step (``step.py``
carries the field only to be handed it, and journals it nowhere). So the two
disjointness tests below are now what catches a trainer-side stamp coming back,
rather than this docstring being what excuses it.
"""


def _close_a_planted_boundary(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    stats: GameStats | None = None,
    *,
    loss_recipe: str = campaign_loop.RECIPE_P0B,
    selfplay: SelfPlayConfig | None = None,
) -> Mapping[str, float]:
    """Run iteration 0's real boundary over a planted root; return its record.

    Everything that decides what lands on the campaign record is the production
    path: :func:`~komugi.campaign.loop.trainer_extras` reads the journal, the
    merge merges, the loss bounds and the EMA advance gate stamp their own
    keys, ``record_fingerprints`` writes the real digests and ``enforce`` runs.
    What is stood in for is only what an iteration boundary *cannot* be given
    in a unit test:

    * ``reconcile_ema`` -- the EMA it would publish is planted instead, at the
      step this boundary owes, so the gate sees a healthy publish;
    * ``_probe_set`` -- there is no shard tree, so there are no probe positions;
      the KL block is skipped exactly as it is on a root whose iteration 0
      produced no readable rows.

    ``CHECKPOINT_ENV`` is cleared, which is what iteration 0 genuinely looks
    like (nothing was served before the first EMA existed): two fingerprint
    records rather than three, which is the ``3N - 1`` count the completeness
    gate expects.

    `loss_recipe` and `selfplay` are the two inputs whose *values* the boundary
    stamps on the record it writes (KOM-159 design review F10), so they are
    overridable here: the alternative -- a campaign 6 boundary -- is the only
    way to see those two keys, and it is one field of a config away from this
    one.

    Returns the extras of the one record carrying
    :data:`~komugi.campaign.boundary.CAMPAIGN_MARKER`.
    """
    root.mkdir(parents=True, exist_ok=True)
    campaign = dataclasses.replace(
        _campaign(root, iterations=1, steps_per_iteration=_BOUNDARY_STEPS),
        loss_recipe=loss_recipe,
    )
    _publish_at(checkpoint_dir(root), _BOUNDARY_STEPS)
    _publish_at(campaign_loop.ema_dir(root), _BOUNDARY_STEPS)

    def already_published(*args: object, **kwargs: object) -> tuple[int, ...]:
        """The EMA this boundary owes is planted above, at its own step."""
        return ()

    def no_probe(*args: object, **kwargs: object) -> tuple[None, str]:
        """No shard tree, so no probe positions -- the KL block is skipped."""
        return (None, "")

    monkeypatch.delenv(CHECKPOINT_ENV, raising=False)
    monkeypatch.setattr(campaign_loop, "reconcile_ema", already_published)
    monkeypatch.setattr(campaign_loop, "_probe_set", no_probe)

    campaign_loop._close_iteration(
        root,
        campaign,
        _selfplay() if selfplay is None else selfplay,
        0,
        parameter_keys(KomugiNet(SMALL_CONFIG)),
        served="",
        resume=None,
        trained=KomugiNet(SMALL_CONFIG),
        history=[{TOTAL_KEY: 1.0}, {TOTAL_KEY: 0.9}],
        games=1,
        rows=4,
        # By default the real census function over this root's (absent) shard
        # tree, rather than a hand-built GameStats: a planted boundary has no
        # shards, and "an iteration with nothing on disk stamps zeros instead
        # of raising" is worth exercising here (KOM-150). A caller that is
        # pinning *which* census number lands on *which* key hands one in.
        stats=game_stats(shard_root(root, 0)) if stats is None else stats,
        mutes={},
        previous_bn=None,
        pending_mute=None,
        results=[],
    )

    read = read_records(
        campaign_loop.journal_path(root), IterationMetrics, missing_ok=True
    )
    written = [
        record
        for record in read.records
        if record.extras.get(boundary.CAMPAIGN_MARKER, 0.0) == 1.0
    ]
    assert len(written) == 1, (
        f"want exactly one campaign record for the boundary, got "
        f"{[sorted(record.extras) for record in written]}"
    )
    return written[0].extras


def test_the_trainers_extras_reach_the_campaign_record_through_the_journal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F3 / M10: carried across, not recomputed -- provably, from a sentinel.

    The equality assertion in the behavioural suite cannot tell "read the
    trainer's number" from "recompute the same number here", because a faithful
    re-derivation over the same step history is bit-identical. This can: the
    trainer's record on disk claims a first-decile mean of ``12345.5``, which is
    not a statistic of the ``{1.0, 0.9}`` history the boundary holds, so a loop
    that re-derived would write ``1.0`` and a loop that carries across writes
    the sentinel.

    The whole record crosses, not only the deciles: ``loss_total``, ``steps``
    and ``batch_size`` are what the staging ladder reads beside them.
    """
    root = tmp_path / "root"
    root.mkdir()
    write_record(
        campaign_loop.journal_path(root),
        _trainer_record(
            0,
            loss_total=_SENTINEL,
            loss_total_first_decile=_SENTINEL,
            steps=99.0,
            batch_size=64.0,
        ),
    )

    written = _close_a_planted_boundary(root, monkeypatch)

    assert written.get("loss_total_first_decile") == _SENTINEL, (
        f"the decile mean on the campaign record is "
        f"{written.get('loss_total_first_decile')!r}, not the trainer's "
        f"{_SENTINEL!r}: it was recomputed here rather than carried across"
    )
    assert written.get("loss_total") == _SENTINEL
    assert written.get("steps") == 99.0
    assert written.get("batch_size") == 64.0


def test_the_boundarys_own_keys_win_the_merge_against_a_colliding_trainer_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F4 / M11: "trainer first, so the boundary overrides", made observable.

    The documented precedence is unobservable while the two key namespaces are
    disjoint, so this plants the collision: the trainer's record claims a row
    count and a BatchNorm digest, both of which the boundary measures itself.
    The campaign record's subject is what the *boundary* saw -- ``rows`` is the
    self-play count this iteration wrote, and ``bn_digest_lo`` is the series a
    restarted campaign compares against -- so a merge in the other order would
    leave the record looking complete while quietly changing what two gates
    read.
    """
    root = tmp_path / "root"
    root.mkdir()
    write_record(
        campaign_loop.journal_path(root),
        _trainer_record(
            0,
            **{"rows": 999.0, boundary.BN_DIGEST_LO: _SENTINEL},
        ),
    )

    written = _close_a_planted_boundary(root, monkeypatch)

    assert written["rows"] == 4.0, (
        f"the trainer's planted row count won the merge: rows is "
        f"{written['rows']!r}, and the boundary counted 4"
    )
    assert written[boundary.BN_DIGEST_LO] != _SENTINEL, (
        "the trainer's planted BatchNorm digest won the merge; the series a "
        "restart compares against is the boundary's own"
    )


def test_the_boundary_stamps_the_censuss_game_shape_and_draft_on_the_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """KOM-150's four extras and KOM-156b's three, each from its own field.

    Every number handed in is distinct, and the two medians are distinct from
    each other: a stamping that swapped white for black, or that read
    ``turn_cap_games`` where ``early_shu_games`` belongs, produces a record
    that is complete and wrong -- and a curriculum ramp reads these keys.

    ``early_shu_rate`` is stamped as the **rate** (3 of 8 games) while the
    count travels to ``selfplay.jsonl``: the denominator is on both records,
    so the rate is recoverable either way, and the ramp gate reads it here.
    """
    root = tmp_path / "root"
    root.mkdir()
    census = GameStats(
        games=8,
        rows=4,
        draws=1,
        turn_cap_games=2,
        mean_plies=100.0,
        median_plies=90.0,
        max_plies=256,
        median_placements_w=3.0,
        median_placements_b=12.0,
        # Three of eight, so the early-終 rate is 0.375 and cannot be confused
        # with the cap rate beside it.
        early_shu_games=3,
        # KOM-245: one of the two cap games carries a soft label, and the three
        # numbers are pairwise distinct -- and distinct from every number above
        # -- so a stamping that read one field where another belongs is visible.
        cap_adjudicated_games=1,
        cap_p_mean=0.2,
        cap_p_ge_03_share=0.125,
    )

    written = _close_a_planted_boundary(root, monkeypatch, stats=census)

    assert written[campaign_loop.DRAWS] == 1.0
    assert written[campaign_loop.TURN_CAP_GAMES] == 2.0
    assert written[campaign_loop.CAP_RATE] == pytest.approx(0.25)
    assert written[campaign_loop.MAX_PLIES] == 256.0
    assert written[campaign_loop.MEDIAN_PLACEMENTS_W] == pytest.approx(3.0)
    assert written[campaign_loop.MEDIAN_PLACEMENTS_B] == pytest.approx(12.0)
    assert written[campaign_loop.EARLY_SHU_RATE] == pytest.approx(0.375)
    # KOM-245's three, beside the cap rate they qualify: how the cap games were
    # LABELLED, which is what K1 is read off.
    assert written[campaign_loop.CAP_ADJUDICATED_GAMES] == 1.0
    assert written[campaign_loop.CAP_P_MEAN] == pytest.approx(0.2)
    assert written[campaign_loop.CAP_P_GE_03_SHARE] == pytest.approx(0.125)


def test_a_boundary_with_no_shards_stamps_the_draft_keys_as_zeros(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The three draft keys are present even when nothing was drafted.

    A key that appears only in the iterations that drafted is a key a ramp
    gate cannot read across a campaign: "absent" and "zero" would be the same
    lookup. The planted boundary has no shard tree at all, which is the
    strongest version of the case.
    """
    root = tmp_path / "root"
    root.mkdir()

    written = _close_a_planted_boundary(root, monkeypatch)

    for key in (
        campaign_loop.EARLY_SHU_RATE,
        campaign_loop.MEDIAN_PLACEMENTS_W,
        campaign_loop.MEDIAN_PLACEMENTS_B,
        # KOM-245's three are on the record for the same reason: a key that
        # appeared only after the switch is a key the pre-switch baseline
        # cannot be read from.
        campaign_loop.CAP_ADJUDICATED_GAMES,
        campaign_loop.CAP_P_MEAN,
        campaign_loop.CAP_P_GE_03_SHARE,
    ):
        assert written[key] == 0.0, f"{key} is {written.get(key)!r}, not a zero"


def test_a_healthy_boundary_records_the_step_its_ema_was_published_at(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F6: a *passing* EMA gate leaves the number it decided on, on disk.

    ``outcome.verdicts`` has no reader in ``src/``, so before this a healthy
    gate was indistinguishable on disk from a gate that had been deleted. The
    stamp is the file's own step, and the stall flag is absent when nothing
    stalled -- which is what makes counting stalls across a campaign possible.
    """
    root = tmp_path / "root"
    root.mkdir()

    written = _close_a_planted_boundary(root, monkeypatch)

    assert written[campaign_loop.EMA_PUBLISHED_STEP] == float(_BOUNDARY_STEPS), (
        f"the campaign record carries "
        f"{written.get(campaign_loop.EMA_PUBLISHED_STEP)!r} for the published "
        f"EMA step; iteration 0 owed step {_BOUNDARY_STEPS}"
    )
    assert pinned.EMA_PUBLISH_STALLED not in written, (
        "a healthy boundary flagged a stalled EMA publish"
    )


def test_the_trainers_key_namespace_does_not_collide_with_the_boundarys(
    tmp_path: Path,
) -> None:
    """F4's other half, over the **production** recipe rather than a fixture.

    The precedence above only matters where the namespaces meet; this states
    that today they do not, for the keys ``training_iteration`` actually writes
    -- so the day a loss head named ``tail_median`` or ``spike`` is added, this
    goes red at the point the collision is introduced rather than silently
    changing what ``spike_baseline`` reads back across a restart.
    """
    journal = campaign_loop.journal_path(tmp_path)
    net = _tiny_net()
    training_iteration(
        net,
        [_sample(row) for row in range(_ROWS)],
        _step(net),
        steps=_STEPS,
        rng_seed=0,
        checkpoint_dir=tmp_path / "ckpt",
        iteration=0,
        global_step_base=0,
        journal_path=journal,
    )

    produced = set(campaign_loop.trainer_extras(journal, 0))

    assert produced, "the trainer wrote no extras; this assertion would be vacuous"
    assert produced & _BOUNDARY_OWNED == set(), (
        f"the trainer writes {sorted(produced & _BOUNDARY_OWNED)}, which the "
        f"boundary also owns: the merge's precedence now decides what several "
        f"gates read back"
    )


def test_the_cap_label_extras_are_boundary_owned() -> None:
    """KOM-245's three keys are in :data:`_BOUNDARY_OWNED`, so the tests above bite.

    They are a census of the *games*, read off the shard tree the boundary
    owns; the trainer has no view of a cap label at all. Membership is what
    puts them inside the two disjointness assertions -- a later trainer-side
    stamp of ``cap_p_mean`` would then go red where it is introduced instead of
    silently deciding, through the merge's precedence, which of two numbers
    K1's gate reads back.
    """
    assert {
        campaign_loop.CAP_P_MEAN,
        campaign_loop.CAP_P_GE_03_SHARE,
        campaign_loop.CAP_ADJUDICATED_GAMES,
    } <= _BOUNDARY_OWNED


# --------------------------------------------------------------------------
# KOM-130 -- the merge as a pure function: its precedence, and the namespaces
# --------------------------------------------------------------------------

_TRAINER_SIDE: Final[float] = 1.0
_BOUNDARY_SIDE: Final[float] = 2.0
"""The two sides of a planted collision, distinct so the winner is readable.

Neither is a value the merge could synthesise: :func:`boundary_extras` copies
whatever it was handed, so a result of ``1.0`` under
:data:`~komugi.campaign.boundary.LOSS_TAIL_MEDIAN` can only mean the trainer's
mapping was applied *over* the boundary's.
"""

_ENABLED_SUFFIX: Final[str] = "_enabled"
"""How a :class:`~komugi.train.losses.LossConfig` field names a head's switch."""

_HEAD_PREFIX: Final[str] = "HEAD_"
"""Prefix of the ``losses`` constants naming a key of the per-head breakdown."""

_SUMMARY_SUFFIXES: Final[tuple[str, ...]] = (
    "",
    "_first_decile",
    "_last_decile",
    "_step_std",
)
"""The four forms ``training_iteration`` writes per loss scalar.

The last step's value under the bare name, plus KOM-005b's two decile means and
the per-step dispersion (``step.py``'s telemetry-mapping docstring).
"""

_FIXED_TRAINER_KEYS: Final[frozenset[str]] = frozenset(
    {
        "steps",
        "batch_size",
        # The knobs that REDEFINE a journalled loss scalar, which is why the
        # trainer stamps them beside the scalars themselves (`step.py`'s
        # telemetry-mapping docstring). `mlh_utility_lambda` is deliberately
        # NOT here: it is a property of the games and the boundary is its
        # single writer (KOM-171), so it sits in `_BOUNDARY_OWNED` instead.
        "cap_row_value_weight",
        "mlh_cap_row_weight",
        "cap_plies",
    }
)
"""Every extras key the trainer writes that is not a loss scalar."""


def _trainer_key_namespace(recipe: train_losses.LossConfig) -> set[str]:
    """Return every ``extras`` key ``training_iteration`` writes under `recipe`.

    Derived from the recipe rather than listed: which loss scalars exist is
    ``recipe``'s decision (the assembler reports a head iff it is enabled), and
    a hand-typed list would go on passing unchanged the day a head is added --
    which is precisely the day the disjointness below stops being true.

    The head names and the ``<head>_enabled`` fields are one vocabulary, so the
    mapping is checked here rather than assumed: a head constant with no switch
    field would silently drop that head out of the namespace and make the
    collision check blind to it.

    An **over**-approximation, deliberately: a head whose composed sample
    weights carry zero mass is omitted from a given step's breakdown (KOM-036's
    discard arm on an all-fast batch), so a real run writes a subset of this.
    Over-approximating is the safe direction for a disjointness claim -- it can
    only report a collision the trainer might one day write, never miss one.
    """
    switches = {field.name for field in dataclasses.fields(recipe)}
    heads = {
        value
        for name, value in vars(train_losses).items()
        if name.startswith(_HEAD_PREFIX) and isinstance(value, str)
    }
    assert heads, "no HEAD_* constants found; this derivation would be vacuous"
    unswitched = sorted(
        head for head in heads if f"{head}{_ENABLED_SUFFIX}" not in switches
    )
    assert not unswitched, (
        f"the loss heads {unswitched} have no {_ENABLED_SUFFIX} field on "
        f"LossConfig, so this derivation cannot tell whether the trainer "
        f"reports them"
    )
    scalars = [
        head for head in sorted(heads) if getattr(recipe, f"{head}{_ENABLED_SUFFIX}")
    ]
    assert scalars, "the recipe enables no head; LossConfig should have refused it"
    return {
        f"{_METRIC_PREFIX}{scalar}{suffix}"
        for scalar in (*scalars, TOTAL_KEY)
        for suffix in _SUMMARY_SUFFIXES
    } | set(_FIXED_TRAINER_KEYS)


def _named_calls(tree: ast.AST, name: str) -> list[ast.Call]:
    """Return every ``name(...)`` call in `tree`, parsed rather than grepped."""
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == name
    ]


def test_the_boundarys_measurement_overrides_the_trainers_in_the_pure_merge() -> None:
    """F4 / M11 at the merge itself: trainer first, boundary over the top.

    ``_close_a_planted_boundary`` pins the same rule end to end, through a real
    boundary and two of the keys it happens to measure. This pins it where it is
    *stated* -- one call, one planted collision -- so the precedence survives
    the day the boundary stops measuring ``bn_digest_lo`` and there is nothing
    left at the far end to collide with.

    The collision is planted on :data:`~komugi.campaign.boundary.LOSS_TAIL_MEDIAN`
    deliberately: it is the boundary key sitting *inside* the trainer's
    ``loss_`` prefix, so it is the collision production will actually reach --
    a loss head named ``tail_median`` -- and it is the number
    :func:`~komugi.campaign.boundary.spike_baseline` reads back off the journal
    after a restart. Inverting the merge feeds that baseline the trainer's
    last-step total instead of the boundary's tail median, silently.
    """
    merged = campaign_loop.boundary_extras(
        {
            boundary.LOSS_TAIL_MEDIAN: _TRAINER_SIDE,
            f"{_METRIC_PREFIX}{TOTAL_KEY}": _SENTINEL,
        },
        {boundary.LOSS_TAIL_MEDIAN: _BOUNDARY_SIDE},
        rows=4,
        pending_mute=None,
    )

    assert merged[boundary.LOSS_TAIL_MEDIAN] == _BOUNDARY_SIDE, (
        f"the merged record carries {merged[boundary.LOSS_TAIL_MEDIAN]!r} for "
        f"{boundary.LOSS_TAIL_MEDIAN}; the boundary measured "
        f"{_BOUNDARY_SIDE!r} and the trainer's {_TRAINER_SIDE!r} won, so the "
        f"two mappings were applied in the wrong order"
    )
    assert merged[f"{_METRIC_PREFIX}{TOTAL_KEY}"] == _SENTINEL, (
        "a trainer key the boundary does not measure was dropped by the merge; "
        "the trainer's numbers ride along on the campaign's record"
    )
    assert merged[boundary.CAMPAIGN_MARKER] == 1.0, (
        "the merge did not stamp the marker every campaign-level reader filters "
        "on, so the record it produces is indistinguishable from the trainer's"
    )
    assert merged["rows"] == 4.0
    assert "mute_applied" not in merged, (
        "an unmuted boundary claimed a mute was applied"
    )


def test_the_pure_merge_stamps_the_mute_it_was_applying() -> None:
    """Step 7 of the ack protocol: telemetry *about* the mute, on the record."""
    merged = campaign_loop.boundary_extras(
        {},
        {},
        rows=0,
        pending_mute=Mute(
            tripwire=pinned.LOSS_SPIKE,
            acked_iteration=2,
            from_iteration=3,
            until_iteration=6,
        ),
    )

    assert merged["mute_applied"] == 1.0
    assert merged["mute_until_iteration"] == 6.0, (
        f"the record says the mute runs to "
        f"{merged.get('mute_until_iteration')!r}; the mute itself ends at 6"
    )


def test_the_production_recipes_trainer_keys_miss_the_boundarys_namespace() -> None:
    """F4's tripwire, over :data:`P0B_RECIPE` -- what the campaign actually runs.

    The precedence above only *matters* where the two namespaces meet, and
    ``test_the_trainers_key_namespace_does_not_collide_with_the_boundarys``
    states they do not for the keys a P0a fixture writes. This states it for the
    recipe production trains under, derived from the recipe rather than from a
    run: it needs no trainer, no shard tree and no net, so it stays true of a
    head that is enabled long before anyone writes a test that trains it.

    It goes red the day a loss head is named ``tail_median`` or ``spike`` --
    at the point the collision is introduced, naming the colliding keys, rather
    than three months later as a ``spike_baseline`` that reads the trainer's
    last-step total back off the journal after a restart.
    """
    produced = _trainer_key_namespace(P0B_RECIPE)

    assert f"{_METRIC_PREFIX}{TOTAL_KEY}" in produced, (
        "the derivation lost the weighted total, which every recipe reports"
    )
    # The derivation reads the recipe rather than reciting a fixed list:
    # enabling one of P0b's staged heads has to widen the namespace. This is
    # also the motion the ladder makes, one head at a time -- so it is the
    # motion this tripwire has to keep seeing.
    staged = sorted(
        field.name
        for field in dataclasses.fields(P0B_RECIPE)
        if field.name.endswith(_ENABLED_SUFFIX)
        and not getattr(P0B_RECIPE, field.name)
    )
    assert staged, "P0B_RECIPE enables every head; this widening cannot be shown"
    widened = _trainer_key_namespace(
        dataclasses.replace(P0B_RECIPE, **{staged[0]: True})
    )
    assert produced < widened, (
        f"enabling {staged[0]} did not widen the trainer's key namespace: the "
        f"derivation is not reading the recipe it was handed, so it would go "
        f"on describing P0b after the ladder has moved past it"
    )

    collisions = produced & _BOUNDARY_OWNED
    assert collisions == set(), (
        f"the trainer writes {sorted(collisions)} under P0B_RECIPE, which the "
        f"boundary also owns: the merge's precedence now decides what "
        f"{sorted(collisions)} means to every gate that reads it back off the "
        f"journal"
    )


def test_the_merge_is_still_one_call_from_the_boundary() -> None:
    """The extraction cannot be quietly re-inlined, nor half-reverted.

    Read off ``_close_iteration``'s own AST, because the alternative -- running
    a boundary and inspecting the record -- cannot distinguish "calls the pure
    merge" from "does the same thing inline again", which is exactly the state
    this ticket removed. Two claims:

    * the merge happens **once**, so a second inline build of ``extras`` (the
      plausible refactor accident) is red rather than a silent second
      precedence;
    * the trainer's extras reach the record only *through* it -- every
      ``trainer_extras`` call in the boundary is an argument of that one merge,
      so a re-inlined ``extras.update(...)`` beside it is red too.
    """
    tree = ast.parse(
        textwrap.dedent(inspect.getsource(campaign_loop._close_iteration))
    )
    merges = _named_calls(tree, campaign_loop.boundary_extras.__name__)
    carried = _named_calls(tree, campaign_loop.trainer_extras.__name__)

    assert len(merges) == 1, (
        f"_close_iteration calls boundary_extras {len(merges)} times; the "
        f"record's extras are built by exactly one merge, and a second one is "
        f"a second precedence nothing states"
    )
    assert carried, (
        "_close_iteration no longer reads the trainer's extras at all; the "
        "campaign record would carry no loss scalars and no decile means"
    )
    orphans = [call for call in carried if call not in merges[0].args]
    assert not orphans, (
        f"{len(orphans)} trainer_extras call(s) in _close_iteration are not an "
        f"argument of the boundary_extras merge: the trainer's keys reach the "
        f"record by some other path, whose precedence is unstated"
    )


# --------------------------------------------------------------------------
# KOM-036 §1.7 -- the arm's wire from the campaign to the trainer (#30)
# --------------------------------------------------------------------------

_ARM_A: Final[float] = 0.125
"""KOM-036 §1.3's derived down-weight, ``gumbel_n_fast / gumbel_n_full``."""

_ARM_FIELD: Final[str] = "fast_policy_weight"
"""The one ``CampaignConfig`` field that differs between the two arms."""


def _armed_campaign(weight: float) -> CampaignConfig:
    """A minimal valid campaign carrying one A/B arm."""
    return CampaignConfig(
        iterations=1,
        run_seed=0,
        root="/tmp/kom-036-arm",
        steps_per_iteration=1,
        batch_size=1,
        lr=1e-3,
        workers=1,
        games_per_worker=1,
        fast_policy_weight=weight,
    )


def _step_config_calls(source: str) -> list[ast.Call]:
    """Return every ``StepConfig(...)`` call in `source`, parsed rather than grepped."""
    tree = ast.parse(textwrap.dedent(source))
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "StepConfig"
    ]


def test_the_campaign_passes_the_arm_to_the_step_config() -> None:
    """#30: ``CampaignConfig.fast_policy_weight`` reaches ``StepConfig``.

    This is the wire without which the KOM-036 experiment does not exist: the
    two arms are two campaigns that differ in exactly one field, and that field
    has to arrive at the trainer or both campaigns run arm C and the whole
    comparison is a null result nobody can attribute.

    **Three assertions, and the third is structural on purpose.** The first two
    are executed: the value survives ``CampaignConfig``, it survives
    ``StepConfig``, and both refuse it outside ``[0, 1]``. The third reads
    ``run_campaign``'s own AST, because the call site sits after a real
    self-play iteration -- a spawned server, a played game, a written shard --
    and standing that up to observe one keyword argument would cost minutes per
    run and would test the orchestrator rather than the wire.

    What the AST check is red against is exactly the mutation that matters:
    dropping the keyword (``StepConfig.fast_policy_weight`` then defaults to
    ``1.0``, i.e. arm C, silently), hard-coding a literal, or passing a
    different attribute of the campaign. It is also red if a *second*
    ``StepConfig`` is built in ``run_campaign`` without the arm, which is how
    this wire would most plausibly be lost in a refactor.
    """
    # (1) the campaign holds the arm, and refuses what is not an arm.
    assert _armed_campaign(_ARM_A).fast_policy_weight == _ARM_A
    assert _campaign(Path("/tmp/kom-036-arm-c")).fast_policy_weight == 1.0, (
        "the default arm is C (today's behaviour), so a campaign that says "
        "nothing keeps training exactly as it did before KOM-036"
    )
    for refused in (-0.1, 1.5, float("nan")):
        with pytest.raises(ValueError, match=_ARM_FIELD):
            _armed_campaign(refused)

    # (2) the trainer's config holds it too, with the same bound.
    assert (
        StepConfig(batch_size=2, lr=1e-3, fast_policy_weight=_ARM_A)
        .fast_policy_weight
        == _ARM_A
    )
    for refused in (-0.1, 1.5, float("nan")):
        with pytest.raises(ValueError, match=_ARM_FIELD):
            StepConfig(batch_size=2, lr=1e-3, fast_policy_weight=refused)

    # (3) the wire itself.
    calls = _step_config_calls(inspect.getsource(campaign_loop.run_campaign))
    assert len(calls) == 1, (
        f"run_campaign builds {len(calls)} StepConfigs; every one of them has "
        f"to carry the arm, so this node asserts there is exactly one"
    )
    keywords = {
        keyword.arg: keyword.value
        for keyword in calls[0].keywords
        if keyword.arg is not None
    }
    assert _ARM_FIELD in keywords, (
        f"run_campaign's StepConfig does not pass {_ARM_FIELD}; it therefore "
        f"defaults to 1.0 and BOTH arms of the A/B would train as arm C"
    )
    passed = keywords[_ARM_FIELD]
    assert isinstance(passed, ast.Attribute) and passed.attr == _ARM_FIELD, (
        f"run_campaign passes {ast.dump(passed)} as {_ARM_FIELD}; it must pass "
        f"the campaign's own field, not a literal or a derived value"
    )
    assert isinstance(passed.value, ast.Name) and passed.value.id == "campaign"


# --------------------------------------------------------------------------
# KOM-160 -- the cap-row value weight's wire from the campaign to the trainer
# --------------------------------------------------------------------------

_CAP_ARM: Final[float] = 0.25
"""Campaign 4D's pre-registered arm (design §2 "The number")."""

_CAP_FIELD: Final[str] = "cap_row_value_weight"
"""The one ``CampaignConfig`` field that differs between 4A and 4D."""


def _cap_armed_campaign(weight: float) -> CampaignConfig:
    """A minimal valid campaign carrying one cap-row arm."""
    return CampaignConfig(
        iterations=1,
        run_seed=0,
        root="/tmp/kom-160-arm",
        steps_per_iteration=1,
        batch_size=1,
        lr=1e-3,
        workers=1,
        games_per_worker=1,
        cap_row_value_weight=weight,
    )


def test_cap_row_value_weight_reaches_step_config() -> None:
    """KOM-160: ``CampaignConfig.cap_row_value_weight`` reaches ``StepConfig``.

    The same wire, and the same three-assertion shape, as the KOM-036 node
    above -- including the structural third, for the same reason: the call site
    sits after a real self-play iteration, and standing that up to observe one
    keyword argument would test the orchestrator rather than the wire.

    What the AST check is red against is the mutation that matters: dropping
    the keyword, at which point ``StepConfig.cap_row_value_weight`` silently
    defaults to ``1.0`` and campaign 4D trains identically to its own control,
    producing a null result nobody could attribute to the lever.
    """
    # (1) the campaign holds the arm, and its default is the identity.
    assert _cap_armed_campaign(_CAP_ARM).cap_row_value_weight == _CAP_ARM
    assert _campaign(Path("/tmp/kom-160-default")).cap_row_value_weight == 1.0, (
        "the default is 1.0, so a campaign that says nothing trains exactly as "
        "it did before KOM-160"
    )

    # (2) the trainer's config holds it too, with the same bound.
    assert (
        StepConfig(batch_size=2, lr=1e-3, cap_row_value_weight=_CAP_ARM)
        .cap_row_value_weight
        == _CAP_ARM
    )

    # (3) the wire itself.
    calls = _step_config_calls(inspect.getsource(campaign_loop.run_campaign))
    assert len(calls) == 1, (
        f"run_campaign builds {len(calls)} StepConfigs; every one of them has "
        f"to carry the arm, so this node asserts there is exactly one"
    )
    keywords = {
        keyword.arg: keyword.value
        for keyword in calls[0].keywords
        if keyword.arg is not None
    }
    assert _CAP_FIELD in keywords, (
        f"run_campaign's StepConfig does not pass {_CAP_FIELD}; it therefore "
        f"defaults to 1.0 and 4D would train as its own control"
    )
    passed = keywords[_CAP_FIELD]
    assert isinstance(passed, ast.Attribute) and passed.attr == _CAP_FIELD, (
        f"run_campaign passes {ast.dump(passed)} as {_CAP_FIELD}; it must pass "
        f"the campaign's own field, not a literal or a derived value"
    )
    assert isinstance(passed.value, ast.Name) and passed.value.id == "campaign"


def test_cap_row_value_weight_outside_unit_interval_is_refused_before_spawn() -> None:
    """The campaign refuses a non-arm before it spawns a single worker.

    ``StepConfig`` repeats the check because it is constructible without a
    campaign, but a campaign that only found out at its first optimiser step
    would already have played an iteration of self-play -- minutes of GPU, and
    a shard tree -- for a run that cannot train.

    NaN is in the sweep because the predicate is ``not 0.0 <= w <= 1.0``: the
    ``<``/``>`` spelling admits it, and the failure would then be a WDL loss
    that is NaN from the first batch with nothing naming the field.
    """
    for refused in (1.5, -0.1, float("nan")):
        with pytest.raises(ValueError, match=_CAP_FIELD):
            _cap_armed_campaign(refused)
        with pytest.raises(ValueError, match=_CAP_FIELD):
            StepConfig(batch_size=2, lr=1e-3, cap_row_value_weight=refused)


# --------------------------------------------------------------------------
# KOM-159 -- the moves-left levers' wire from the campaign to the trainer,
# and the two keys every iteration's record has to carry (design review F10)
# --------------------------------------------------------------------------

_MLH_ARM: Final[float] = 0.0
"""Campaign 6's pre-registered censoring arm: cap rows discarded outright."""

_MLH_FIELD: Final[str] = "mlh_cap_row_weight"
"""The ``CampaignConfig`` field carrying that arm."""

_CAP_PLIES_FIELD: Final[str] = "cap_plies"
"""Not an arm -- the CAP the censoring predicate is evaluated against."""

_RECIPE_FIELD: Final[str] = "loss_recipe"
"""The field naming which entry of ``LOSS_RECIPES`` the trainer runs under."""

_MAX_TURNS_METHOD: Final[str] = "max_turns"
"""The engine classmethod the loop reads the cap off."""


def _mlh_armed_campaign(
    weight: float = _MLH_ARM, recipe: str = campaign_loop.RECIPE_MLH
) -> CampaignConfig:
    """A minimal valid campaign carrying one moves-left arm and one recipe."""
    return CampaignConfig(
        iterations=1,
        run_seed=0,
        root="/tmp/kom-159-arm",
        steps_per_iteration=1,
        batch_size=1,
        lr=1e-3,
        workers=1,
        games_per_worker=1,
        loss_recipe=recipe,
        mlh_cap_row_weight=weight,
    )


def test_the_campaign_passes_the_moves_left_arm_and_the_engines_cap_to_the_step() -> None:
    """KOM-159 D4: ``mlh_cap_row_weight`` **and** ``cap_plies`` reach ``StepConfig``.

    Two keywords, and they fail differently.

    The *arm* is the same wire as KOM-036's and KOM-160's above, with the same
    consequence when it is dropped: ``StepConfig.mlh_cap_row_weight`` defaults
    to ``0.0``, so a campaign configured to keep its cap rows would censor them
    and the comparison would be a null nobody could attribute.

    The *cap* is the subtler one, and it is why this node exists rather than
    resting on the trainer's default. ``DEFAULT_CAP_PLIES`` equals
    ``Game.max_turns()`` **today**, which is asserted below -- so dropping the
    keyword changes no number now, goes green everywhere, and silently censors
    against a stale cap the next time KOM-165's successor bumps the engine's.
    That is a mutation only the call site itself can see, and the value passed
    has to be the engine's own call rather than a literal for the same reason.

    The third assertion is structural, for the reason the two nodes above give:
    ``run_campaign``'s ``StepConfig`` sits after a real self-play iteration, and
    standing that up to read two keyword arguments would test the orchestrator.
    """
    # (1) the campaign holds the arm and the recipe, and their defaults are the
    # pre-KOM-159 behaviour.
    assert _mlh_armed_campaign().mlh_cap_row_weight == _MLH_ARM
    assert _mlh_armed_campaign().loss_recipe == campaign_loop.RECIPE_MLH
    default = _campaign(Path("/tmp/kom-159-default"))
    assert default.loss_recipe == campaign_loop.RECIPE_P0B, (
        "the default recipe is P0b, so a campaign that says nothing trains "
        "exactly as campaigns 1-5 did"
    )
    assert default.mlh_cap_row_weight == 0.0, (
        "the default censors cap rows outright, which is inert under P0b -- the "
        "moves-left head is off there, so the composed vector reaches no loss"
    )

    # (2) the trainer's config holds both, and its fallback cap is the engine's
    # own number -- the equality that makes assertion (3) the only falsifier of
    # a dropped `cap_plies`.
    held = StepConfig(
        batch_size=2, lr=1e-3, mlh_cap_row_weight=_MLH_ARM, cap_plies=Game.max_turns()
    )
    assert held.mlh_cap_row_weight == _MLH_ARM
    assert held.cap_plies == Game.max_turns()
    assert StepConfig(batch_size=2, lr=1e-3).cap_plies == Game.max_turns(), (
        f"the trainer's fallback cap is "
        f"{StepConfig(batch_size=2, lr=1e-3).cap_plies} and the engine's is "
        f"{Game.max_turns()}: they agree today, which is exactly why dropping "
        f"the keyword from run_campaign is invisible to every executed test"
    )

    # (3) the wire itself.
    calls = _step_config_calls(inspect.getsource(campaign_loop.run_campaign))
    assert len(calls) == 1, (
        f"run_campaign builds {len(calls)} StepConfigs; every one of them has "
        f"to carry the arm and the cap, so this node asserts there is exactly "
        f"one"
    )
    keywords = {
        keyword.arg: keyword.value
        for keyword in calls[0].keywords
        if keyword.arg is not None
    }

    assert _MLH_FIELD in keywords, (
        f"run_campaign's StepConfig does not pass {_MLH_FIELD}; it therefore "
        f"defaults to 0.0 and every campaign would censor its cap rows"
    )
    arm = keywords[_MLH_FIELD]
    assert isinstance(arm, ast.Attribute) and arm.attr == _MLH_FIELD, (
        f"run_campaign passes {ast.dump(arm)} as {_MLH_FIELD}; it must pass the "
        f"campaign's own field, not a literal or a derived value"
    )
    assert isinstance(arm.value, ast.Name) and arm.value.id == "campaign"

    assert _CAP_PLIES_FIELD in keywords, (
        f"run_campaign's StepConfig does not pass {_CAP_PLIES_FIELD}; it "
        f"therefore falls back to the trainer's DEFAULT_CAP_PLIES, which equals "
        f"the engine's cap today and will not the next time the cap is bumped"
    )
    cap = keywords[_CAP_PLIES_FIELD]
    assert isinstance(cap, ast.Call), (
        f"run_campaign passes {ast.dump(cap)} as {_CAP_PLIES_FIELD}; it must "
        f"call the engine's own {_MAX_TURNS_METHOD}() rather than restate the "
        f"number the shards were written under"
    )
    assert (
        isinstance(cap.func, ast.Attribute)
        and cap.func.attr == _MAX_TURNS_METHOD
        and isinstance(cap.func.value, ast.Name)
        and cap.func.value.id == Game.__name__
    ), (
        f"run_campaign passes {ast.dump(cap)} as {_CAP_PLIES_FIELD}; want "
        f"{Game.__name__}.{_MAX_TURNS_METHOD}()"
    )

    # ... and the mapping the recipe is selected by NAME from, whose keys are
    # what `CampaignConfig.__post_init__`'s refusal is written against.
    assert set(campaign_loop.LOSS_RECIPES) == {
        campaign_loop.RECIPE_P0B,
        campaign_loop.RECIPE_MLH,
        campaign_loop.RECIPE_MLH4,  # KOM-159d: the weight-4.0 moves-left arm
    }, (
        f"LOSS_RECIPES holds {sorted(campaign_loop.LOSS_RECIPES)}; the campaign "
        f"config's refusal is written against this mapping's keys"
    )


@pytest.mark.parametrize(
    ("recipe", "lambda_"),
    [
        (campaign_loop.RECIPE_P0B, 0.0),
        (campaign_loop.RECIPE_MLH, 1.0),
    ],
)
def test_every_iterations_record_carries_the_moves_left_levers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recipe: str, lambda_: float
) -> None:
    """F10: both keys, on **every** iteration's record, at their configured values.

    Design review F10's rule is that a key written only when it is interesting
    is a key whose absence means two different things -- "the lever was off" and
    "this campaign predates the field" -- and the one question campaign 6 has to
    answer off the journal alone is *which iteration the utility came on*. So
    the P0b row of this sweep is not redundant with the MLH row: it is the claim
    that the keys are present carrying zero rather than absent.

    ``mlh_utility_lambda`` is the **boundary's** key and the boundary's alone
    (KOM-171): what is asserted is that the record carries the value the
    self-play config it was handed ran at. It used to be written by the trainer
    too, and this node used to plant a sentinel on the trainer's record to pin
    which side won the merge; with one writer there is no precedence left to
    observe, and the disjointness tests plus
    :data:`_BOUNDARY_OWNED` are what keep it that way. A planted trainer copy
    here would be a collision no production path can produce.

    ``draw_score=0.0`` because a positive lambda with contempt is refused
    outright (:class:`~komugi.config.SelfPlayConfig`, review F13).
    """
    root = tmp_path / "root"
    root.mkdir()

    written = _close_a_planted_boundary(
        root,
        monkeypatch,
        loss_recipe=recipe,
        selfplay=dataclasses.replace(
            _selfplay(), mlh_utility_lambda=lambda_, draw_score=0.0
        ),
    )

    assert written.get(campaign_loop.MLH_UTILITY_LAMBDA) == lambda_, (
        f"the campaign record says the search-side lambda was "
        f"{written.get(campaign_loop.MLH_UTILITY_LAMBDA)!r}; self-play ran at "
        f"{lambda_!r}. A record carrying nothing means the iteration the "
        f"utility came on is unreadable off the journal"
    )
    assert written.get(campaign_loop.LOSS_RECIPE_IS_MLH) == float(
        recipe == campaign_loop.RECIPE_MLH
    ), (
        f"the campaign record says loss_recipe_is_mlh="
        f"{written.get(campaign_loop.LOSS_RECIPE_IS_MLH)!r} for a campaign "
        f"running {recipe!r}: the flag is the only durable record of which "
        f"recipe trained this iteration"
    )


# --------------------------------------------------------------------------
# KOM-159 -- what the two configs refuse before a campaign spawns anything
# --------------------------------------------------------------------------


def test_an_unknown_loss_recipe_is_refused_before_spawn() -> None:
    """A typo'd recipe name is refused, never silently fallen back from.

    The failure this forbids is not a crash: a campaign 6 that fell back to P0b
    would run for days, produce a checkpoint series, and report a null result
    that nobody could attribute to either the recipe or the lever -- and the
    recipe's name is the only thing in the run that says which arithmetic the
    curve came from.

    ``"P0B"`` and ``"mlh "`` are in the sweep because a closed-mapping lookup is
    exact: a case fold or a stray space is precisely the typo a hand-written
    campaign config carries.
    """
    for refused in ("", "p0a", "P0B", "mlh ", "MLH"):
        with pytest.raises(ValueError, match=_RECIPE_FIELD):
            _mlh_armed_campaign(recipe=refused)
    for accepted in (campaign_loop.RECIPE_P0B, campaign_loop.RECIPE_MLH):
        assert _mlh_armed_campaign(recipe=accepted).loss_recipe == accepted


def test_a_moves_left_arm_outside_the_unit_interval_is_refused_before_spawn() -> None:
    """The third head-weight arm, on the same terms as the other two.

    NaN is in the sweep for the reason it is in KOM-160's: the predicate is
    ``not 0.0 <= w <= 1.0``, the ``<``/``>`` spelling admits it, and the failure
    would then be a moves-left loss that is NaN from the first batch with
    nothing naming the field. A campaign that only found out at its first
    optimiser step would already have played an iteration of self-play.
    """
    for refused in (1.5, -0.1, float("nan")):
        with pytest.raises(ValueError, match=_MLH_FIELD):
            _mlh_armed_campaign(weight=refused)
        with pytest.raises(ValueError, match=_MLH_FIELD):
            StepConfig(batch_size=2, lr=1e-3, mlh_cap_row_weight=refused)
    for accepted in (0.0, 0.5, 1.0):
        assert _mlh_armed_campaign(weight=accepted).mlh_cap_row_weight == accepted


def test_the_default_selfplay_config_leaves_the_moves_left_utility_off() -> None:
    """The premise of every refusal below: the defaults are accepted.

    ``mlh_utility_lambda`` at zero is what makes campaigns 1-5's search
    unchanged by KOM-159 -- the gate and the slope are configured, and inert,
    until the lambda is positive.
    """
    config = _selfplay()
    assert config.mlh_utility_lambda == 0.0
    assert 0.0 <= config.mlh_gate_tau0 < config.mlh_gate_tau1 <= 1.0
    assert config.mlh_slope_plies >= 1


@pytest.mark.parametrize("refused", [-0.1, 1.5, float("nan")])
def test_a_moves_left_lambda_outside_the_unit_interval_is_refused(
    refused: float,
) -> None:
    """The search-side lever's bound, NaN included.

    NaN because the predicate is ``not 0.0 <= lambda <= 1.0``: admitted by a
    ``<``/``>`` spelling, and the consequence is a search whose every gated leaf
    utility is NaN -- a policy target of nothing, written to shards, for a whole
    iteration.
    """
    with pytest.raises(ValueError, match="mlh_utility_lambda"):
        dataclasses.replace(_selfplay(), mlh_utility_lambda=refused, draw_score=0.0)


@pytest.mark.parametrize(
    ("tau0", "tau1"),
    [(0.4, 0.2), (0.2, 0.2), (-0.1, 0.4), (0.2, 1.5)],
)
def test_an_inverted_or_empty_moves_left_gate_is_refused(
    tau0: float, tau1: float
) -> None:
    """``0 <= tau0 < tau1 <= 1``, and the equal pair is refused too.

    The equal pair is the one that matters: ``tau1 - tau0`` is the ramp's
    denominator, so an empty band is a division by zero at the first gated leaf
    rather than the narrowest possible gate.
    """
    with pytest.raises(ValueError, match="tau"):
        dataclasses.replace(_selfplay(), mlh_gate_tau0=tau0, mlh_gate_tau1=tau1)


@pytest.mark.parametrize("refused", [0, -1, -32])
def test_a_moves_left_slope_below_one_ply_is_refused(refused: int) -> None:
    """The slope is the denominator of the moves-left difference."""
    with pytest.raises(ValueError, match="mlh_slope_plies"):
        dataclasses.replace(_selfplay(), mlh_slope_plies=refused)


def test_a_positive_moves_left_lambda_with_contempt_is_refused() -> None:
    """Design review F13: the conversion term is pre-registered at contempt 0.

    The two terms compose badly and have never been measured together -- the
    conversion term is odd under the frame swap and the contempt term is even,
    so it oscillates with depth parity. A run that asked for both would produce
    a policy target attributable to neither lever, which is why it is refused at
    construction rather than discovered in the analysis.

    Both halves are asserted: contempt with the lambda off is the configuration
    campaigns 1-5 actually ran, so refusing *that* would be a regression.
    """
    with pytest.raises(ValueError, match="mlh_utility_lambda"):
        dataclasses.replace(_selfplay(), mlh_utility_lambda=0.5, draw_score=0.20)
    assert (
        dataclasses.replace(
            _selfplay(), mlh_utility_lambda=0.0, draw_score=0.20
        ).draw_score
        == 0.20
    )
    assert (
        dataclasses.replace(
            _selfplay(), mlh_utility_lambda=0.5, draw_score=0.0
        ).mlh_utility_lambda
        == 0.5
    )
