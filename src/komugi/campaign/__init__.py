"""The campaign: the driver that turns the round trip into a training run.

Self-play (:mod:`komugi.selfplay`) and the trainer (:mod:`komugi.train`) are
each complete and each stops at the other's door. This package is the loop that
chains them -- play an iteration, mix it into the recency window, train, publish
a checkpoint, serve *that* checkpoint next iteration -- plus the two things that
only exist once there is a sequence: a globally monotone checkpoint step so
iterations do not overwrite each other, and a resume that costs at most the
iteration that was running.

* :mod:`komugi.campaign.factory` -- the module-level, no-argument serving-net
  factory the orchestrator requires, configured through the environment
  because a spawned server cannot receive a closure.
* :mod:`komugi.campaign.loop` -- :func:`~komugi.campaign.loop.run_campaign`
  and the configuration it reads.
* :mod:`komugi.campaign.telemetry` -- the per-iteration self-play record
  (games, rows, restarts, duration) the trainer's journal cannot hold, written
  to ``<root>/selfplay.jsonl`` the moment self-play returns.
"""

from __future__ import annotations

from komugi.campaign.factory import (
    CHECKPOINT_ENV,
    CONFIG,
    DEVICE_ENV,
    NET_ENV,
    SMALL_CONFIG,
    SMALL_NET,
    net_config,
    new_net,
    serving_net,
    small_serving_net,
)
from komugi.campaign.loop import (
    CampaignConfig,
    IterationResult,
    ResumePoint,
    WindowSource,
    checkpoint_dir,
    ema_dir,
    iteration_seed,
    iteration_source,
    journal_path,
    resume_point,
    run_campaign,
    shard_root,
    window_refs,
)
from komugi.campaign.pause import CampaignPaused
from komugi.campaign.telemetry import (
    SELFPLAY_JOURNAL_NAME,
    SelfPlayTelemetry,
    append_selfplay,
    read_selfplay,
    selfplay_journal_path,
)

__all__ = [
    "CHECKPOINT_ENV",
    "CONFIG",
    "DEVICE_ENV",
    "NET_ENV",
    "SELFPLAY_JOURNAL_NAME",
    "SMALL_CONFIG",
    "SMALL_NET",
    "CampaignConfig",
    "CampaignPaused",
    "IterationResult",
    "ResumePoint",
    "SelfPlayTelemetry",
    "WindowSource",
    "append_selfplay",
    "checkpoint_dir",
    "ema_dir",
    "iteration_seed",
    "iteration_source",
    "journal_path",
    "net_config",
    "new_net",
    "read_selfplay",
    "resume_point",
    "run_campaign",
    "selfplay_journal_path",
    "serving_net",
    "shard_root",
    "small_serving_net",
]
