"""The trainer's data and loss core -- the middle third of the round trip.

Self-play produces samples; the net consumes gradients. Between them sits the
machinery in this package, which the v1 codebase never owned as a unit and
which the design review flagged as unowned: where samples are *stored*, how a
batch is *drawn* from recent history, what the loss *is*, and how state
survives a preemption.

* :mod:`komugi.train.shards` -- compressed sample shards, written so that a
  kill forfeits at most one ``flush_every_seconds`` interval's worth of
  *appended* samples (the trigger is checked in ``append``; a caller wanting a
  wall-clock bound heartbeats ``ShardWriter.flush``) and never a shard that
  has already been published, and read so that a damaged shard costs that
  shard alone.
* :mod:`komugi.train.window` -- the §4 replay window: the 5 most recent
  iterations with a documented geometric recency tilt, deterministic under a
  seed.
* :mod:`komugi.train.losses` -- per-head weighted loss assembly over **eight**
  heads (policy CE against a full distribution, KataGo soft policy, WDL,
  moves-left, plus KOM-005b's four staged auxiliaries: uncertainty, ownership,
  short-term value and draft outcome) plus KataGo policy surprise weighting
  from priors recorded at self-play time.
* :mod:`komugi.train.checkpoint` -- atomic checkpoint writes whose committed
  filename carries a digest of the bytes, and a loader that walks back to the
  newest whole one.
* :mod:`komugi.train.wiring` -- the single translation between
  :class:`~komugi.net.model.NetOutput`'s field names, the sample schema's
  target names and the assembler's keys, plus the P0a loss recipe
  (:data:`~komugi.train.wiring.P0A_RECIPE`) and its P0b successor
  (:data:`~komugi.train.wiring.P0B_RECIPE`, the four staged heads present at
  ``enabled=False, weight=0.0``).
* :mod:`komugi.train.step` -- one optimiser step (AdamW, built inside) and the
  loop that runs a few thousand of them, checkpointing and journalling at the
  end.

Network-agnostic *at runtime*: nothing here imports :mod:`komugi.net` outside a
``TYPE_CHECKING`` block, and every entry point takes tensors, callables or
state dicts. :mod:`komugi.train.wiring` names :class:`NetOutput` in its
annotations -- that is the module whose whole job is knowing what the network
emits -- but reads it structurally, which is why the tests below still run
against three-line ``nn.Module``s and hand-typed numbers as well as against a
real net.
"""

from __future__ import annotations

from komugi.train.checkpoint import (
    CHECKPOINT_GLOB,
    FORMAT_VERSION,
    Checkpoint,
    CheckpointRef,
    CheckpointWarning,
    list_checkpoints,
    load_checkpoint,
    save_checkpoint,
)
from komugi.train.losses import (
    DRAFT_OUTCOME_LOGITS,
    DRAFT_OUTCOME_TARGET,
    FINAL_CONTROL_TARGET,
    HEAD_DRAFT_OUTCOME,
    HEAD_MLH,
    HEAD_OWNERSHIP,
    HEAD_POLICY,
    HEAD_SHORT_TERM_VALUE,
    HEAD_SOFT_POLICY,
    HEAD_UNCERTAINTY,
    HEAD_WDL,
    MLH_PRED,
    MLH_TARGET,
    OWNERSHIP_LOGITS,
    POLICY_LOGITS,
    POLICY_TARGET,
    PRIOR_LOGITS,
    SHORT_TERM_VALUE_PRED,
    SHORT_TERM_VALUE_TARGET,
    SOFT_POLICY_LOGITS,
    UNCERTAINTY_PRED,
    WDL_LOGITS,
    WDL_TARGET,
    LossAssembler,
    LossConfig,
    ownership_target,
    policy_surprise_weight,
    soft_policy_target,
    uncertainty_target,
)
from komugi.train.shards import (
    COUNT_KEY,
    DIGEST_KEY,
    FINGERPRINT_KEY,
    RESERVED_KEYS,
    SCHEMA_KEY,
    SHARD_GLOB,
    FieldSpec,
    Schema,
    ShardInfo,
    ShardReader,
    ShardWarning,
    ShardWriter,
    SkippedShard,
    stack_samples,
)
from komugi.train.step import (
    OBSERVATION_KEY,
    POLICY_KL_KEY,
    TOTAL_KEY,
    StepConfig,
    TrainStep,
    training_iteration,
)
from komugi.train.window import (
    IterationRef,
    RecencyWindow,
    SampleSource,
    WindowConfig,
    recency_weights,
)
from komugi.train.wiring import (
    P0A_RECIPE,
    P0B_RECIPE,
    TARGET_KEYS,
    outputs_from_net,
    surprise_weights_from_kl,
    targets_from_batch,
)

__all__ = [
    "CHECKPOINT_GLOB",
    "COUNT_KEY",
    "DIGEST_KEY",
    "DRAFT_OUTCOME_LOGITS",
    "DRAFT_OUTCOME_TARGET",
    "FINAL_CONTROL_TARGET",
    "FINGERPRINT_KEY",
    "FORMAT_VERSION",
    "HEAD_DRAFT_OUTCOME",
    "HEAD_MLH",
    "HEAD_OWNERSHIP",
    "HEAD_POLICY",
    "HEAD_SHORT_TERM_VALUE",
    "HEAD_SOFT_POLICY",
    "HEAD_UNCERTAINTY",
    "HEAD_WDL",
    "MLH_PRED",
    "MLH_TARGET",
    "OBSERVATION_KEY",
    "OWNERSHIP_LOGITS",
    "P0A_RECIPE",
    "P0B_RECIPE",
    "POLICY_KL_KEY",
    "POLICY_LOGITS",
    "POLICY_TARGET",
    "PRIOR_LOGITS",
    "RESERVED_KEYS",
    "SCHEMA_KEY",
    "SHARD_GLOB",
    "SHORT_TERM_VALUE_PRED",
    "SHORT_TERM_VALUE_TARGET",
    "SOFT_POLICY_LOGITS",
    "TARGET_KEYS",
    "TOTAL_KEY",
    "UNCERTAINTY_PRED",
    "WDL_LOGITS",
    "WDL_TARGET",
    "Checkpoint",
    "CheckpointRef",
    "CheckpointWarning",
    "FieldSpec",
    "IterationRef",
    "LossAssembler",
    "LossConfig",
    "RecencyWindow",
    "SampleSource",
    "Schema",
    "ShardInfo",
    "ShardReader",
    "ShardWarning",
    "ShardWriter",
    "SkippedShard",
    "StepConfig",
    "TrainStep",
    "WindowConfig",
    "list_checkpoints",
    "load_checkpoint",
    "outputs_from_net",
    "ownership_target",
    "policy_surprise_weight",
    "recency_weights",
    "save_checkpoint",
    "soft_policy_target",
    "stack_samples",
    "surprise_weights_from_kl",
    "targets_from_batch",
    "training_iteration",
    "uncertainty_target",
]
