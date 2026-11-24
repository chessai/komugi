"""Campaign telemetry: metrics that survive, and gates that notice a dead run.

Written against two specific v1 failures. Telemetry was lost twice to log
truncation on relaunch, and a campaign ran six iterations on a self-play net
that had silently stopped updating. Neither was a subtle bug; both were
invisible because the evidence needed to see them was never durably recorded.

This package makes both structurally hard rather than merely discouraged:

* :mod:`komugi.telemetry.records` -- frozen, self-validating
  :class:`~komugi.telemetry.records.IterationMetrics` and
  :class:`~komugi.telemetry.records.NetFingerprint`. The fingerprint is the
  record whose absence made the frozen loop undetectable.
* :mod:`komugi.telemetry.journal` -- append-only JSONL, one fsync per record,
  and a reader that tolerates a crash-truncated trailing line instead of
  losing the file.
* :mod:`komugi.telemetry.liveness` -- pure gate functions returning a
  :class:`~komugi.telemetry.liveness.LivenessVerdict`: the frozen-self-play
  detector, an EMA/train KL band, and a log-staleness bound.
* :mod:`komugi.telemetry.archive` -- truncation is reachable only by first
  archiving, under a name derived from the log's own mtime.

Import-light and engine-independent: nothing here imports a net, a search or
an inference backend, so the gates can be tested from literals.
"""

from __future__ import annotations

from komugi.telemetry.archive import (
    TruncationToken,
    archive_before_truncate,
    archive_name,
    guard_truncation,
)
from komugi.telemetry.journal import (
    ITERATION_METRICS_KIND,
    NET_FINGERPRINT_KIND,
    SCHEMA_VERSION,
    JournalDecodeError,
    JournalRead,
    Record,
    decode_line,
    encode_record,
    kind_of,
    read_records,
    write_record,
)
from komugi.telemetry.liveness import (
    LivenessError,
    LivenessVerdict,
    displacement_band,
    displacement_ratio_within_band,
    ema_within_band,
    fingerprint_series_complete,
    log_freshness,
    self_play_net_advanced,
    served_net_is_the_previous_ema,
)
from komugi.telemetry.records import (
    EMA,
    NET_ROLES,
    SELF_PLAY,
    SHA256_HEX_LENGTH,
    TRAIN,
    IterationMetrics,
    NetFingerprint,
)

__all__ = [
    "EMA",
    "ITERATION_METRICS_KIND",
    "NET_FINGERPRINT_KIND",
    "NET_ROLES",
    "SCHEMA_VERSION",
    "SELF_PLAY",
    "SHA256_HEX_LENGTH",
    "TRAIN",
    "IterationMetrics",
    "JournalDecodeError",
    "JournalRead",
    "LivenessError",
    "LivenessVerdict",
    "NetFingerprint",
    "Record",
    "TruncationToken",
    "archive_before_truncate",
    "archive_name",
    "decode_line",
    "displacement_band",
    "displacement_ratio_within_band",
    "ema_within_band",
    "encode_record",
    "fingerprint_series_complete",
    "guard_truncation",
    "kind_of",
    "log_freshness",
    "read_records",
    "self_play_net_advanced",
    "served_net_is_the_previous_ema",
    "write_record",
]
