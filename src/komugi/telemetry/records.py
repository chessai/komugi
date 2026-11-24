"""Immutable telemetry records: the two things an iteration is allowed to claim.

A campaign's post-mortem is only as good as the numbers it kept, and v1's
numbers were kept as free-form log lines: an iteration could report a draw
rate of ``1.7``, or a self-play net could go unrecorded entirely, and nothing
noticed until the run was over. Both records here validate at construction,
in the same spirit as :mod:`komugi.eval.records` -- a nonsensical metric
raises where it is *built*, not where it is finally plotted.

Two records, deliberately:

* :class:`IterationMetrics` -- the per-iteration scalars a campaign reports.
  Anything not in the fixed schema goes in ``extras``, so adding a probe never
  requires a format change and never silently shadows a real field.
* :class:`NetFingerprint` -- *which* weights an iteration actually used, per
  role. This is the record that makes the v1 frozen-self-play-loop detectable
  at all: without a hash per iteration there is nothing for
  :func:`komugi.telemetry.liveness.self_play_net_advanced` to compare.

Every float is required to be finite. ``NaN`` propagates silently through
means and comparisons -- ``nan <= x`` is ``False`` for every ``x``, so a
single poisoned value turns a liveness gate into a permanent pass -- and it
is not representable in JSON, so it could not round-trip through
:mod:`komugi.telemetry.journal` even if it were wanted.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Final

TRAIN: Final[str] = "train"
"""The net being optimised right now."""

EMA: Final[str] = "ema"
"""The exponential-moving-average (target) net."""

SELF_PLAY: Final[str] = "self_play"
"""The net the self-play workers are actually generating games with."""

NET_ROLES: Final[frozenset[str]] = frozenset({TRAIN, EMA, SELF_PLAY})
"""The roles a :class:`NetFingerprint` may describe.

Closed, unlike ``komugi.eval.records.KNOWN_FAMILIES``: a role is not a label
but a slot in the training loop, and a typo (``'selfplay'``) would make a
liveness gate silently find nothing to check.
"""

SHA256_HEX_LENGTH: Final[int] = 64
"""Length of a hex-encoded SHA-256 digest."""

_HEX_DIGITS: Final[frozenset[str]] = frozenset("0123456789abcdef")


def _require_finite(value: float, field_name: str) -> None:
    """Raise ``ValueError`` if `value` is ``NaN`` or an infinity."""
    if not math.isfinite(value):
        raise ValueError(f"{field_name} must be finite, got {value!r}")


def _require_non_negative(value: float, field_name: str) -> None:
    """Raise ``ValueError`` unless `value` is finite and ``>= 0``."""
    _require_finite(value, field_name)
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative, got {value!r}")


def _no_extras() -> Mapping[str, float]:
    """Return the extras mapping of a record that carried no extra probes."""
    return {}


@dataclass(frozen=True, slots=True)
class IterationMetrics:
    """The scalars one campaign iteration reports about itself.

    ``draw_rate`` is a proportion in ``[0, 1]``; ``avg_game_len`` is in plies
    and ``policy_entropy`` in nats, both non-negative. ``extras`` carries
    anything experiment-specific (``{'value_loss': 0.31}``) and is defensively
    copied, so a caller reusing one mutable dict across iterations cannot
    retroactively rewrite records it already handed over.
    """

    iteration: int
    games: int
    draw_rate: float
    avg_game_len: float
    policy_entropy: float
    extras: Mapping[str, float] = field(default_factory=_no_extras)

    def __post_init__(self) -> None:
        if self.iteration < 0:
            raise ValueError(
                f"IterationMetrics.iteration must be >= 0, got {self.iteration!r}"
            )
        if self.games < 0:
            raise ValueError(
                f"IterationMetrics.games must be >= 0, got {self.games!r}"
            )
        _require_finite(self.draw_rate, "IterationMetrics.draw_rate")
        if not 0.0 <= self.draw_rate <= 1.0:
            raise ValueError(
                f"IterationMetrics.draw_rate must be a proportion in [0, 1], "
                f"got {self.draw_rate!r}"
            )
        _require_non_negative(self.avg_game_len, "IterationMetrics.avg_game_len")
        _require_non_negative(self.policy_entropy, "IterationMetrics.policy_entropy")

        copied: dict[str, float] = {}
        for key, value in self.extras.items():
            if not key:
                raise ValueError("IterationMetrics.extras keys must be non-empty")
            _require_finite(value, f"IterationMetrics.extras[{key!r}]")
            copied[key] = float(value)
        object.__setattr__(self, "extras", copied)


@dataclass(frozen=True, slots=True)
class NetFingerprint:
    """Which weights an iteration used, for one role in the training loop.

    The digest is over the parameters themselves, so two fingerprints being
    equal means the *net* did not change -- which is precisely the signal
    :mod:`komugi.telemetry.liveness` reads. ``param_count`` is carried
    alongside it as a cheap shape check: a hash collision is not a real
    concern, but hashing the wrong buffer (an optimiser state, an empty
    tensor) is, and a mismatched parameter count catches that.
    """

    iteration: int
    role: str
    sha256: str
    param_count: int

    def __post_init__(self) -> None:
        if self.iteration < 0:
            raise ValueError(
                f"NetFingerprint.iteration must be >= 0, got {self.iteration!r}"
            )
        if self.role not in NET_ROLES:
            raise ValueError(
                f"NetFingerprint.role must be one of {sorted(NET_ROLES)}, "
                f"got {self.role!r}"
            )
        if len(self.sha256) != SHA256_HEX_LENGTH or not _HEX_DIGITS.issuperset(
            self.sha256
        ):
            raise ValueError(
                f"NetFingerprint.sha256 must be {SHA256_HEX_LENGTH} lowercase hex "
                f"digits, got {self.sha256!r}"
            )
        if self.param_count <= 0:
            raise ValueError(
                f"NetFingerprint.param_count must be positive, got "
                f"{self.param_count!r}"
            )

    def same_net_as(self, other: NetFingerprint) -> bool:
        """Return whether `other` describes the same weights, ignoring iteration.

        Role is ignored too: the same tensors can legitimately be fingerprinted
        under two roles (a fresh run where ``ema`` still equals ``train``), and
        callers that care about the role compare it themselves.
        """
        return self.sha256 == other.sha256 and self.param_count == other.param_count
