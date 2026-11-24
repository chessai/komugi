"""The frozen vocabulary for the two optional layers the search can run with.

**Proof mode.** ``interval_v1`` turns the tree into a partial proof: after the
search has finished, exact scalar game-value intervals are inferred from the
terminal descendants the search actually reached, propagated over the attached
tree, and the final action and the training target are then restricted to the
admissible root actions -- the ones the intervals have not refuted (KOM-221).
This is the MCTS-Solver / lc0 certainty-propagation idea: a node whose value is
already known exactly does not need more visits, and a root action that is
proven worse than another can be dropped from both the move choice and the
policy target instead of being smoothed over by visit counts. ``off`` leaves
the search a pure Gumbel sequential-halving search, with the survivor as the
decision and the completed-Q over the full legal action set as the target.

**Discovery mode.** ``root_marshal_v1`` runs before sequential halving begins:
the ordinary marshal captures supplied by a tactical provider are verified by
clone/play at the root, and at every initial candidate's reply, and their
terminal results are backed up into the tree before any simulation budget is
spent (KOM-220). This is the decisive / anti-decisive move rule -- a move that
wins immediately should never have to be rediscovered by sampling, and a reply
that loses immediately should never survive a round of halving. Discovery only
means anything when the proof layer is there to consume the terminal results it
backs up, so a discovery mode requires ``proof_mode == "interval_v1"``.

``root_win_v1`` is the same rule with the root half made **exhaustive**: every
offered legal root action is verified by clone/play, so an immediate win of ANY
move type -- a capture, a checkmate by an ordinary move, a checkmate by a drop
or by a stack -- is found, not just the ones a provider happens to enumerate.
The KOM-232 census is why: with ``root_marshal_v1`` live, every available
marshal capture was taken (31/31), and 100% of the wins still missed were
checkmates the provider never named, because it enumerates ordinary captures
only. The reply (anti-decisive) half stays provider-based and unchanged in both
modes -- an exhaustive reply scan costs ``|C| x |replies|`` transitions per
candidate, which fast-mode self-play cannot afford, and captures are the
observed loss mode there. ``root_marshal_v1`` is kept as published provenance
for the iterations that ran it, not as a mode to prefer.

The identity strings are **provenance**: callers -- self-play manifests, arena
players, gauntlet pins -- record them to say which algorithm produced a game or
a training row. A published identity must never change meaning; a change in
what the decision rule or the target computes gets a new ``_v2`` string, never
a redefinition of an existing one. There is no independently configurable
decision or target knob: the identity is derived from the two modes and from
nothing else, so a recorded pair of modes always names exactly one algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

ProofMode = Literal["off", "interval_v1"]
DiscoveryMode = Literal["off", "root_marshal_v1", "root_win_v1"]

PROOF_MODES: Final[tuple[str, ...]] = ("off", "interval_v1")
DISCOVERY_MODES: Final[tuple[str, ...]] = ("off", "root_marshal_v1", "root_win_v1")


@dataclass(frozen=True, slots=True)
class SearchAlgorithmIdentity:
    """The two modes a search ran with, and the algorithm they name."""

    proof_mode: str
    discovery_mode: str
    decision_id: str
    target_id: str


def validate_search_modes(proof_mode: str, discovery_mode: str) -> None:
    """Raise :class:`ValueError` unless the mode pair is one we can run."""
    if not isinstance(proof_mode, str):
        raise ValueError(f"proof_mode must be a str, got {proof_mode!r}")
    if not isinstance(discovery_mode, str):
        raise ValueError(f"discovery_mode must be a str, got {discovery_mode!r}")
    if proof_mode not in PROOF_MODES:
        raise ValueError(
            f"unknown proof_mode {proof_mode!r}; expected one of {PROOF_MODES!r}"
        )
    if discovery_mode not in DISCOVERY_MODES:
        raise ValueError(
            f"unknown discovery_mode {discovery_mode!r}; "
            f"expected one of {DISCOVERY_MODES!r}"
        )
    if discovery_mode != "off" and proof_mode == "off":
        raise ValueError(
            f"discovery_mode {discovery_mode!r} requires the proof layer; "
            f"got proof_mode {proof_mode!r}"
        )


def search_algorithm_identity(
    proof_mode: str, discovery_mode: str
) -> SearchAlgorithmIdentity:
    """Derive the provenance identity of the search these two modes select."""
    validate_search_modes(proof_mode, discovery_mode)
    if proof_mode == "off":
        decision_id = "gumbel_sh_survivor_v1"
        target_id = "completed_q_full_legal_v1"
    else:
        decision_id = "gumbel_sh_interval_v1"
        target_id = "completed_q_proof_support_v1"
    return SearchAlgorithmIdentity(
        proof_mode=proof_mode,
        discovery_mode=discovery_mode,
        decision_id=decision_id,
        target_id=target_id,
    )


LEGACY_IDENTITY: Final[SearchAlgorithmIdentity] = search_algorithm_identity("off", "off")
