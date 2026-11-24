"""The gauntlet: 64 fast games against a frozen anchor, sharded, with a floor.

KOM-022 §4. A regression detector, not a strength measurement -- at 64 games
the floor resolves roughly a 70-Elo deficit and is blind to anything smaller.
The strength measurement is the ladder over the whole checkpoint sequence.

The anchor is pinned by **path and weight digest**, in an append-only pinfile,
and physically copied **under its own published checkpoint filename**::

    <root>/anchors/anchor.jsonl                                  the pinfile
    <root>/anchors/gen_<g>/checkpoint-<step:09d>-<digest16>.pt    the copy

Keeping the published basename is not cosmetic.
:func:`~komugi.train.checkpoint.load_checkpoint_file` does not open the path it
is handed: it *lists* the parent directory and looks for the path among the
listed checkpoints, and the listing admits only
``checkpoint-(\\d{9,})-([0-9a-f]{16})\\.pt``. A copy named ``gen_0.pt`` is
therefore never listed, the loader returns ``None``, and both readings of that
``None`` are bad -- treat it as "verification failed" and every gauntlet
aborts, treat it as benign and the anchor is never verified at all. The
published name also brings the module's own digest-in-the-name integrity check
along for free, which is most of what a hand-rolled digest comparison was
buying.

**The pin is ordered so a kill cannot leave a pinned-but-unreadable anchor**:
mkdir, copy to a dot-prefixed temp, fsync, ``os.replace`` to the published
name, fsync the directory, re-read *through the loader the players use* and
check the digests -- and **only then** append the pinfile line. The pinfile
line is the commit point and it is appended last, so a kill mid-copy leaves an
unpinned directory rather than a pinned generation whose gauntlet is wedged
permanently behind an append-only file.

**Both digest halves are checked before every gauntlet**, deliberately: for an
*anchor* the question is file identity, not learning, so §3a's narrowing of the
liveness digest to parameters must not narrow this. An opponent whose
normalisation statistics were swapped underneath is a different opponent, which
is why the pinfile carries ``bn_sha256`` beside ``weight_sha256``.

**The parent process never plays.** Games run in shard subprocesses, each given
its device and its ``NET_ENV`` explicitly, because
:func:`~komugi.eval.arena.pin_determinism` sets *process-global* torch state:
calling it in the campaign process would hand the trainer deterministic-kernel
selection from iteration 3 onward -- a silent performance and reproducibility
change mid-run, and one nothing afterwards could attribute to the gauntlet.
"""

from __future__ import annotations

import dataclasses
import json
import multiprocessing
import os
import shutil
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import Any, Final

import numpy as np

from komugi.campaign import factory, pinned
from komugi.campaign.factory import NET_ENV
from komugi.eval.arena import (
    ArenaConfig,
    ArenaPlayer,
    ArenaSummary,
    MatchOutcome,
    play_arena_game,
)
from komugi.eval.records import GameRecord, Player
from komugi.search.modes import (
    SearchAlgorithmIdentity,
    search_algorithm_identity,
    validate_search_modes,
)
from komugi.selfplay.inference import (
    BatchEvaluator,
    InferenceConfig,
    InProcessEvaluator,
)
from komugi.train.checkpoint import load_checkpoint_file, require_matching_plane_count
from komugi.train.digest import bn_digest, canonical_digest

ANCHORS_DIRNAME: Final[str] = "anchors"
PINFILE_NAME: Final[str] = "anchor.jsonl"
GENERATION_DIR: Final[str] = "gen_{generation}"

GAUNTLET_DOMAIN: Final[int] = 0x9022_0003
"""Spawn-key domain for the gauntlet's seeds.

Reusing ``iteration_seed`` would correlate the gauntlet's Gumbel streams with
self-play's; :class:`~komugi.eval.arena.ArenaConfig`'s own docstring warns
about exactly this class of collision.
"""

GAUNTLET_FAMILY: Final[str] = "komugi"
_PINFILE_MODE: Final[int] = 0o644

GAUNTLET_PROFILE_LEGACY: Final[str] = "legacy"
"""Both sides play the legacy search (``off``/``off``), whatever self-play runs.

The default, and what campaign 6 phase b keeps. The internal gauntlet is a
**net-vs-net** comparison whose entire history was played by the legacy search:
switching it to the self-play profile mid-campaign would change what the floor
is a measurement of, without changing the number it is compared against.
"""

GAUNTLET_PROFILE_SELFPLAY: Final[str] = "selfplay"
"""Both sides play the campaign's self-play profile -- for a FUTURE campaign.

Offered so the alternative is a recorded switch rather than an edit, and so the
pin can say which of the two produced it.
"""

PROFILE_SOURCE_RECORDED: Final[str] = "recorded"
"""The pin line carried its modes: the pair is observed, not assumed."""

PROFILE_SOURCE_LEGACY_INFERRED: Final[str] = "legacy_inferred"
"""The pin line predates the mode keys, so ``off``/``off`` is an INFERENCE.

Every line written before KOM-226 was produced by a build that could only play
the legacy search, so the inference is sound -- but a census must be able to
tell "observed off" from "assumed off", which is the whole point of recording
the source beside the modes.
"""

PROFILE_SOURCES: Final[tuple[str, ...]] = (
    PROFILE_SOURCE_RECORDED,
    PROFILE_SOURCE_LEGACY_INFERRED,
)


def gauntlet_modes(profile: str, selfplay_modes: tuple[str, str]) -> tuple[str, str]:
    """Return the ``(proof_mode, discovery_mode)`` pair `profile` selects.

    Args:
        profile: :data:`GAUNTLET_PROFILE_LEGACY` or
            :data:`GAUNTLET_PROFILE_SELFPLAY`.
        selfplay_modes: The campaign's self-play pair, used only by the
            ``selfplay`` profile.

    Raises:
        ValueError: If `profile` is neither of the two, or if the self-play pair
            the ``selfplay`` profile selects is not a pair the search can run.
            An unknown profile is refused rather than falling back to legacy: a
            typo'd profile that silently played the legacy search would be a
            gauntlet whose recorded provenance is not what ran.
    """
    if profile == GAUNTLET_PROFILE_LEGACY:
        return ("off", "off")
    if profile == GAUNTLET_PROFILE_SELFPLAY:
        proof_mode, discovery_mode = selfplay_modes
        validate_search_modes(proof_mode, discovery_mode)
        return (proof_mode, discovery_mode)
    raise ValueError(
        f"unknown gauntlet profile {profile!r}; expected one of "
        f"{(GAUNTLET_PROFILE_LEGACY, GAUNTLET_PROFILE_SELFPLAY)!r}"
    )


class GauntletAborted(RuntimeError):
    """The anchor could not be verified, so no games were played.

    An anchor that can be swapped underneath is not an anchor: a mismatch that
    the recovery path cannot resolve aborts loudly rather than quietly rating
    against something else.
    """


class GauntletLost(RuntimeError):
    """A shard process died without returning -- a signal, an ``os._exit``, the OOM killer.

    Distinct from a shard that *raised*: a raised exception is re-raised in the
    parent with the child's traceback, and is a bug in the shard's own work. A
    lost child carries no exception at all, only a missing result, and the two
    want different first questions of an operator (``dmesg`` versus the
    traceback).

    Raised for a pool that was already broken at **submit** time as well as for
    one that breaks under a collected future: the two differ only in which
    child died first, and a caller promised this module's typed errors should
    not have to catch :class:`~concurrent.futures.process.BrokenProcessPool`
    for the earlier of them (KOM-180 impl review F7).

    Unlike :class:`GauntletAborted` this is **not** folded into a liveness
    verdict: it is raised out of :func:`run_gauntlet` so the campaign's boundary
    crash handler writes the ordinary crash ``PAUSED`` for it. A lost shard is
    an infrastructure failure, not a measurement, and scoring a gauntlet that is
    missing a shard's games would be a number that never happened.
    """


class GauntletStalled(RuntimeError):
    """The gauntlet ran past :data:`~komugi.campaign.pinned.GAUNTLET_SHARD_DEADLINE_S`.

    That deadline is one absolute bound on the whole fan-out rather than a
    per-shard allowance (the shards run concurrently, so their walls overlap),
    and this names the first shard found still running when it was reached.

    Alive, and not returning. Before the deadline existed this hung the campaign
    forever (``pool.map`` has no timeout), so the failure's real signature was an
    operator killing the run days later -- landing the kill between the
    published checkpoint and the boundary record, which is the window
    :data:`~komugi.campaign.pinned.UNFINISHED_BOUNDARY` refuses to advance across
    (review CP-04 and CP-01, which is why they land together).

    Raised, never retried: ``campaign_records`` has no per-iteration dedupe and
    the re-pin rule reads ``scores[-2:]``, so a second gauntlet inside one
    boundary would write a second score for one iteration and shift what the
    re-pin is a function of.
    """


@dataclass(frozen=True, slots=True)
class AnchorPin:
    """One line of ``<root>/anchors/anchor.jsonl``: a frozen opponent.

    The pin records **which search the gauntlet against this anchor plays**
    (KOM-226): a score is only comparable across generations if the algorithm
    that produced it is the same one, and the modes are the only thing that says
    so. ``profile_source`` distinguishes an observed pair from one inferred for
    a line written before the keys existed -- see
    :data:`PROFILE_SOURCE_LEGACY_INFERRED`.
    """

    generation: int
    pinned_at_iteration: int
    path: str
    name_digest: str
    weight_sha256: str
    bn_sha256: str
    m: int
    n: int
    games: int
    floor: float
    draw_score: float
    proof_mode: str = "off"
    discovery_mode: str = "off"
    profile_source: str = PROFILE_SOURCE_RECORDED

    def __post_init__(self) -> None:
        """Refuse a pin whose provenance claim is not one the search can run."""
        validate_search_modes(self.proof_mode, self.discovery_mode)
        if self.profile_source not in PROFILE_SOURCES:
            raise ValueError(
                f"unknown profile_source {self.profile_source!r}; expected one "
                f"of {PROFILE_SOURCES!r}"
            )

    @property
    def search_identity(self) -> SearchAlgorithmIdentity:
        """Return the provenance identity of the search this pin's modes select.

        Derived, never stored: the decision rule and the target are a function
        of the two modes and of nothing else, so a pin that records the pair
        already names exactly one algorithm.
        """
        return search_algorithm_identity(self.proof_mode, self.discovery_mode)

    def as_json(self) -> dict[str, Any]:
        """Return the JSON object this pin is stored as.

        The modes and their source are written on every new line. Old lines are
        never rewritten -- the pinfile is append-only, and a rewritten history
        would turn an inference into a claim of observation.
        """
        return {
            "gen": self.generation,
            "pinned_at_iteration": self.pinned_at_iteration,
            "path": self.path,
            "name_digest": self.name_digest,
            "weight_sha256": self.weight_sha256,
            "bn_sha256": self.bn_sha256,
            "m": self.m,
            "n": self.n,
            "games": self.games,
            "floor": self.floor,
            "draw_score": self.draw_score,
            "proof_mode": self.proof_mode,
            "discovery_mode": self.discovery_mode,
            "profile_source": self.profile_source,
        }


@dataclass(frozen=True, slots=True)
class GauntletResult:
    """What one gauntlet came to, including the ways it can come to nothing."""

    generation: int
    score: float
    summary: ArenaSummary
    wall_s: float
    uninformative: bool

    @property
    def informative(self) -> bool:
        """Whether the floor may be evaluated against this result."""
        return not self.uninformative


def anchors_dir(root: str | os.PathLike[str]) -> Path:
    """Return ``<root>/anchors``, whether or not it exists."""
    return Path(root) / ANCHORS_DIRNAME


def pinfile_path(root: str | os.PathLike[str]) -> Path:
    """Return the append-only anchor pinfile."""
    return anchors_dir(root) / PINFILE_NAME


def generation_dir(root: str | os.PathLike[str], generation: int) -> Path:
    """Return ``<root>/anchors/gen_<g>``, whether or not it exists."""
    return anchors_dir(root) / GENERATION_DIR.format(generation=generation)


def gauntlet_seed(run_seed: int, iteration: int) -> int:
    """Return this iteration's gauntlet seed, in its own domain."""
    child = np.random.SeedSequence(run_seed, spawn_key=(GAUNTLET_DOMAIN, iteration))
    return int(child.generate_state(1, dtype=np.uint32)[0])


def _fsync_directory(directory: Path) -> None:
    """``fsync`` a directory so a rename into it is durable, not just atomic."""
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _append_pin(root: str | os.PathLike[str], pin: AnchorPin) -> None:
    """Append `pin` to the pinfile with ``O_APPEND`` + ``fsync``.

    The commit point of the whole pin operation, and the last thing it does.
    """
    path = pinfile_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(pin.as_json(), sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, _PINFILE_MODE)
    try:
        written = 0
        while written < len(payload):
            written += os.write(fd, payload[written:])
        os.fsync(fd)
    finally:
        os.close(fd)


def _pin_search_profile(parsed: Mapping[str, Any]) -> tuple[str, str, str]:
    """Return ``(proof_mode, discovery_mode, profile_source)`` for one pin line.

    A line with NEITHER mode key predates KOM-226 and is read as the legacy
    search it can only have been played by, marked
    :data:`PROFILE_SOURCE_LEGACY_INFERRED`. A line carrying BOTH keys is a
    provenance *claim* and is validated as one: the pair must be runnable, and
    the source is :data:`PROFILE_SOURCE_RECORDED`.

    The source is derived from the keys' presence rather than read from the
    line, so it cannot be forged by a hand-edited pinfile: "this line carried
    its modes" is a fact about the line, not a field it may assert. A
    ``profile_source`` the line *does* carry is therefore not read -- it is
    checked, and must agree with what the keys say.

    Raises:
        ValueError: If the line carries exactly one of the two mode keys (half a
            pair is neither a legacy line nor a claim, and defaulting the missing
            half would invent the very provenance this function exists to
            report); if it asserts a ``profile_source`` other than
            :data:`PROFILE_SOURCE_RECORDED` alongside its modes; or if the
            recorded pair is not one the search can run. Unlike a partial line,
            none of these is the shape a kill leaves -- they are a pin claiming
            something untrue, and skipping one would drop an anchor generation
            out of an append-only history silently.
    """
    has_proof = "proof_mode" in parsed
    has_discovery = "discovery_mode" in parsed
    if not has_proof and not has_discovery:
        return ("off", "off", PROFILE_SOURCE_LEGACY_INFERRED)
    if has_proof != has_discovery:
        present = "proof_mode" if has_proof else "discovery_mode"
        missing = "discovery_mode" if has_proof else "proof_mode"
        raise ValueError(
            f"pin line for generation {parsed.get('gen')!r} carries {present} "
            f"but not {missing}; a search profile is the PAIR, and reading the "
            f"missing half as 'off' would record a search this line does not "
            f"claim. Keys present: {sorted(parsed)!r}"
        )
    proof_mode = str(parsed["proof_mode"])
    discovery_mode = str(parsed["discovery_mode"])
    validate_search_modes(proof_mode, discovery_mode)
    if "profile_source" in parsed:
        asserted = str(parsed["profile_source"])
        if asserted != PROFILE_SOURCE_RECORDED:
            raise ValueError(
                f"pin line for generation {parsed.get('gen')!r} carries its "
                f"modes ({proof_mode!r}, {discovery_mode!r}) but asserts "
                f"profile_source {asserted!r}; a line that carries its modes "
                f"recorded them, so the only source it may claim is "
                f"{PROFILE_SOURCE_RECORDED!r}"
            )
    return (proof_mode, discovery_mode, PROFILE_SOURCE_RECORDED)


def read_pins(root: str | os.PathLike[str]) -> tuple[AnchorPin, ...]:
    """Return every anchor generation this root has pinned, oldest first.

    A trailing partial line -- the shape a kill during the append leaves -- is
    skipped rather than raising: it names a generation whose pin never
    committed, which is exactly the state the ordering above is designed to
    make harmless.

    Raises:
        ValueError: If a *complete* line records a mode pair the search cannot
            run (see :func:`_pin_search_profile`).
    """
    path = pinfile_path(root)
    if not path.is_file():
        return ()
    pins: list[AnchorPin] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            parsed: object = json.loads(line)
        except ValueError:
            continue
        if not isinstance(parsed, dict):
            continue
        try:
            required: dict[str, Any] = {
                "generation": int(parsed["gen"]),
                "pinned_at_iteration": int(parsed["pinned_at_iteration"]),
                "path": str(parsed["path"]),
                "name_digest": str(parsed["name_digest"]),
                "weight_sha256": str(parsed["weight_sha256"]),
                "bn_sha256": str(parsed["bn_sha256"]),
                "m": int(parsed["m"]),
                "n": int(parsed["n"]),
                "games": int(parsed["games"]),
                "floor": float(parsed["floor"]),
                "draw_score": float(parsed["draw_score"]),
            }
        except (KeyError, TypeError, ValueError):
            continue
        # OUTSIDE the skip: a missing required key is an incomplete line, but a
        # mode pair that does not exist is a claim, and it must not be swallowed
        # by the partial-line tolerance above.
        proof_mode, discovery_mode, profile_source = _pin_search_profile(parsed)
        pins.append(
            AnchorPin(
                **required,
                proof_mode=proof_mode,
                discovery_mode=discovery_mode,
                profile_source=profile_source,
            )
        )
    return tuple(pins)


def _copy_durably(source: Path, target: Path) -> None:
    """Copy `source` to `target` atomically: temp, fsync, replace, fsync dir."""
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.parent / f".{target.name}.tmp-{os.getpid()}"
    shutil.copyfile(source, tmp)
    fd = os.open(tmp, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, target)
    _fsync_directory(target.parent)


def pin_anchor(
    root: str | os.PathLike[str],
    generation: int,
    iteration: int,
    source: str | os.PathLike[str],
    param_keys: Sequence[str],
    *,
    m: int,
    n: int,
    games: int,
    floor: float,
    draw_score: float,
    proof_mode: str = "off",
    discovery_mode: str = "off",
) -> AnchorPin:
    """Freeze `source` as anchor generation `generation` and return its pin.

    The copy keeps the source's published basename, so it is loadable through
    :func:`~komugi.train.checkpoint.load_checkpoint_file`; the generation lives
    in the directory name and in the pinfile line, never in the filename.

    Raises:
        GauntletAborted: If the source does not verify, or if the copy does not
            read back with the digests the source had. Nothing is appended to
            the pinfile in either case, so the generation is simply not pinned.
    """
    origin = Path(source)
    original = load_checkpoint_file(origin)
    if original is None:
        raise GauntletAborted(
            f"{origin} does not verify, so it cannot be frozen as an anchor: "
            f"a gauntlet against an unreadable opponent is not a gauntlet"
        )
    target = generation_dir(root, generation) / origin.name
    _copy_durably(origin, target)

    copied = load_checkpoint_file(target)
    if copied is None:
        raise GauntletAborted(
            f"the anchor copy at {target} does not read back through the "
            f"loader the players use; refusing to pin it"
        )
    weights = canonical_digest(copied.model_state, param_keys)
    buffers = bn_digest(copied.model_state, param_keys)
    expected = canonical_digest(original.model_state, param_keys)
    if weights.sha256 != expected.sha256:
        raise GauntletAborted(
            f"the anchor copy at {target} hashes to {weights.sha256[:16]} but "
            f"its source {origin} hashes to {expected.sha256[:16]}"
        )
    pin = AnchorPin(
        generation=generation,
        pinned_at_iteration=iteration,
        # The *source*, not the copy: the copy's location is a function of the
        # generation, and recording the source is what makes the recovery in
        # `verify_anchor` possible rather than a manual re-pin of an
        # append-only file.
        path=str(origin),
        name_digest=target.name.rsplit("-", 1)[-1].removesuffix(".pt"),
        weight_sha256=weights.sha256,
        bn_sha256=buffers.sha256,
        m=m,
        n=n,
        games=games,
        floor=floor,
        draw_score=draw_score,
        # The modes the gauntlet against this anchor will actually be played
        # with -- stamped here, where the pin is written, so the line records an
        # observation rather than this build's default.
        proof_mode=proof_mode,
        discovery_mode=discovery_mode,
        profile_source=PROFILE_SOURCE_RECORDED,
    )
    _append_pin(root, pin)
    return pin


def anchor_copy_path(root: str | os.PathLike[str], pin: AnchorPin) -> Path:
    """Return where `pin`'s frozen copy lives, under its published basename."""
    return generation_dir(root, pin.generation) / Path(pin.path).name


def _digests_match(
    path: Path, pin: AnchorPin, param_keys: Sequence[str]
) -> tuple[bool, str, str]:
    """Return whether `path` still verifies to `pin`'s two digests."""
    checkpoint = load_checkpoint_file(path)
    if checkpoint is None:
        return (False, "", "")
    weights = canonical_digest(checkpoint.model_state, param_keys).sha256
    buffers = bn_digest(checkpoint.model_state, param_keys).sha256
    return (weights == pin.weight_sha256 and buffers == pin.bn_sha256, weights, buffers)


def verify_anchor(
    root: str | os.PathLike[str], pin: AnchorPin, param_keys: Sequence[str]
) -> Path:
    """Re-check a pinned anchor's two digests and return the path to play.

    **Both halves, deliberately.** For an anchor the question is *file
    identity*, not learning, so §3a's narrowing of the liveness digest to
    parameters must not narrow this: a mutation confined to ``running_mean``
    changes the opponent without touching a single parameter, and the parameter
    digest alone would wave it through.

    Recovery, defined rather than left to surgery: if the copy is missing or
    fails its digests, and the pin's recorded source ``path`` still holds a file
    that verifies to the pinned digests, the copy is re-made from it and
    re-checked. Only if that also fails does the gauntlet abort -- which is the
    difference between a transient bad copy and an anchor that was swapped.

    Raises:
        GauntletAborted: Naming the pinned and recomputed digests.
    """
    copy = anchor_copy_path(root, pin)
    matched, weights, buffers = _digests_match(copy, pin, param_keys)
    if matched:
        return copy

    source = Path(pin.path)
    if source.exists() and _digests_match(source, pin, param_keys)[0]:
        _copy_durably(source, copy)
        recovered, weights, buffers = _digests_match(copy, pin, param_keys)
        if recovered:
            return copy

    raise GauntletAborted(
        f"pinned anchor gen {pin.generation} does not verify: {copy} hashes to "
        f"parameters {weights[:16] or '<unreadable>'} / BatchNorm "
        f"{buffers[:16] or '<unreadable>'}, the pinfile says "
        f"{pin.weight_sha256[:16]} / {pin.bn_sha256[:16]}, and the recorded "
        f"source {source} could not supply a replacement. An anchor that can be "
        f"swapped underneath is not an anchor; no games were played"
    )


def checkpoint_player(
    path: str | os.PathLike[str],
    player_id: str,
    *,
    device: str = "cpu",
    m: int,
    n: int,
    proof_mode: str = "off",
    discovery_mode: str = "off",
) -> ArenaPlayer:
    """Return an entrant that serves exactly `path`, at the fast budget.

    The modes are **per entrant** and default to the legacy search, so every
    existing call site keeps the player it already had: the gauntlet's two sides
    are given their own pair by :func:`run_shard`, because "what does the proof
    layer buy" is a match between entrants that differ in exactly that.

    The public counterpart of the arena's private ``_checkpoint_evaluator``,
    with one difference that matters here: the factory **memoises**. A shard
    plays several games, and the arena's per-game door builds an evaluator per
    call, so an unmemoised factory would reload and rebuild the net once per
    game -- seconds of torch work to produce identical weights.

    ``eval()`` is not optional: a net left in training mode updates its
    BatchNorm running statistics from whatever wave the search submitted, which
    would make a player's strength depend on its opponent.
    """
    built: list[BatchEvaluator] = []

    def build() -> BatchEvaluator:
        """Load the checkpoint once per process and serve it thereafter."""
        if built:
            return built[0]
        checkpoint = load_checkpoint_file(path)
        if checkpoint is None:
            raise FileNotFoundError(
                f"{path} names no checkpoint that verifies; refusing to rate "
                f"freshly initialised weights under a checkpoint's name"
            )
        net = factory.new_net()
        # On the returned object, before load_state_dict (KOM-002 §5.3, review
        # F3): an entrant from a previous plane map dies here by name instead
        # of as a raw ``size mismatch for stem.weight`` traceback from inside a
        # gauntlet shard -- which is the failure :func:`run_gauntlet`'s own
        # docstring already anticipates, several hundred lines below.
        require_matching_plane_count(checkpoint, net.config.trunk.in_planes)
        net.load_state_dict(checkpoint.model_state)
        net.to(device)
        net.eval()
        evaluator = InProcessEvaluator(
            net,
            InferenceConfig(
                in_planes=factory.IN_PLANES, num_actions=factory.NUM_ACTIONS
            ),
        )
        built.append(evaluator)
        return evaluator

    return ArenaPlayer(
        player=Player(id=player_id, family=GAUNTLET_FAMILY),
        evaluator_factory=build,
        m=m,
        n=n,
        proof_mode=proof_mode,
        discovery_mode=discovery_mode,
    )


def summarise(records: Sequence[GameRecord], plies: Sequence[int]) -> ArenaSummary:
    """Return the summary of a set of games -- one pass over the union.

    Deliberately **not** a fold of per-shard summaries.
    ``ArenaSummary.shortest_plies`` uses ``min(plies, default=0)``, so an empty
    shard contributes a ``0`` that a careless fold would adopt as the global
    minimum, and ``decisive`` -- the validity condition the whole floor rests
    on -- would inherit the same class of error.
    """
    draws = sum(1 for record in records if record.is_draw)
    return ArenaSummary(
        games=len(records),
        decisive=len(records) - draws,
        draws=draws,
        total_plies=sum(plies),
        shortest_plies=min(plies, default=0),
        longest_plies=max(plies, default=0),
    )


def play_pairs(
    a: ArenaPlayer,
    b: ArenaPlayer,
    config: ArenaConfig,
    pairs: range,
) -> tuple[tuple[GameRecord, ...], tuple[int, ...]]:
    """Play a contiguous slice of a match's pairs, **keeping the global index**.

    :func:`~komugi.eval.arena.play_match` hard-codes ``for pair in
    range(config.pairs)`` from zero, so it cannot express "play pairs 8..15 of
    this match". This does, and the sharding is *exact* rather than
    approximate: every game's randomness is ``game_seeds(config, pair, seat)``,
    a pure function of the match seed and the global coordinates, so the union
    of the shards' records is identical to a serial run -- which
    :func:`merge_outcomes` then summarises in one pass.

    Returns the records in play order and their ply counts, because
    :class:`~komugi.eval.records.GameRecord` deliberately carries no length and
    the summary needs one.
    """
    if a.player.id == b.player.id:
        raise ValueError(
            f"both entrants are {a.player.id!r}: a game record cannot name one "
            f"player in both seats"
        )
    if pairs.start < 0 or pairs.stop > config.pairs or pairs.step != 1:
        raise ValueError(
            f"pairs must be a contiguous sub-range of range(0, {config.pairs}), "
            f"got {pairs!r}"
        )
    records: list[GameRecord] = []
    plies: list[int] = []
    for pair in pairs:
        pair_id = f"{a.player.id}:{b.player.id}:{pair:05d}"
        for seat in (0, 1):
            white, black = (a, b) if seat == 0 else (b, a)
            outcome = play_arena_game(white, black, config, pair=pair, seat=seat)
            records.append(
                GameRecord(
                    white=white.player.id,
                    black=black.player.id,
                    result=outcome.result,
                    book_id=config.book_id,
                    pair_id=pair_id,
                )
            )
            plies.append(len(outcome.moves))
    return tuple(records), tuple(plies)


def merge_outcomes(
    shards: Sequence[tuple[Sequence[GameRecord], Sequence[int]]],
) -> MatchOutcome:
    """Concatenate the shards' records and summarise the union exactly once."""
    records: list[GameRecord] = []
    plies: list[int] = []
    for shard_records, shard_plies in shards:
        records.extend(shard_records)
        plies.extend(shard_plies)
    return MatchOutcome(tuple(records), summarise(records, plies))


def score_for(records: Sequence[GameRecord], player_id: str) -> float:
    """Return ``(wins + draws/2) / games`` from `player_id`'s side.

    ``0.0`` for an empty set rather than a division error: an aborted gauntlet
    has a caller that already knows not to read this.
    """
    if not records:
        return 0.0
    total = 0.0
    for record in records:
        # GameRecord.result is stated from white's perspective.
        if record.white == player_id:
            total += record.result
        else:
            total += 1.0 - record.result
    return total / len(records)


@dataclass(frozen=True, slots=True)
class ShardRequest:
    """One shard process's whole instruction. Picklable by construction.

    **Never rebind this name on the module** -- not even to a wrapper that
    returns real instances. ``pickle`` serialises an instance by looking its
    class up again as ``komugi.campaign.gauntlet.ShardRequest``, so a rebound
    attribute makes every request fail in ``multiprocessing.reduction`` with
    "it's not the same object as". :func:`_shard_requests` exists to be the
    observable seam instead.
    """

    challenger_path: str
    challenger_id: str
    anchor_path: str
    anchor_id: str
    pairs_start: int
    pairs_stop: int
    games: int
    seed: int
    draw_score: float
    m: int
    n: int
    device: str
    net_env: str
    # Appended LAST, and per side: the shard is where the two entrants are
    # built, so it is the only place that can give each of them its own search.
    challenger_proof_mode: str = "off"
    challenger_discovery_mode: str = "off"
    anchor_proof_mode: str = "off"
    anchor_discovery_mode: str = "off"

    def __post_init__(self) -> None:
        """Refuse a request no shard could play, in the PARENT process.

        Both pairs, because either side alone can name a search that does not
        exist, and a shard that discovers that after ``spawn`` reports it as a
        dead child rather than as a bad request.
        """
        validate_search_modes(
            self.challenger_proof_mode, self.challenger_discovery_mode
        )
        validate_search_modes(self.anchor_proof_mode, self.anchor_discovery_mode)

    def __getstate__(self) -> tuple[object, ...]:
        """Return every field value, in field order, as a plain tuple.

        This object is pickled into each spawned shard, and a shard started by a
        parent running OLD code can be handed the payload that code built. A
        positional tuple plus the length-dispatching :meth:`__setstate__` below
        is what makes that crossing legible (the KOM-223 pattern).
        """
        return _request_getstate(self)

    def __setstate__(self, state: object) -> None:
        """Restore from a current (17-field) or legacy (13-field) payload.

        A 13-field payload predates the modes, so it is restored as
        ``off``/``off`` on both sides -- exactly the search it was produced by.
        Any other length is refused rather than padded: it is a payload this
        build cannot interpret, and a silently-defaulted mode would be a shard
        playing a search its request does not name.

        Raises:
            ValueError: If *state* is not a list or tuple, is of any length
                other than 13 or 17, or carries a mode pair the search cannot
                run.
        """
        _request_setstate(self, state)


_LEGACY_REQUEST_FIELDS: Final[int] = 13
"""Field count of :class:`ShardRequest` before the per-side modes (KOM-226).

A frozen literal, not ``len(fields) - 4``: the legacy payload's shape is history
and cannot change, so a future appended field must appear here as a NEW accepted
length rather than shift what "legacy" means.
"""

_CURRENT_REQUEST_FIELDS: Final[int] = 17


def _request_getstate(request: ShardRequest) -> tuple[object, ...]:
    """Return every field of *request*, in field order, as a plain tuple."""
    return tuple(getattr(request, f.name) for f in dataclasses.fields(request))


def _request_setstate(request: ShardRequest, state: object) -> None:
    """Restore *request* from a current or legacy positional payload."""
    # The default dataclass hook on the PRE-KOM-226 class sends a 13-element
    # LIST (dataclasses._dataclass_getstate), so a list is the legacy shape too.
    if isinstance(state, list):
        state = tuple(state)
    if not isinstance(state, tuple):
        raise ValueError(
            f"ShardRequest.__setstate__ expects a tuple of field values, got "
            f"{type(state).__name__}"
        )
    if len(state) == _LEGACY_REQUEST_FIELDS:
        values: tuple[object, ...] = (*state, "off", "off", "off", "off")
    elif len(state) == _CURRENT_REQUEST_FIELDS:
        values = state
    else:
        raise ValueError(
            f"ShardRequest.__setstate__ expects {_CURRENT_REQUEST_FIELDS} field "
            f"values (or the legacy {_LEGACY_REQUEST_FIELDS}), got {len(state)}"
        )
    names = tuple(f.name for f in dataclasses.fields(request))
    # object.__setattr__ because the class is frozen (and slotted, so there is
    # no __dict__ to update instead); strict=True because a field count that has
    # drifted from the two constants above is a pickle contract that no longer
    # holds, and it must fail here rather than restore a half-populated request.
    for name, value in zip(names, values, strict=True):
        object.__setattr__(request, name, value)
    # Exactly the validation a constructed request gets -- arriving over a pipe
    # is not an exemption.
    request.__post_init__()


# Re-attached from the module-level implementations for the reason
# `komugi.config` re-attaches its own: `@dataclass(slots=True)` RECREATES the
# class object, and a frozen slotted dataclass installs pickle hooks of its own,
# guarded against clobbering a user's on current CPython and unguarded on
# earlier ones. A no-op whenever the class-body definitions survived.
setattr(ShardRequest, "__getstate__", _request_getstate)  # noqa: B010
setattr(ShardRequest, "__setstate__", _request_setstate)  # noqa: B010


def run_shard(request: ShardRequest) -> tuple[tuple[GameRecord, ...], tuple[int, ...]]:
    """Play one shard's pairs. **Module-level, so ``spawn`` can reach it.**

    Sets :data:`~komugi.campaign.factory.NET_ENV` from the request rather than
    trusting inheritance: a shard spawned without it against a small-net
    campaign fails inside ``load_state_dict`` with a shape error that reads
    like a corrupt anchor. The device is likewise explicit, never a default.
    """
    if request.net_env:
        os.environ[NET_ENV] = request.net_env
    else:
        os.environ.pop(NET_ENV, None)
    config = ArenaConfig(
        games=request.games, seed=request.seed, draw_score=request.draw_score
    )
    # Each side with ITS OWN modes: a gauntlet whose two entrants were built
    # from one pair could never express the comparison the profile switch
    # exists for, and would silently rate a challenger under the anchor's search.
    challenger = checkpoint_player(
        request.challenger_path,
        request.challenger_id,
        device=request.device,
        m=request.m,
        n=request.n,
        proof_mode=request.challenger_proof_mode,
        discovery_mode=request.challenger_discovery_mode,
    )
    anchor = checkpoint_player(
        request.anchor_path,
        request.anchor_id,
        device=request.device,
        m=request.m,
        n=request.n,
        proof_mode=request.anchor_proof_mode,
        discovery_mode=request.anchor_discovery_mode,
    )
    return play_pairs(
        challenger, anchor, config, range(request.pairs_start, request.pairs_stop)
    )


def shard_ranges(pairs: int, shards: int) -> tuple[range, ...]:
    """Split ``range(0, pairs)`` into `shards` contiguous, disjoint slices.

    Contiguous and index-preserving, so each shard plays the pairs a serial run
    would have played under the same indices. A shard count above the pair
    count yields empty ranges rather than an error -- an empty shard is a
    legitimate (if wasteful) configuration, and the merge above is written to
    survive one.
    """
    if shards < 1:
        raise ValueError(f"shards must be >= 1, got {shards!r}")
    bounds = [(pairs * index) // shards for index in range(shards + 1)]
    return tuple(range(bounds[i], bounds[i + 1]) for i in range(shards))


def _shard_requests(
    challenger_path: str | os.PathLike[str],
    challenger_id: str,
    anchor_path: str | os.PathLike[str],
    anchor_id: str,
    pin: AnchorPin,
    ranges: Sequence[range],
    *,
    seed: int,
    device: str,
    challenger_modes: tuple[str, str] = ("off", "off"),
    anchor_modes: tuple[str, str] = ("off", "off"),
) -> tuple[ShardRequest, ...]:
    """Build one request per non-empty shard range, in range order.

    A named module-level seam rather than a comprehension inlined into
    :func:`run_gauntlet`, because this is the last point at which the *parent*
    process holds what each shard is about to be handed -- its device and its
    ``NET_ENV`` -- and the only place a caller can observe that without
    substituting :class:`ShardRequest`, which pickle would then refuse to
    reconstruct in the child.

    Empty ranges are dropped: a shard count above the pair count is legitimate
    (see :func:`shard_ranges`), but a process that plays nothing is pure cost.
    """
    return tuple(
        ShardRequest(
            challenger_path=str(challenger_path),
            challenger_id=challenger_id,
            anchor_path=str(anchor_path),
            anchor_id=anchor_id,
            pairs_start=piece.start,
            pairs_stop=piece.stop,
            games=pin.games,
            seed=seed,
            draw_score=pin.draw_score,
            m=pin.m,
            n=pin.n,
            device=device,
            net_env=os.environ.get(NET_ENV, ""),
            challenger_proof_mode=challenger_modes[0],
            challenger_discovery_mode=challenger_modes[1],
            anchor_proof_mode=anchor_modes[0],
            anchor_discovery_mode=anchor_modes[1],
        )
        for piece in ranges
        if len(piece) > 0
    )


_CHILD_JOIN_S: Final[float] = 5.0
"""How long a terminated shard is given to die before it is killed outright.

``SIGTERM`` then, if that was not enough, ``SIGKILL``: a shard sitting in a
torch call may not reach its signal handler promptly, and the guarantee this
module owes its caller is that ``run_gauntlet`` leaves no live child, not that
it leaves none politely.
"""


def _children(executor: ProcessPoolExecutor) -> list[BaseProcess]:
    """Snapshot the worker processes `executor` owns. Never raises.

    Must be taken **before** ``shutdown``: that method drops the executor's own
    handle on its workers (``self._processes = None``) as its last act, whether
    or not it waited for them, so a snapshot taken afterwards is empty and the
    children go unreaped.

    ``_processes`` is private to :class:`~concurrent.futures.ProcessPoolExecutor`
    and is read defensively for exactly that reason: a future CPython that
    renames it costs the cleanup below, not the campaign its crash pause.
    """
    processes: Mapping[int, BaseProcess] = getattr(executor, "_processes", None) or {}
    return list(processes.values())


def _reap(children: list[BaseProcess]) -> None:
    """Terminate and join every process in `children`. Never raises.

    ``shutdown(wait=False, cancel_futures=True)`` returns without waiting -- by
    design, since the reason we are here may be a child that will never finish
    -- so the children are this function's to end. Joining is not tidiness: a
    child that exited on its own leaves a **zombie** until it is reaped, and a
    zombie's pid is still live to ``os.kill(pid, 0)``, so "the gauntlet left no
    live child" would be untestable and, on a long campaign, untrue of the
    process table.
    """
    for child in children:
        try:
            if child.is_alive():
                child.terminate()
        except (OSError, ValueError, AssertionError):  # pragma: no cover - racy
            continue
    for child in children:
        try:
            child.join(_CHILD_JOIN_S)
            if child.is_alive():
                child.kill()
                child.join(_CHILD_JOIN_S)
        except (OSError, ValueError, AssertionError):  # pragma: no cover - racy
            continue


def run_gauntlet(
    root: str | os.PathLike[str],
    challenger_path: str | os.PathLike[str],
    challenger_id: str,
    pin: AnchorPin,
    param_keys: Sequence[str],
    *,
    seed: int,
    device: str = "cpu",
    shards: int = 4,
    min_decisive: int,
    deadline_s: float | None = None,
    gauntlet_profile: str = GAUNTLET_PROFILE_LEGACY,
    selfplay_modes: tuple[str, str] | None = None,
) -> GauntletResult:
    """Verify the anchor, play the gauntlet in subprocesses, and score it.

    The campaign process **plays nothing**: it verifies, fans out, merges and
    scores. ``spawn`` rather than ``fork`` for the reason every other
    multi-process seam in this repo uses it -- a forked child inherits torch's
    threading state and the parent's CUDA context.

    **Every shard is supervised, under one deadline.** One future per shard,
    all submitted before any is collected, and every surviving child terminated
    and joined on the way out whatever happened -- so this call returns or
    raises within roughly `deadline_s` *in total*, and leaves no process behind
    either way. The deadline is absolute: it is fixed once, before the first
    wait, and each wait gets whatever is left of it, because the shards run
    concurrently and a per-shard timer would silently bound the call at
    ``shards x deadline_s``. Its predecessor, ``Pool.map``, did none of that: it
    waits with no timeout and reconciles no dead child, so an ``os._exit`` in a
    shard or a wedged search hung the whole campaign indefinitely (review
    CP-04).

    There is deliberately **no retry**: a shard is replayed by re-running the
    boundary, not inside it. ``campaign_records`` has no per-iteration dedupe
    and the re-pin rule reads ``scores[-2:]``, so a second gauntlet within one
    iteration would write a second score for that iteration and silently change
    what the anchor re-pin is a function of.

    Args:
        gauntlet_profile: Which search BOTH sides play --
            :data:`GAUNTLET_PROFILE_LEGACY` (the default, and what every
            gauntlet ever played has used) or :data:`GAUNTLET_PROFILE_SELFPLAY`.
            The gauntlet is a net-vs-net comparison against a frozen opponent,
            so changing the search mid-campaign changes what the floor measures
            without changing the floor.
        selfplay_modes: The campaign's self-play pair, read only under the
            ``selfplay`` profile, which REQUIRES it: ``None`` is refused there
            rather than standing in for ``off``/``off``, because a self-play
            profile that quietly played the legacy search is the one outcome the
            switch exists to prevent. Under ``legacy`` it is ignored.
        deadline_s: Wall limit for the **whole** fan-out, measured from the
            moment the shards are submitted -- not a per-shard allowance.
            ``None`` reads
            :data:`~komugi.campaign.pinned.GAUNTLET_SHARD_DEADLINE_S` **at call
            time**, so the pin is one number rather than a default frozen into
            this signature at import.

    Raises:
        ValueError: If the ``selfplay`` profile is selected without a
            `selfplay_modes` pair, or if the pin's recorded modes are not the
            modes this call would play. Both are configuration errors -- a
            mis-launched gauntlet, not a gauntlet that came to nothing -- so
            neither is folded into a :class:`GauntletAborted` verdict the
            campaign would record as a failed anchor.
        GauntletAborted: If the anchor does not verify. No games are played.
        GauntletStalled: If any shard is still running when the deadline is
            reached.
        GauntletLost: If a shard process died without returning a result, or if
            the pool was already broken when the shards were submitted.
    """
    anchor_path = verify_anchor(root, pin, param_keys)
    anchor_id = f"anchor-g{pin.generation}"
    ranges = shard_ranges(pin.games // 2, shards)
    # BOTH sides from one profile, resolved once here: an internal gauntlet is a
    # net comparison, and the two entrants differing in search would make the
    # score a measurement of the search instead of of the net.
    if gauntlet_profile == GAUNTLET_PROFILE_SELFPLAY and selfplay_modes is None:
        raise ValueError(
            f"the {GAUNTLET_PROFILE_SELFPLAY!r} gauntlet profile needs "
            f"selfplay_modes: it plays whatever self-play plays, and there is "
            f"no such pair to read here. Pass the campaign's pair, or launch "
            f"with the {GAUNTLET_PROFILE_LEGACY!r} profile"
        )
    modes = gauntlet_modes(
        gauntlet_profile, ("off", "off") if selfplay_modes is None else selfplay_modes
    )
    # The pin says which search the gauntlet against THIS anchor is played with,
    # and that is what a census reads back. Playing a different one would make
    # the pinfile's provenance a claim about a run that did not happen -- and
    # scores from the two searches are not comparable, so the floor would be
    # evaluated against a history measured under another algorithm.
    if (pin.proof_mode, pin.discovery_mode) != modes:
        raise ValueError(
            f"anchor generation {pin.generation} was pinned under search modes "
            f"{(pin.proof_mode, pin.discovery_mode)!r} (profile_source "
            f"{pin.profile_source!r}), but this gauntlet would play {modes!r} "
            f"under gauntlet_profile {gauntlet_profile!r}: the pin records "
            f"{(pin.proof_mode, pin.discovery_mode)!r} but this gauntlet would "
            f"play {modes!r}; re-pin the anchor with the intended profile or "
            f"launch with the recorded one"
        )
    requests = _shard_requests(
        challenger_path,
        challenger_id,
        anchor_path,
        anchor_id,
        pin,
        ranges,
        seed=seed,
        device=device,
        challenger_modes=modes,
        anchor_modes=modes,
    )
    deadline = (
        pinned.GAUNTLET_SHARD_DEADLINE_S if deadline_s is None else float(deadline_s)
    )
    started = time.monotonic()
    context = multiprocessing.get_context("spawn")
    executor = ProcessPoolExecutor(
        max_workers=max(len(requests), 1), mp_context=context
    )
    played: list[tuple[tuple[GameRecord, ...], tuple[int, ...]]] = []
    try:
        # One future per shard, submitted before any is collected: the fan-out
        # is the point, and `run_shard` is looked up on the module at call time
        # so a test can substitute it.
        try:
            futures = [executor.submit(run_shard, request) for request in requests]
        except BrokenProcessPool as exc:
            # A pool that broke while it was being filled -- a child that died
            # during its own start-up, a fork/spawn refused by the box. Typed
            # here for the same reason the collection path below is: the caller
            # is promised this module's two errors, and a bare stdlib class
            # arriving from *submit* rather than from *result* would be the
            # same infrastructure failure under a name nothing catches.
            raise GauntletLost(
                f"the gauntlet's shard pool was already broken while its "
                f"{len(requests)} shard(s) were being submitted against anchor "
                f"generation {pin.generation}: {exc}. No games were played and "
                f"no score was recorded"
            ) from exc
        # ONE deadline for the whole fan-out, fixed here and spent down by the
        # waits below -- not a fresh timer per shard. The shards run
        # concurrently (`max_workers` is one per request), so their walls
        # overlap and a per-shard timer would bound this call at
        # `len(futures) * deadline` while claiming to bound it at `deadline`
        # (KOM-180 impl review F6).
        end = time.monotonic() + deadline
        for index, future in enumerate(futures):
            try:
                played.append(future.result(timeout=max(0.0, end - time.monotonic())))
            except FutureTimeout as exc:
                # Imported under its own name rather than caught as the builtin
                # it became an alias of in 3.11: what is being caught here is a
                # *future* that did not finish, not an ``OSError`` timeout, and
                # the two only happen to be the same class today.
                # We raise on the first shard to overrun rather than polling the
                # rest: the others are cancelled and killed in the `finally`, so
                # the whole call is bounded by one deadline, not by N of them.
                raise GauntletStalled(
                    f"gauntlet shard {index} of {len(futures)} (pairs "
                    f"{requests[index].pairs_start}..{requests[index].pairs_stop}) "
                    f"had not returned {deadline:g}s after the fan-out started, "
                    f"against anchor generation {pin.generation}. Campaign 6's "
                    f"64-game gauntlets take 130-385s of wall for all shards "
                    f"together (`gauntlet_wall_s`), and the pinned deadline is "
                    f"ten times the worst of those, so a gauntlet that reaches "
                    f"it is wedged rather than slow. Every shard process has "
                    f"been killed; no score was recorded"
                ) from exc
            except BrokenProcessPool as exc:
                # A child that died without returning: a signal, an `os._exit`,
                # the OOM killer. The pool is broken for every other shard too,
                # which is why this is terminal rather than per-shard.
                raise GauntletLost(
                    f"a gauntlet shard process died without returning a result "
                    f"(collecting shard {index} of {len(futures)} against anchor "
                    f"generation {pin.generation}): {exc}. The surviving shard "
                    f"processes have been killed; no score was recorded"
                ) from exc
    finally:
        # Not tidiness: without this, a stalled shard outlives the campaign
        # process's gauntlet and keeps playing games nobody will read, and an
        # abruptly-dead one stays a zombie. `cancel_futures` stops the ones that
        # never started; `_reap` ends the ones that did -- over a snapshot taken
        # first, because `shutdown` drops the executor's handle on its workers
        # even when it did not wait for them.
        children = _children(executor)
        executor.shutdown(wait=False, cancel_futures=True)
        _reap(children)
    wall_s = time.monotonic() - started

    outcome = merge_outcomes(played)
    return GauntletResult(
        generation=pin.generation,
        score=score_for(outcome.records, challenger_id),
        summary=outcome.summary,
        wall_s=wall_s,
        uninformative=outcome.summary.decisive < min_decisive,
    )
