"""Pause, ack, mute: a stop that can be un-stopped, exactly once, durably.

KOM-022 §5. An auto-pause whose un-pause path does not close is worse than no
auto-pause, because the campaign stops and *stays* stopped -- so the protocol
is pinned here rather than assembled at the call site.

The three states, and they are a pure function of which files exist:

===========  ==============================  =================================
state        files                           ``run_campaign``
===========  ==============================  =================================
``RUNNING``  neither ``PAUSED`` nor ``ACK``  runs
``PAUSED``   ``PAUSED``, no ``ACK``          raises, before creating any
                                             directory or spawning anything
``ACKED``    ``PAUSED`` **and** ``ACK``      validates, transitions, then runs
===========  ==============================  =================================

``ACK`` with no ``PAUSED`` is not a fourth state: it is the stale-ack-outliving-
its-pause case, and it is refused loudly.

**One owner.** The *loop* performs every transition. The operator's only action
is to write ``<root>/ACK`` beside a still-present ``PAUSED``; the operator never
deletes or moves anything. That single rule is what makes the protocol
implementable -- "the operator removes ``PAUSED``, then the loop validates the
ack against ``PAUSED``" cannot both hold.

One owner *of the ack protocol*, and the qualifier matters: a tripwire in
:data:`~komugi.campaign.pinned.UNMUTABLE` has no ack path at all (step 4 raises
:class:`PauseError` for it), so its recovery is not a transition this module
performs -- it is a rollback the operator performs with the campaign stopped,
and :func:`_unacked_message` is where the loop prints it rather than printing an
``ACK`` instruction that would return an error (KOM-211).

**The transition, in order** (:func:`apply_startup_state`):

1. read ``PAUSED``, ``ACK`` **and** ``<root>/mutes/`` -- the mute read happens
   on *every* start including the ``RUNNING`` path, because a mute must survive
   the very restart that has nothing left to remind the loop it exists;
2. ``PAUSED`` without ``ACK`` -> raise, change nothing;
3. ``ACK`` without ``PAUSED`` -> raise, change nothing;
4. both -> validate iteration **and** tripwire against ``PAUSED``;
5. **claim** the ack: ``os.replace(ACK, ACK.<i>.consumed)``. Atomic, so two
   ``run_campaign`` processes on one root cannot both apply one ack: the
   loser's call raises ``FileNotFoundError`` on a source that no longer exists,
   and the loser then **refuses outright without re-reading the state table**
   (re-reading after step 6 would find ``RUNNING`` and run, which is two
   campaigns on one root). This serialises *the ack*, not the campaign --
   §11.2's absent lockfile is unchanged and undisguised;
5b. **publish the mute durably, before anything is archived**:
   ``<root>/mutes/<tripwire>.json``. This is what makes an ack worth more than
   ``rm PAUSED``, and it is deliberately ordered *before* step 6 so that no
   crash can consume an ack and lose what it bought;
6. **archive** the pause: ``os.replace(PAUSED, PAUSED.<i>.acked)``. Archived,
   not deleted -- every pause and its ack survive for a post-mortem;
7. the caller writes the mute and its expiry to the journal. **The journal line
   is telemetry about the mute; step 5b's file is the mute.**

Every crash window is conservative: a crash between 5 and 6 leaves ``PAUSED``
present and the next start refuses again (the operator writes a fresh ``ACK``,
and 5b rewrites byte-identical content, because the mute file is keyed by
*tripwire*, not by attempt); a crash between 6 and 7 leaves ``RUNNING`` with the
mute on disk, and step 1 reads it back. The failure mode is "asks again", never
"silently resumes".

Expiry is a property of the file's **content**, not of an action: a mute
suppresses its tripwire for ``from_iteration <= j < until_iteration`` and is
inert after, so a mute nobody ever cleared has already stopped muting. The
archival in :func:`expire_mutes` is tidiness with a post-mortem trail, not the
mechanism.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from komugi.campaign.pinned import MUTE_DEFAULTS, TRIPWIRES, UNMUTABLE, mute_length

PAUSE_NAME: Final[str] = "PAUSED"
ACK_NAME: Final[str] = "ACK"
MUTES_DIRNAME: Final[str] = "mutes"

TRIPWIRE_CLASS: Final[str] = "tripwire"
CRASH_CLASS: Final[str] = "crash"
INTERRUPT_CLASS: Final[str] = "interrupt"
PAUSE_CLASSES: Final[frozenset[str]] = frozenset(
    {TRIPWIRE_CLASS, CRASH_CLASS, INTERRUPT_CLASS}
)
"""Why the campaign stopped, and therefore whether it refuses to restart.

A tripwire and a crash refuse -- a crash may have poisoned the newest
checkpoint, which is exactly what must not be resumed onto blindly. A
``KeyboardInterrupt`` or ``SystemExit`` does **not**: a deliberate stop is not
a fault, and an unattended campaign that refuses to restart after an ``ssh``
disconnect is a night lost for a benign reason. The interrupt marker is for the
operator's benefit and the next start clears it.
"""

CRASH_TRIPWIRE: Final[str] = "crash"
"""The ``tripwire`` field of a ``class: crash`` pause.

An ``ACK`` naming it *clears* the pause and writes **no** mute file: there is
no recurring statistic to mute. A deliberate case, not a failed write.
"""

_FILE_MODE: Final[int] = 0o644

_CKPT_DIRNAME: Final[str] = "ckpt"
_EMA_DIRNAME: Final[str] = "ema"
"""The two publish directories the rollback message names, spelled here.

Mirrors :data:`~komugi.campaign.loop.CHECKPOINT_DIRNAME` and
:data:`~komugi.campaign.loop.EMA_DIRNAME`, which cannot be imported: ``loop``
imports *this* module, so the dependency only runs one way. Two names in a
message rather than an import cycle, and the layout they describe is fixed by
:func:`~komugi.campaign.loop.checkpoint_dir` / :func:`~komugi.campaign.loop.ema_dir`
and asserted by every test that plants a root.
"""

_STEP_DIGITS: Final[int] = 9
"""How :func:`~komugi.train.checkpoint.save_checkpoint` pads a step in a filename.

``checkpoint-000000124-<digest>.pt``. The glob this module prints has to pad the
same way or the ``rm`` an operator pastes matches nothing.
"""


class PauseError(RuntimeError):
    """The pause/ack/mute state on disk cannot be validated. Nothing changed."""


class AckRaceLost(PauseError):
    """Another process claimed this root's ``ACK`` first; this one refuses."""


class CampaignPaused(RuntimeError):
    """The campaign stopped at an iteration boundary and did not finish.

    Carries the :class:`~komugi.campaign.loop.IterationResult` s completed so
    far. Raising rather than returning early is deliberate: ``run_campaign``
    documents its return as "one result per iteration *this call ran*", and a
    completed root already returns an empty list -- so a pause that returned
    would be indistinguishable from success to every caller and every watcher.
    """

    def __init__(self, results: Sequence[object], reason: str) -> None:
        super().__init__(reason)
        self.results: tuple[object, ...] = tuple(results)
        self.reason = reason


@dataclass(frozen=True, slots=True)
class PauseRecord:
    """The contents of ``<root>/PAUSED``."""

    pause_class: str
    paused_at_iteration: int
    tripwire: str
    detail: str
    newest_checkpoint: str = ""
    newest_ema: str = ""
    last_good_checkpoint: str = ""
    written_at_step: int = 0

    def as_json(self) -> dict[str, Any]:
        """Return the JSON object this record is stored as."""
        return {
            "class": self.pause_class,
            "paused_at_iteration": self.paused_at_iteration,
            "tripwire": self.tripwire,
            "detail": self.detail,
            "newest_checkpoint": self.newest_checkpoint,
            "newest_ema": self.newest_ema,
            "last_good_checkpoint": self.last_good_checkpoint,
            "written_at_step": self.written_at_step,
        }


@dataclass(frozen=True, slots=True)
class Mute:
    """One live or expired suppression, as ``<root>/mutes/<tripwire>.json``."""

    tripwire: str
    acked_iteration: int
    from_iteration: int
    until_iteration: int
    source: str = "ACK"

    def covers(self, iteration: int) -> bool:
        """Return whether this mute suppresses its tripwire at `iteration`.

        Half-open, and written at ``from_iteration = acked_iteration + 1``: an
        ``n``-iteration mute acked at ``i`` covers ``i+1 .. i+n`` and
        ``i+n+1`` can pause again.
        """
        return self.from_iteration <= iteration < self.until_iteration

    def as_json(self) -> dict[str, Any]:
        """Return the JSON object this mute is stored as."""
        return {
            "tripwire": self.tripwire,
            "acked_iteration": self.acked_iteration,
            "from_iteration": self.from_iteration,
            "until_iteration": self.until_iteration,
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class StartupState:
    """What :func:`apply_startup_state` found, and what it did about it."""

    running: bool
    mutes: Mapping[str, Mute]
    applied: Mute | None = None
    cleared_interrupt: bool = False


def pause_path(root: str | os.PathLike[str]) -> Path:
    """Return ``<root>/PAUSED``, whether or not it exists."""
    return Path(root) / PAUSE_NAME


def ack_path(root: str | os.PathLike[str]) -> Path:
    """Return ``<root>/ACK``, whether or not it exists."""
    return Path(root) / ACK_NAME


def mutes_dir(root: str | os.PathLike[str]) -> Path:
    """Return ``<root>/mutes``, whether or not it exists."""
    return Path(root) / MUTES_DIRNAME


def _fsync_directory(directory: Path) -> None:
    """``fsync`` a directory so a rename into it is durable, not just atomic."""
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_json_durably(path: Path, payload: Mapping[str, Any]) -> None:
    """Write `payload` to `path` atomically and durably.

    Temp file -> ``fsync`` -> ``os.replace`` -> ``fsync`` the directory: the
    idiom :func:`~komugi.train.checkpoint.save_checkpoint` already uses, for
    the same reason. A watcher must never read a half-written pause reason,
    and a mute must survive ``kill -9``, the OOM killer and a host reboot on
    exactly the terms ``PAUSED`` itself does.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode("utf-8")
    tmp = path.parent / f".{path.name}.tmp-{os.getpid()}"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, _FILE_MODE)
    try:
        written = 0
        while written < len(data):
            written += os.write(fd, data[written:])
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, path)
    _fsync_directory(path.parent)


def _read_json(path: Path) -> dict[str, Any]:
    """Return `path`'s JSON object, or raise :class:`PauseError`.

    A truncated or hand-mangled file is a **refusal**, never a crash part-way
    through a transition: ``ACK`` is written by a human with an editor, so a
    partial write is a live possibility rather than a hypothetical.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PauseError(f"{path} could not be read: {exc}") from exc
    try:
        parsed: object = json.loads(text)
    except ValueError as exc:
        raise PauseError(
            f"{path} is not valid JSON ({exc}); refusing to act on a "
            f"half-written file and changing nothing"
        ) from exc
    if not isinstance(parsed, dict):
        raise PauseError(f"{path} is not a JSON object: {parsed!r}")
    return parsed


def _require_int(payload: Mapping[str, Any], key: str, path: Path) -> int:
    """Return an integer field of `payload`, or raise :class:`PauseError`."""
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise PauseError(f"{path}: field {key!r} must be an integer, got {value!r}")
    return value


def _require_str(payload: Mapping[str, Any], key: str, path: Path) -> str:
    """Return a string field of `payload`, or raise :class:`PauseError`."""
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise PauseError(
            f"{path}: field {key!r} must be a non-empty string, got {value!r}"
        )
    return value


def write_pause(root: str | os.PathLike[str], record: PauseRecord) -> Path:
    """Write ``<root>/PAUSED`` durably and return its path."""
    if record.pause_class not in PAUSE_CLASSES:
        raise ValueError(
            f"pause class must be one of {sorted(PAUSE_CLASSES)}, got "
            f"{record.pause_class!r}"
        )
    path = pause_path(root)
    write_json_durably(path, record.as_json())
    return path


def read_pause(root: str | os.PathLike[str]) -> PauseRecord | None:
    """Return the pause on `root`, or ``None`` if there is none.

    Raises:
        PauseError: If ``PAUSED`` exists but cannot be validated. A pause file
            that cannot be read is not a licence to run.
    """
    path = pause_path(root)
    if not path.exists():
        return None
    payload = _read_json(path)
    pause_class = _require_str(payload, "class", path)
    if pause_class not in PAUSE_CLASSES:
        raise PauseError(
            f"{path}: class {pause_class!r} is not one of {sorted(PAUSE_CLASSES)}"
        )
    return PauseRecord(
        pause_class=pause_class,
        paused_at_iteration=_require_int(payload, "paused_at_iteration", path),
        tripwire=_require_str(payload, "tripwire", path),
        detail=str(payload.get("detail", "")),
        newest_checkpoint=str(payload.get("newest_checkpoint", "")),
        newest_ema=str(payload.get("newest_ema", "")),
        last_good_checkpoint=str(payload.get("last_good_checkpoint", "")),
        written_at_step=int(payload.get("written_at_step", 0)),
    )


def read_mutes(root: str | os.PathLike[str]) -> dict[str, Mute]:
    """Return every mute file under ``<root>/mutes``, keyed by tripwire.

    Read at step 1 of **every** start, on every path -- including ``RUNNING``,
    where there is no ``PAUSED`` and no ``ACK`` left to remind the loop the
    mute exists. Expired mutes are returned too; expiry is decided by the
    caller against the iteration it is about to run, because a mute's expiry is
    content and not an event.

    Raises:
        PauseError: On a mute naming an unknown tripwire, one naming a gate
            that is unmutable at any length, one longer than its tripwire's
            default (the ``ACK`` rule enforced on the durable copy, which
            outlives the ``ACK`` that justified it), or one that is malformed
            or truncated. All four raise and change nothing: a mute that
            cannot be validated must never silently suppress a tripwire.
    """
    directory = mutes_dir(root)
    if not directory.is_dir():
        return {}
    mutes: dict[str, Mute] = {}
    for path in sorted(directory.glob("*.json")):
        payload = _read_json(path)
        tripwire = _require_str(payload, "tripwire", path)
        if tripwire in UNMUTABLE:
            raise PauseError(
                f"{path} mutes {tripwire!r}, which is not mutable at any "
                f"length: it has no false-alarm mode. Refusing to start"
            )
        if tripwire not in MUTE_DEFAULTS:
            raise PauseError(
                f"{path} names unknown tripwire {tripwire!r}; known: "
                f"{sorted(TRIPWIRES)}"
            )
        if path.stem != tripwire:
            raise PauseError(
                f"{path} is keyed {path.stem!r} but names tripwire "
                f"{tripwire!r}: a mute is one file per tripwire, so the two "
                f"must agree or 'which mutes are live' is not a listing"
            )
        from_iteration = _require_int(payload, "from_iteration", path)
        until_iteration = _require_int(payload, "until_iteration", path)
        span = until_iteration - from_iteration
        if span < 0 or span > MUTE_DEFAULTS[tripwire]:
            raise PauseError(
                f"{path} mutes {tripwire!r} for {span} iterations; its "
                f"pre-registered default is {MUTE_DEFAULTS[tripwire]} and an "
                f"ack may shorten a mute and never lengthen one"
            )
        mutes[tripwire] = Mute(
            tripwire=tripwire,
            acked_iteration=_require_int(payload, "acked_iteration", path),
            from_iteration=from_iteration,
            until_iteration=until_iteration,
            source=str(payload.get("source", "ACK")),
        )
    return mutes


def live_mutes(root: str | os.PathLike[str], iteration: int) -> dict[str, Mute]:
    """Return the mutes suppressing a tripwire *at* `iteration`."""
    return {
        name: mute for name, mute in read_mutes(root).items() if mute.covers(iteration)
    }


def expire_mutes(root: str | os.PathLike[str], iteration: int) -> tuple[str, ...]:
    """Archive every mute `iteration` has passed, and return their tripwires.

    ``mutes/<tripwire>.json`` -> ``mutes/<tripwire>.<until>.expired``, then
    ``fsync`` the directory. Archived rather than deleted, for the reason
    ``PAUSED`` is archived. A crash before this is harmless: the file is read
    back, found expired by its own content, and suppresses nothing.
    """
    directory = mutes_dir(root)
    if not directory.is_dir():
        return ()
    expired: list[str] = []
    for name, mute in read_mutes(root).items():
        if iteration < mute.until_iteration:
            continue
        source = directory / f"{name}.json"
        os.replace(source, directory / f"{name}.{mute.until_iteration}.expired")
        expired.append(name)
    if expired:
        _fsync_directory(directory)
    return tuple(expired)


def _validate_ack(ack: Mapping[str, Any], pause: PauseRecord, path: Path) -> int | None:
    """Return the mute length `ack` buys, or ``None`` for an ack that mutes nothing.

    Raises:
        PauseError: If the ack is not an ack of *this* pause, or names a gate
            that may not be acked at all.
    """
    acked_iteration = _require_int(ack, "acked_iteration", path)
    tripwire = _require_str(ack, "tripwire", path)
    if acked_iteration != pause.paused_at_iteration:
        raise PauseError(
            f"{path} acks iteration {acked_iteration} but this root is paused "
            f"at iteration {pause.paused_at_iteration}: an ack must be an ack "
            f"of *this* pause. Nothing on disk was changed"
        )
    if tripwire != pause.tripwire:
        raise PauseError(
            f"{path} acks tripwire {tripwire!r} but this root is paused on "
            f"{pause.tripwire!r}. Nothing on disk was changed"
        )
    if tripwire in UNMUTABLE:
        raise PauseError(
            f"{path} acks {tripwire!r}, which cannot be acked: a frozen net is "
            f"never a false alarm, so there is nothing here for an operator to "
            f"wave through. Fix the cause"
        )
    if pause.pause_class == CRASH_CLASS or tripwire == CRASH_TRIPWIRE:
        # A crash pause is *cleared* by an ack and mutes nothing: there is no
        # recurring statistic to mute. Step 5b therefore writes no file, which
        # is a deliberate case rather than a failed write.
        return None
    requested = ack.get("mute_iterations")
    if requested is not None and (
        isinstance(requested, bool) or not isinstance(requested, int)
    ):
        raise PauseError(
            f"{path}: mute_iterations must be an integer, got {requested!r}"
        )
    try:
        return mute_length(tripwire, requested)
    except ValueError as exc:
        raise PauseError(f"{path}: {exc}") from exc


def _unacked_message(root: Path, pause: PauseRecord, ack_file: Path) -> str:
    """Return what an unacked `pause` tells the operator to do about it.

    Two messages, because there are two kinds of pause and only one of them can
    be resumed. A **mutable** tripwire is waved through with an ``ACK``, and the
    message is the file to write, verbatim enough to paste.

    An **unmutable** one cannot: :func:`_validate_ack` raises
    :class:`PauseError` for every member of
    :data:`~komugi.campaign.pinned.UNMUTABLE` before a single byte on disk is
    changed, so "to resume, write ``<root>/ACK``" -- which is what this function
    used to print for *every* pause -- handed the operator, mid-incident, an
    instruction that returns an error (KOM-180 impl review F3/F4). What it prints
    instead is the rollback: drop the pause marker, drop the abandoned attempt's
    trainer checkpoint and its EMA sibling, relaunch -- or fork the root at the
    last judged iteration, whose checkpoint ``last_good_checkpoint`` names.

    The step in those two globs is ``written_at_step``, the step the pause was
    written at, which for
    :data:`~komugi.campaign.pinned.UNFINISHED_BOUNDARY` is exactly the abandoned
    attempt's ``(i + 1) * steps_per_iteration``. A ``PAUSED`` that carries no
    step (an older marker, or a hand-planted one) gets the same recipe with the
    step left as a placeholder rather than a wrong number.
    """
    head = (
        f"campaign root {root} is paused at iteration "
        f"{pause.paused_at_iteration} on {pause.tripwire!r} "
        f"({pause.pause_class}): {pause.detail}."
    )
    if pause.tripwire not in UNMUTABLE:
        return (
            f"{head} To resume, write {ack_file} containing "
            f'{{"acked_iteration": {pause.paused_at_iteration}, "tripwire": '
            f'"{pause.tripwire}"}} beside the still-present PAUSED file; the '
            f"loop does the rest"
        )
    step = (
        f"{pause.written_at_step:0{_STEP_DIGITS}d}"
        if pause.written_at_step > 0
        else "<(i+1)*steps_per_iteration>"
    )
    fork_from = (
        pause.last_good_checkpoint
        if pause.last_good_checkpoint
        else "no committed checkpoint (start a fresh root)"
    )
    abandoned_name = f"checkpoint-{step}-*.pt"
    abandoned_train = Path(root) / _CKPT_DIRNAME / abandoned_name
    abandoned_ema = Path(root) / _EMA_DIRNAME / abandoned_name
    return (
        f"{head} {pause.tripwire!r} cannot be acked: an ACK naming it is "
        f"refused with a PauseError and clears nothing, so there is no "
        f"resume-in-place for this pause. Fix the cause, or roll the root back "
        f"to its last judged iteration: rm {pause_path(root)}, "
        f"rm {abandoned_train}, rm {abandoned_ema} -- then "
        f"relaunch; or fork the root at iteration "
        f"{pause.paused_at_iteration - 1} from {fork_from}. The EMA file must go "
        f"with the trainer checkpoint: left behind it is the newest published "
        f"EMA, so the re-run of iteration {pause.paused_at_iteration} would play "
        f"self-play against the abandoned attempt's smoothed net instead of "
        f"EMA_{pause.paused_at_iteration - 1}, silently changing the recursion "
        f"the displacement band is derived from, and no gate notices"
    )


def apply_startup_state(
    root: str | os.PathLike[str],
    *,
    hook: Callable[[str], None] | None = None,
) -> StartupState:
    """Read this root's pause state, apply a validated ack, and say what happened.

    The whole of the module docstring's seven steps. Called **once**, at the
    top of ``run_campaign``, before any directory is created or anything is
    spawned.

    `hook` is called with the name of each completed step (``"claimed"``,
    ``"muted"``, ``"archived"``) and exists so a test can stop the process
    inside a named crash window -- the design tabulates four of them and
    asserts a property in each, and a window that cannot be entered on purpose
    is a window nobody tests. It is never set in production.

    Raises:
        CampaignPaused: The root holds an unacked tripwire or crash pause.
        AckRaceLost: Another process claimed the ack between step 4 and step 5.
        PauseError: A stale, malformed or mismatched ``ACK``, or an invalid
            mute. Nothing on disk is changed in any of these cases.
    """
    notify = hook if hook is not None else (lambda _step: None)
    mutes = read_mutes(root)
    pause = read_pause(root)
    ack_file = ack_path(root)
    has_ack = ack_file.exists()

    if pause is None:
        if has_ack:
            raise PauseError(
                f"{ack_file} exists but {pause_path(root)} does not: this is a "
                f"stale ack, outliving the pause it was written for. Remove it "
                f"deliberately; this run refuses rather than treating it as a "
                f"permanent mute"
            )
        return StartupState(running=True, mutes=mutes)

    if pause.pause_class == INTERRUPT_CLASS:
        # A deliberate stop is not a fault. The marker is archived rather than
        # deleted, so the operator's post-mortem still finds it.
        #
        # Clearing it says the *stop* was intentional. It says nothing about
        # what the stop interrupted: a Ctrl-C during a boundary leaves a
        # published checkpoint whose iteration never closed, exactly as a kill
        # does. `run_campaign` therefore runs its unfinished-boundary check on
        # this path too -- it is sited after this call, not inside it, so there
        # is no return here that can skip it (review CP-01,
        # :func:`~komugi.campaign.loop._refuse_unfinished_boundary`).
        os.replace(
            pause_path(root),
            Path(root) / f"{PAUSE_NAME}.{pause.paused_at_iteration}.cleared",
        )
        _fsync_directory(Path(root))
        return StartupState(running=True, mutes=mutes, cleared_interrupt=True)

    if not has_ack:
        # The instruction depends on whether this tripwire can be acked at all
        # (:func:`_unacked_message`): an unmutable one is refused by
        # `_validate_ack`, so telling the operator to write an ACK would be
        # telling them to reproduce an error.
        raise CampaignPaused((), _unacked_message(Path(root), pause, ack_file))

    ack = _read_json(ack_file)
    mute_iterations = _validate_ack(ack, pause, ack_file)

    consumed = Path(root) / f"{ACK_NAME}.{pause.paused_at_iteration}.consumed"
    try:
        os.replace(ack_file, consumed)
    except FileNotFoundError as exc:
        # The serialisation point. The loser refuses **without re-reading the
        # state table**: re-reading after the winner's step 6 would find
        # RUNNING and run, which is two campaigns on one root.
        raise AckRaceLost(
            f"another process claimed {ack_file} first; refusing to run rather "
            f"than re-reading a root whose state is being changed underneath "
            f"this process"
        ) from exc
    notify("claimed")

    applied: Mute | None = None
    if mute_iterations is not None:
        # The window opens at the iteration *after* the one that paused, and
        # covers `mute_iterations` of them: an n-iteration mute buys n runnable
        # iterations, which is what §5's own prose says ("iteration i+3 can
        # pause again" for n = 2). Anchoring it at `paused_at` instead spends
        # the first iteration of the window on an iteration that has already
        # completed -- the boundary that fired is over -- so an n-iteration
        # mute silences n-1 (KOM-022 review F2).
        applied = Mute(
            tripwire=pause.tripwire,
            acked_iteration=pause.paused_at_iteration,
            from_iteration=pause.paused_at_iteration + 1,
            until_iteration=pause.paused_at_iteration + 1 + mute_iterations,
        )
        write_json_durably(
            mutes_dir(root) / f"{pause.tripwire}.json", applied.as_json()
        )
        mutes = dict(mutes)
        mutes[applied.tripwire] = applied
    notify("muted")

    os.replace(
        pause_path(root),
        Path(root) / f"{PAUSE_NAME}.{pause.paused_at_iteration}.acked",
    )
    _fsync_directory(Path(root))
    notify("archived")
    return StartupState(running=True, mutes=mutes, applied=applied)
