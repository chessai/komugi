"""What search a campaign played, written down where a reader can find it.

A campaign's rows say nothing about the algorithm that produced them. The
shard :data:`~komugi.selfplay.records.SCHEMA` is frozen (fingerprints depend on
it), the journal's ``extras`` are floats, and a
:class:`~komugi.config.SelfPlayConfig` is never persisted -- so when campaign 6
switched its search on at iteration 571 (KOM-232), the only record of the
switch was a ticket and a launcher's startup line. This module is the
machine-readable half: one JSON object per *search profile*, appended to
``<root>/search_profile.jsonl``, plus a copy of the live profile under every
iteration directory so rows and their provenance travel together.

Three properties carry the whole design.

**A static identity.** :func:`static_digest` hashes the canonical JSON of a
profile with ``schema``, ``written_at``, ``main_sha`` and ``from_iteration``
blanked, so two relaunches of the same configuration produce one digest and the
file grows only when the search or the recipe actually changes. ``schema`` is
blanked with the occasion and for the same reason: it is a property of the
*writer*, so a deploy that bumps it without changing a knob must not read as a
changed search (KOM-237 A2). That digest is what
:func:`~komugi.campaign.loop.guard_search_profile` compares, which is why the
relaunch of an unchanged campaign appends nothing at all.

**Forward-only.** A root whose iterations predate this file has no profile and
never acquires one: :func:`read_profiles` returns ``()`` and a caller must read
that as *unrecorded*, never as observed ``off``/``off``. Rewriting history
would be inventing a measurement.

**Append-only and durable**, the journal's discipline restated for a third
record type (see :func:`~komugi.campaign.telemetry.append_selfplay` for why it
is restated rather than shared): ``O_APPEND`` so the kernel owns the offset,
and ``fsync`` before :func:`append_profile` returns -- the profile for the
iterations a campaign is about to play has to be on the platter before it
plays them, because the campaigns this exists for are the ones that die.

Unlike the two telemetry readers, :func:`read_profiles` **raises** on an
interior line it cannot decode, naming the line number. A torn trailing line in
a log of per-iteration measurements costs one measurement; a torn line here is
the provenance of every row after it, and a reader that silently skipped it
would attribute those rows to the wrong search. The one exception is an
*unterminated* final line: ``O_APPEND`` plus the short-write loop means a line
with no newline after it can only be the profile of a launch that died writing
it -- a launch that therefore never played a game -- so it is warned about and
ignored rather than made into a permanent relaunch blocker. A tolerant reader
alone would not be enough: the *writer* heals it, because an unconditional
``O_APPEND`` onto a fragment fuses the two into one newline-terminated line
that no reader can decode and no reader may ignore. :func:`append_profile`
truncates the fragment first, so the file a torn launch leaves behind is
repaired by the next launch rather than made permanently unreadable by it.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import warnings
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from komugi.selfplay.tactical import GameMarshalCaptureProvider

if TYPE_CHECKING:  # pragma: no cover - typing only
    from komugi.campaign.loop import CampaignConfig
    from komugi.config import SelfPlayConfig

PROFILE_FILENAME: Final[str] = "search_profile.jsonl"
"""The campaign-root log: one line per profile, in the order they took effect."""

ITERATION_PROFILE_FILENAME: Final[str] = "search_profile.json"
"""The per-iteration copy: ``<root>/iter_<i>/search_profile.json``."""

PROFILE_SCHEMA: Final[int] = 2
"""Version of the object below. First field of every line, so a future reader
can dispatch on it without guessing from the key set.

Documentation rather than identity: :func:`static_fields` blanks it, so a bump
alone never refuses a relaunch (KOM-237 A2). Schema 2 appended KOM-249's four
fields; :data:`_SCHEMA_2_DEFAULTS` is what a schema-1 line is read as.
"""

_SCHEMA_2_DEFAULTS: Final[dict[str, object]] = {
    "cap_adjudication": "off",
    "cap_adjudication_p_max": 0.8,
    "cap_adjudication_d_full": 8,
    "min_placements_max": 0,
}
"""The keys schema 2 added, and the value a schema-1 line is read as.

Every one of them is the :class:`~komugi.config.SelfPlayConfig` default, which
is what makes filling them a *transcription* rather than an invention: a
campaign that wrote a schema-1 line ran on a build that had no such knob, so it
played the behaviour these values describe. Only missing keys are filled, and
only on a line whose ``schema`` is below :data:`PROFILE_SCHEMA`.

A key missing from a *current* line is an error, and :func:`_decode` is what
raises it: the dataclass carries these four with defaults (that is how a
migrated payload constructs at all), so the constructor would quietly default
them instead of refusing. That would read a damaged record as an old one --
attributing a search nobody can name to the values a schema-1 run played.
"""

MODE_OFF: Final[str] = "off"
"""The value both :attr:`~komugi.config.SelfPlayConfig.proof_mode` and
:attr:`~komugi.config.SelfPlayConfig.discovery_mode` take when their layer does
not run -- the search campaigns 1 through 6a played."""

_DISCOVERY_PROVIDER_ID: Final[str] = GameMarshalCaptureProvider.provider_id
"""Which tactical provider discovery runs with. Imported rather than restated:
a provider id is provenance, and a copy of it here could drift from the one the
search actually used and misattribute a campaign's games."""

_RECOVERY: Final[str] = "truncate the damaged line and relaunch"
"""What an operator staring at the refusal is supposed to do about it. A line
this reader cannot decode is not recoverable by reading harder, and the file is
append-only: the fix is always to cut the damaged line out and let the guard
re-record the profile."""

_GIT_TIMEOUT_S: Final[float] = 5.0

_PROFILE_MODE: Final[int] = 0o644


@dataclass(frozen=True, slots=True)
class SearchProfile:
    """The search, the recipe and the moment -- one appended line.

    Everything that decides *what a game is* and *what the rows are trained
    into*: the two KOM-223 modes and the algorithm identity they name, the
    Gumbel budgets and scales, the contempt, the draft threshold, the
    repetition rule, the moves-left utility's four knobs, and the campaign's
    loss recipe. Not the campaign's *size* (workers, games, steps, lr): those
    change how much of a thing was produced, not which thing it was, and they
    are already on the per-iteration records.

    ``from_iteration``, ``written_at`` and ``main_sha`` are the *occasion*
    rather than the identity -- :func:`static_digest` blanks them, and blanks
    ``schema`` with them -- so relaunching an unchanged campaign from a
    different commit at a different time, on a build that writes a later schema,
    appends nothing.

    Attributes:
        schema: :data:`PROFILE_SCHEMA`.
        proof_mode: :attr:`~komugi.config.SelfPlayConfig.proof_mode`.
        discovery_mode: :attr:`~komugi.config.SelfPlayConfig.discovery_mode`.
        decision_id: The decision rule the pair names.
        target_id: The training target the pair names.
        provider_id: The tactical provider discovery ran with, or ``None``
            when discovery is off -- discovery is the only consumer of one.
        loss_recipe: :attr:`~komugi.campaign.loop.CampaignConfig.loss_recipe`.
        from_iteration: The iteration this profile takes effect at. The
            campaign's start iteration when the line was appended; rows below
            it belong to the previous line, or to no line at all.
        written_at: ISO-8601 UTC, when the line was appended.
        main_sha: ``git rev-parse HEAD`` of the tree that appended it, or
            ``None`` when it could not be read.
        cap_adjudication: :attr:`~komugi.config.SelfPlayConfig.cap_adjudication`
            -- how this campaign's capped games were labelled (KOM-245).
        cap_adjudication_p_max:
            :attr:`~komugi.config.SelfPlayConfig.cap_adjudication_p_max`.
        cap_adjudication_d_full:
            :attr:`~komugi.config.SelfPlayConfig.cap_adjudication_d_full`.
        min_placements_max:
            :attr:`~komugi.config.SelfPlayConfig.min_placements_max` -- the high
            edge of the drawn draft threshold (KOM-246), ``0`` for the fixed one.

    The last four carry defaults and the fields above them do not, which is the
    schema-2 migration (:data:`_SCHEMA_2_DEFAULTS`) expressed in the dataclass:
    a schema-1 line has none of these keys and is read as the behaviour it was
    played under. New fields go on the END with a default, for the reason
    :class:`~komugi.config.SelfPlayConfig` appends its own.
    """

    schema: int
    proof_mode: str
    discovery_mode: str
    decision_id: str
    target_id: str
    provider_id: str | None
    gumbel_m_fast: int
    gumbel_n_fast: int
    gumbel_m_full: int
    gumbel_n_full: int
    prob_fast: float
    fast_gumbel_scale: float
    draft_gumbel_scale: float
    draw_score: float
    min_placements: int
    repetition_rule: str
    mlh_utility_lambda: float
    mlh_gate_tau0: float
    mlh_gate_tau1: float
    mlh_slope_plies: int
    loss_recipe: str
    from_iteration: int | None
    written_at: str | None
    main_sha: str | None
    cap_adjudication: str = "off"
    cap_adjudication_p_max: float = 0.8
    cap_adjudication_d_full: int = 8
    min_placements_max: int = 0


def profile_of(
    selfplay: SelfPlayConfig, campaign: CampaignConfig, from_iteration: int
) -> SearchProfile:
    """Read the profile of the search `campaign` is about to play.

    ``written_at`` is stamped now and ``main_sha`` is read now, which is why
    neither is part of the static identity: this function is called on every
    launch, including the ones that change nothing.
    """
    identity = selfplay.search_identity
    discovery_on = selfplay.discovery_mode != MODE_OFF
    return SearchProfile(
        schema=PROFILE_SCHEMA,
        proof_mode=selfplay.proof_mode,
        discovery_mode=selfplay.discovery_mode,
        decision_id=identity.decision_id,
        target_id=identity.target_id,
        provider_id=_DISCOVERY_PROVIDER_ID if discovery_on else None,
        gumbel_m_fast=int(selfplay.gumbel_m_fast),
        gumbel_n_fast=int(selfplay.gumbel_n_fast),
        gumbel_m_full=int(selfplay.gumbel_m_full),
        gumbel_n_full=int(selfplay.gumbel_n_full),
        prob_fast=float(selfplay.prob_fast),
        fast_gumbel_scale=float(selfplay.fast_gumbel_scale),
        draft_gumbel_scale=float(selfplay.draft_gumbel_scale),
        draw_score=float(selfplay.draw_score),
        min_placements=int(selfplay.min_placements),
        # `.value` rather than the member: the field is a `str` and a canonical
        # line has to decode back to an equal profile without a translation
        # table (`RepetitionRule` is a `str` enum, so the value is its name on
        # the wire: "full" / "placement").
        repetition_rule=str(selfplay.repetition_rule.value),
        mlh_utility_lambda=float(selfplay.mlh_utility_lambda),
        mlh_gate_tau0=float(selfplay.mlh_gate_tau0),
        mlh_gate_tau1=float(selfplay.mlh_gate_tau1),
        mlh_slope_plies=int(selfplay.mlh_slope_plies),
        loss_recipe=campaign.loss_recipe,
        from_iteration=from_iteration,
        written_at=datetime.now(UTC).isoformat(),
        main_sha=main_sha(),
        cap_adjudication=str(selfplay.cap_adjudication),
        cap_adjudication_p_max=float(selfplay.cap_adjudication_p_max),
        cap_adjudication_d_full=int(selfplay.cap_adjudication_d_full),
        min_placements_max=int(selfplay.min_placements_max),
    )


def canonical_json(profile: SearchProfile) -> str:
    """Render `profile` as one line: sorted keys, no whitespace.

    Sorted and compact so the bytes are a function of the *values* alone --
    which is what lets :func:`static_digest` be a digest of a rendering rather
    than of a field-by-field walk nobody would keep in sync with the dataclass.
    """
    return json.dumps(asdict(profile), sort_keys=True, separators=(",", ":"))


def static_fields(profile: SearchProfile) -> dict[str, object]:
    """Return the fields of `profile` that make up its *identity*.

    Everything except the occasion: ``written_at`` and ``main_sha`` differ on
    every relaunch, and ``from_iteration`` differs by construction (the guard
    compares a stored line against a profile for a later iteration), so all
    three are blanked rather than dropped -- the key set stays the schema's, so
    :func:`static_digest` over it stays a digest of a complete object.

    ``schema`` is blanked with them (KOM-237 A2). It says which *writer*
    produced the line, not which search was played: a line read back through
    :func:`_decode`'s defaults describes exactly the run a later build would
    write with those knobs, so a schema bump alone must leave the digest where
    it was. Otherwise the first relaunch after a deploy -- of an untouched
    campaign -- would be refused with ``schema: 1 -> 2`` and the operator would
    have to record a switch that changed nothing. Blanked in the dict rather
    than on the dataclass because the field is an ``int``, not an
    ``int | None``: the record always carries a real schema.

    Both the digest and the message naming what changed are computed from this
    one mapping, so a field the digest ignores can never be reported as the
    reason a launch was refused.
    """
    static = asdict(
        replace(profile, written_at=None, main_sha=None, from_iteration=None)
    )
    static["schema"] = None
    return static


def static_digest(profile: SearchProfile) -> str:
    """Hash what `profile` *is*, ignoring when and where it was written.

    The comparison the resume guard makes: a digest over the whole object
    would report every restart as a changed search. Taken over
    :func:`static_fields`' mapping, rendered by the same canonical rules
    :func:`canonical_json` uses, so the digest and the guard's "what changed"
    message can never disagree about which fields are the identity.
    """
    static = static_fields(profile)
    rendered = json.dumps(static, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def profile_path(root: str | os.PathLike[str]) -> Path:
    """Return ``<root>/search_profile.jsonl``."""
    return Path(root) / PROFILE_FILENAME


def append_profile(root: str | os.PathLike[str], profile: SearchProfile) -> None:
    """Append `profile` to ``<root>/search_profile.jsonl``, durably.

    Creates the file (not its parent) if it is missing. ``O_APPEND`` plus the
    short-write loop plus ``fsync``, for the reason
    :func:`~komugi.campaign.telemetry.append_selfplay` gives: the provenance of
    the iterations about to be played must survive the crash that interrupts
    them.

    An existing file whose last byte is not a newline is *healed* first: the
    unterminated fragment is truncated away (warning about the file and the
    bytes dropped) and the new line is appended to the last complete one. See
    :func:`_heal_torn_tail` -- appending onto the fragment instead would fuse
    it to this profile and leave a terminated line no reader can decode, which
    is strictly worse than the fragment :func:`read_profiles` already tolerates.
    """
    _heal_torn_tail(profile_path(root))
    payload = (canonical_json(profile) + "\n").encode("utf-8")
    fd = os.open(
        profile_path(root), os.O_WRONLY | os.O_CREAT | os.O_APPEND, _PROFILE_MODE
    )
    try:
        written = 0
        while written < len(payload):
            written += os.write(fd, payload[written:])
        os.fsync(fd)
    finally:
        os.close(fd)


def _heal_torn_tail(path: Path) -> None:
    """Cut an unterminated trailing fragment off `path`, if there is one.

    Truncation rather than a newline prefix, and the module docstring's
    invariant is why: with ``O_APPEND`` and the short-write loop above, bytes
    with no newline after them can only be a launch that died *writing its
    profile* -- before the guard returned, before a worker span, before a row
    existed. The fragment is the provenance of nothing, so keeping it (padded
    into its own line) would only give :func:`read_profiles` an interior line
    it must raise on. Absent and empty files, and files already terminated, are
    left exactly as they are.
    """
    try:
        handle = path.open("r+b")
    except FileNotFoundError:
        return
    with handle:
        size = handle.seek(0, os.SEEK_END)
        if size == 0:
            return
        handle.seek(size - 1)
        if handle.read(1) == b"\n":
            return
        handle.seek(0)
        # Whole-file read: this log has one line per real search change, so it
        # is a handful of lines even for a campaign that ran for months.
        cut = handle.read().rfind(b"\n") + 1
        handle.truncate(cut)
        handle.flush()
        os.fsync(handle.fileno())
    warnings.warn(
        f"{path}: dropped {size - cut} unterminated trailing byte(s) before "
        f"appending -- with O_APPEND they can only be a launch that died while "
        f"writing its profile and therefore never played an iteration",
        stacklevel=3,
    )


def read_profiles(root: str | os.PathLike[str]) -> tuple[SearchProfile, ...]:
    """Return every profile under `root`, oldest first.

    An absent file is ``()`` -- a campaign that predates this record, or one
    that has not launched yet -- and never an error.

    An *unterminated* final line -- no newline after it -- that does not decode
    is warned about and ignored rather than raised on. ``O_APPEND`` gives the
    kernel the offset and :func:`append_profile` fsyncs before it returns, so a
    line with nothing after it can only be a launch that died mid-write: it
    never got past the guard, never played an iteration, and no row on disk
    claims it. Raising would make one interrupted launch a permanent blocker on
    every later relaunch of that campaign.

    Raises:
        ValueError: On an *interior* line that is not a complete profile
            object, naming the 1-based line number. Unlike the telemetry
            readers this does not skip and count: a damaged line with lines
            after it is the provenance of every row that followed, and
            continuing past it would attribute those rows to whichever line
            survived.
    """
    path = profile_path(root)
    if not path.is_file():
        return ()
    profiles: list[SearchProfile] = []
    # Read whole so the final line's *termination* is visible: iterating the
    # handle yields the last line with or without its newline identically.
    for number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(keepends=True), start=1
    ):
        if not line.strip():
            continue
        if not line.endswith("\n"):
            decoded = _decode_trailing(line, number, path)
            if decoded is not None:
                profiles.append(decoded)
            continue
        profiles.append(_decode(line, number, path))
    return tuple(profiles)


def _decode_trailing(line: str, number: int, path: Path) -> SearchProfile | None:
    """Decode the unterminated final line, or warn and return ``None``.

    A complete profile that merely lost its newline is still a profile and is
    kept; anything else is the torn tail of an interrupted append. See
    :func:`read_profiles` for why that is a warning rather than an error.
    """
    try:
        return _decode(line, number, path)
    except ValueError:
        warnings.warn(
            f"{path}: the trailing line (line {number}) is incomplete and was "
            f"ignored -- with O_APPEND it can only be a launch that died while "
            f"writing its profile and therefore never played an iteration",
            stacklevel=3,
        )
        return None


def _migrated(payload: dict[str, Any]) -> dict[str, Any]:
    """Fill the keys a payload from an EARLIER schema could not have carried.

    Only the additions of the schemas between the payload's and this build's
    (today: :data:`_SCHEMA_2_DEFAULTS`), and only where the key is absent. A key
    the payload's own schema *did* define stays missing, which is the difference
    between an old line and a damaged one: the first is provenance this build
    can read, the second is a line whose contents nobody knows. Which of the two
    refuses it depends on the key -- :class:`SearchProfile`'s constructor raises
    on a missing field that has no default, and :func:`_decode` raises on one of
    the four that do (they are exactly the keys this function fills, so a
    still-missing one belongs to a line whose own schema defined it). Unknown
    keys are left in place for the constructor to refuse -- a key this build has
    never heard of is a line written by a LATER build, not an older one, and
    dropping it would read a search this code cannot describe as one it can.

    A payload with no ``schema`` at all counts as older than any schema, the
    conservative reading of a line that cannot say what it is. Nothing reaches a
    record that way: ``schema`` has no default on the dataclass, so the
    constructor refuses such a payload however this function filled it.
    """
    schema = payload.get("schema", 0)
    written_by = (
        schema if isinstance(schema, int) and not isinstance(schema, bool) else 0
    )
    if written_by >= PROFILE_SCHEMA:
        return payload
    filled = {k: v for k, v in _SCHEMA_2_DEFAULTS.items() if k not in payload}
    return {**filled, **payload}


def _decode(line: str, number: int, path: Path) -> SearchProfile:
    """Decode one line, or raise naming where it is."""
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{path}: line {number} is not valid JSON: {exc}; {_RECOVERY}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(
            f"{path}: line {number} is not a JSON object, got "
            f"{type(payload).__name__}; {_RECOVERY}"
        )
    migrated = _migrated(payload)
    # The four schema-2 fields carry defaults on the dataclass, so the
    # constructor below cannot tell a line that never had them from one that
    # lost them. `_migrated` fills them for every schema under this build's, so
    # a key still absent here belongs to a line whose OWN schema defined it:
    # damaged, not old.
    missing = [key for key in _SCHEMA_2_DEFAULTS if key not in migrated]
    if missing:
        raise ValueError(
            f"{path}: line {number} is not a search profile: a schema "
            f"{migrated.get('schema')!r} line lacks {missing}; {_RECOVERY}"
        )
    try:
        return SearchProfile(**migrated)
    except TypeError as exc:
        raise ValueError(
            f"{path}: line {number} is not a search profile: {exc}; {_RECOVERY}"
        ) from exc


def latest_profile(root: str | os.PathLike[str]) -> SearchProfile | None:
    """Return the profile in force under `root`, or ``None`` if there is none.

    The last line, because the file is a log in the order the profiles took
    effect.
    """
    profiles = read_profiles(root)
    return profiles[-1] if profiles else None


def main_sha() -> str | None:
    """Return this working tree's ``HEAD``, or ``None`` if it cannot be read.

    Provenance is worth having and never worth failing a campaign for, so every
    way this can go wrong -- no git, no repository, a detached or empty tree, a
    hung invocation -- is ``None``. The repository asked about is the one
    holding *this file*, not the process's working directory, which is
    routinely a campaign root somewhere else entirely.
    """
    try:
        done = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
        return None
    sha = done.stdout.strip()
    return sha or None
