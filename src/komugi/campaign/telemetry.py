"""The self-play half of a campaign's history, written the moment it is known.

:func:`~komugi.train.step.training_iteration` appends an
:class:`~komugi.telemetry.records.IterationMetrics` record per iteration, and
that record is honest about what the *trainer* saw: it reports ``games=0`` and
zeroed game statistics, because the trainer never met the orchestrator's
:class:`~komugi.selfplay.orchestrate.RunSummary`. So a campaign's journal has
always described half of each iteration. Games played, rows written, shard
restarts and how long self-play took existed only in
:func:`~komugi.campaign.loop.run_campaign`'s in-memory return value -- which
is exactly the thing a campaign that *dies* never returns (KOM-042, the
KOM-031 union review's F5). This module is the other half, on disk, per
iteration, before the training step that follows it begins.

Why a sibling file and not the root journal
-------------------------------------------
The obvious move is a third ``kind`` in ``<root>/journal.jsonl``, and the
journal's own docstring makes it sound supported ("the ``kind`` tag is what
lets one journal hold both record types interleaved... :func:`read_records`
selects by type and quietly passes over the other kinds"). It is not: that
tolerance extends only to the kinds in
:mod:`komugi.telemetry.journal`'s ``_DECODERS`` table.
:func:`~komugi.telemetry.journal.decode_line` raises
:class:`~komugi.telemetry.journal.JournalDecodeError` on an *unknown* kind, and
:func:`~komugi.telemetry.journal.read_records` therefore counts such a line in
:attr:`~komugi.telemetry.journal.JournalRead.skipped` -- the same counter that
reports crash-truncated and corrupt lines, which
``tests/telemetry/test_journal.py::test_unknown_kind_and_future_version_are_skipped``
pins and which ``tests/train/behavior/test_checkpoint_journal.py`` asserts is
zero for a healthy journal. Writing self-play telemetry there would make every
healthy campaign report one damaged journal line per iteration, and the
existing readers cannot be taught otherwise from here:
:mod:`komugi.telemetry` is outside this ticket's boundary.

So the telemetry goes to ``<root>/selfplay.jsonl``, a sibling of the journal,
under the same durability discipline the journal uses
(:func:`append_selfplay`): one line per record, ``O_APPEND`` so the kernel owns
the offset, ``fsync`` before the call returns. The line for iteration ``i`` is
on the platter before iteration ``i``'s training step starts, because the whole
claim is about campaigns that die.

A **log, not a keyed table**: one line per self-play run, in the order the
runs happened. A resumed campaign contributes lines only for the iterations it
actually plays, so recovered iterations are never re-emitted -- but an
iteration that genuinely runs *again* (a crash after its shards were sealed
and before its checkpoint was published re-runs it into a cleaned tree, see
:func:`~komugi.campaign.loop.run_campaign`) appends a second line for that
iteration number, describing the second attempt. That is the honest record of
what happened; an auditor summing rows over a campaign takes the last line per
iteration, since the earlier attempt's shard tree no longer exists.

The mirror image of the root journal's decision follows from being
single-kind: a line in this file that is not a ``selfplay`` record is
corruption rather than interleaving, so :func:`read_selfplay` skips it and
counts it, exactly as :func:`~komugi.telemetry.journal.read_records` does with
a torn trailing line. Reading is never fatal; a campaign's history must not be
lost to its last, half-written line.

Wire format -- one JSON object per line, keys sorted, ``NaN``/``Infinity``
rejected by the encoder::

    {"draws":1,"early_shu_games":2,"games":8,"iteration":0,"kind":"selfplay",
     "max_plies":256,"mean_plies":115.5,"median_placements_b":12.0,
     "median_placements_w":11.0,"restarts":0,"rows":611,"shard_dirs":2,
     "turn_cap_games":1,"v":3,"wall_s":41.7}

The four game-shape fields are schema **v2** (KOM-150); the three draft fields
-- the per-side median placements and the early-終 count a draft-minimum
curriculum is ramped on -- are **v3** (KOM-156b). Older lines still decode:
campaign 1's whole ``selfplay.jsonl`` is v1 and campaign 2's is v2, and the
fields a line predates are set to the not-measured sentinels
(:data:`NOT_MEASURED`, and ``NaN`` for the floats) rather than to zeros. "No
draws" and "the draw share was never measured" are different claims about a
campaign, as are "no game ended its draft early" and "nobody counted"; a zero
would let a reader plot the second as the first. Anything outside
:data:`SELFPLAY_READABLE_VERSIONS` is still an undecodable line, not a record
with missing fields.

Relaunch
--------
:func:`archive_telemetry` is the other half of the same claim, for **both** of
a root's logs rather than only this one. A campaign that is restarted onto a
root it has already written must not be able to lose what is there, and v1
lost its telemetry twice by exactly that route. See that function for why the
archive is a *copy* and why the truncation token it obtains is never spent.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from komugi.telemetry.archive import guard_truncation
from komugi.telemetry.journal import JournalDecodeError

SELFPLAY_JOURNAL_NAME: Final[str] = "selfplay.jsonl"
"""The sibling file, next to ``journal.jsonl`` under the campaign root."""

TELEMETRY_ARCHIVE_DIRNAME: Final[str] = "telemetry_archive"
"""Where a relaunch parks the previous launch's logs: ``<root>/telemetry_archive``.

A directory under the campaign root, not a sibling of it: the root is the unit
an operator copies off a dead box, and a history kept next to it is a history
that does not travel. Its contents are named by
:func:`~komugi.telemetry.archive.archive_name`, i.e. ``<log>.<mtime_ns>``, so
``sorted(...)`` over the directory replays a root's launches oldest-first.
"""

ARCHIVE_RETAINED: Final[int] = 8
"""How many archived copies of **one** log a root keeps, newest first.

The archive is a whole copy of the log per launch, and a launch is not a rare
event: an interrupt marker is cleared and the start continues
(:func:`~komugi.campaign.pause.apply_startup_state`), so a supervision loop that
restarts a campaign leaves one full copy of a monotonically growing journal per
restart. Unbounded, a root under a flapping supervisor fills its own disk with
copies of its own history -- and the thing it runs out of space for is the live
journal every gate reads back (KOM-025 wire-in review, F7).

Bounded per log rather than over the directory, so a big journal cannot evict
the self-play sibling's history. Eight because the copies exist to answer "what
did the previous launches leave behind", and the launches that matter for that
are the recent ones: the oldest copy is the one whose live records were long
ago superseded, and the newest is the only one a resume's baselines could still
be read out of. The **newest is never evicted**, at any bound.
"""

SELFPLAY_KIND: Final[str] = "selfplay"
"""The wire tag every line carries.

Present even though the file is single-kind: it makes a line self-describing
if it is ever grepped out of context, and it is what a future
:mod:`komugi.telemetry.journal` decoder would key on if these records are one
day merged into the root journal.
"""

SELFPLAY_SCHEMA_VERSION: Final[int] = 3
"""Stamped on every line, so a format change is detectable rather than
silently misread. A future version reads as an undecodable line, not as a
record with missing fields.

``2`` since KOM-150 added the four game-shape fields (``draws``,
``turn_cap_games``, ``mean_plies``, ``max_plies``); ``3`` since KOM-156b added
the three draft fields (``median_placements_w``, ``median_placements_b``,
``early_shu_games``)."""

GAME_SHAPE_SINCE: Final[int] = 2
"""The first version whose lines carry the game shape (KOM-150)."""

DRAFT_SINCE: Final[int] = 3
"""The first version whose lines carry the draft telemetry (KOM-156b).

The two ``*_SINCE`` constants are what :func:`decode_selfplay_line` decides
each *group* of fields on, rather than a comparison against
:data:`SELFPLAY_SCHEMA_VERSION`. A version bump then leaves v2's game shape
decoding as a measurement instead of quietly demoting it to a sentinel the day
the head version moves past it -- which is what a single ``>= current`` test
does, and it would have turned campaign 2's whole draw-rate history into
``NaN`` at this very bump.
"""

SELFPLAY_READABLE_VERSIONS: Final[frozenset[int]] = frozenset({1, 2, 3})
"""Every version :func:`decode_selfplay_line` accepts.

Wider than :data:`SELFPLAY_SCHEMA_VERSION`, and every extra entry is a
campaign: v1 is what campaign 1 wrote for its whole life and v2 what campaign 2
wrote for its fifteen iterations, and a bump that made those lines unreadable
would delete the history the file exists to keep -- in the same act that added
the fields for reading it better. The fields a line predates decode as the
not-measured sentinels; see :data:`NOT_MEASURED`.
"""

NOT_MEASURED: Final[int] = -1
"""What an integer field holds on a record decoded from a line that predates it.

Negative on purpose: every count in this record is otherwise ``>= 0``, so the
sentinel is outside the domain and cannot be mistaken for a measurement.
:attr:`SelfPlayTelemetry.mean_plies` and the two placement medians use ``NaN``
for the same reason -- it is the one float value that compares ``False``
against every threshold, so an unmeasured mean cannot silently satisfy a
draw-rate gate and an unmeasured median cannot satisfy a draft-minimum ramp.
"""

_JOURNAL_MODE: Final[int] = 0o644


def selfplay_journal_path(root: str | os.PathLike[str]) -> Path:
    """Return the append-only self-play telemetry file under `root`."""
    return Path(root) / SELFPLAY_JOURNAL_NAME


def telemetry_archive_dir(root: str | os.PathLike[str]) -> Path:
    """Return the directory a relaunch copies `root`'s existing telemetry into."""
    return Path(root) / TELEMETRY_ARCHIVE_DIRNAME


def archive_telemetry(
    root: str | os.PathLike[str], logs: Iterable[str | os.PathLike[str]]
) -> tuple[Path, ...]:
    """Copy every existing log in `logs` into `root`'s archive, and say where.

    Called once at campaign start, before a single byte is appended. v1 lost
    its telemetry **twice** to the same mechanism -- a relaunch opened the run
    log for writing and the previous run's history went with it -- and
    :mod:`komugi.telemetry.archive`'s answer is that the only supported route
    to an empty log runs through :func:`~komugi.telemetry.archive.guard_truncation`,
    which archives first and hands back a
    :class:`~komugi.telemetry.archive.TruncationToken` as the evidence.

    **The token is deliberately never spent.** This campaign's logs are
    append-only by construction and a *resumed* campaign reads its own history
    back out of them -- the loss-spike baseline, the previous BatchNorm digest,
    the pre-registered constants stamped at iteration 0 and the whole
    fingerprint series are all recovered from ``journal.jsonl`` -- so
    truncating here would not merely lose an audit trail, it would silently
    disarm every gate that counts across a restart. What a relaunch needs is
    the *copy*: the archived file is a byte-for-byte snapshot of the log as the
    previous launch left it, which is what makes "launch two did not destroy
    launch one's bytes" checkable after the fact even if a later defect does
    start truncating. Holding the token and declining to use it is the API's
    own documented shape ("simply hold the token as evidence"), not a
    workaround.

    A log that does not exist is skipped: a fresh root has no history to save,
    and that is not an error.

    The copies of one log are **bounded** at :data:`ARCHIVE_RETAINED`, oldest
    evicted first (:func:`_evict_stale_copies`). An archive that grows without
    limit is one restart loop away from filling the disk the live journal is
    still being appended to.

    A destination that already exists is skipped too, and this is the
    interesting case rather than a swallowed failure. The archive name is
    derived from the log's **own mtime**, so a collision means *these exact
    bytes at this exact mtime* are already archived -- the previous launch
    appended nothing, or this is a second start against an untouched root --
    and the history is therefore already preserved. Overwriting it would
    destroy the very copy this function exists to make, and
    :func:`~komugi.telemetry.archive.archive_before_truncate` refuses at
    ``O_EXCL`` rather than racing a check.

    Returns:
        The archived copies' paths, in the order `logs` named them. Empty on a
        fresh root, which is what a first launch looks like.
    """
    archive_dir = telemetry_archive_dir(root)
    archived: list[Path] = []
    for log in logs:
        path = Path(log)
        if not path.is_file():
            continue
        try:
            token = guard_truncation(path, archive_dir)
        except FileExistsError:
            # These exact bytes at this exact mtime are already saved (see
            # above). Every *other* failure -- ENOSPC, EACCES, EROFS -- is
            # loud, deliberately: a start that could not save the previous
            # launch's history must not proceed to append to it, which is v1's
            # failure class exactly.
            continue
        archived.append(token.archive_path)
        _evict_stale_copies(archive_dir, path, ARCHIVE_RETAINED)
    return tuple(archived)


def _evict_stale_copies(archive_dir: Path, log: Path, retain: int) -> None:
    """Keep at most `retain` archived copies of `log`, deleting the oldest.

    The copies of one log are exactly the files named ``<log name>.<mtime_ns>``
    (:func:`~komugi.telemetry.archive.archive_name`, fixed-width), so their
    lexicographic order **is** their chronological order and the tail of the
    sort is the newest. Nothing else in the directory is touched: another log's
    copies carry another stem, and a file that does not match the shape is not
    something this function put there.

    Deletion, not compression or hardlinking: the copies are of the same
    append-only log, so a kept copy is a strict prefix of every later one --
    what is evicted is a *prefix of history that a newer copy still holds*, not
    a launch's records. The newest copy is never a candidate (a bound below one
    is clamped to one), because it is the one a resume's baselines and pinned
    constants could still be read out of if the live log were ever lost.
    """
    prefix = f"{log.name}."
    copies = sorted(
        entry
        for entry in archive_dir.iterdir()
        if entry.is_file()
        and entry.name.startswith(prefix)
        and entry.name[len(prefix) :].isdigit()
    )
    for stale in copies[: -max(1, retain)]:
        stale.unlink(missing_ok=True)


@dataclass(frozen=True, slots=True)
class SelfPlayTelemetry:
    """One iteration's self-play, as measured by the orchestrator.

    Every field is a *measurement*, never the request: ``games`` and ``rows``
    come from :class:`~komugi.selfplay.orchestrate.RunSummary`, which counts
    cleanly exited workers' games and the rows actually readable in the shard
    directories, and ``restarts`` is why the first can legitimately exceed
    ``workers * games_per_worker``.

    Attributes:
        iteration: Which campaign iteration this was.
        games: ``RunSummary.games_completed``.
        rows: ``RunSummary.rows_written``.
        restarts: ``RunSummary.restarts`` -- whole-shard restarts. A campaign
            whose throughput fell off a cliff is usually a campaign that is
            silently restarting, and this is the only place that number
            survives the process.
        shard_dirs: How many directories ``RunSummary.shard_dirs`` named --
            the count, not the paths, because the paths are derivable from
            ``root/iter_<i>/shards`` and the count is the part that is not
            (a restart adds an incarnation's directory).

            **The baseline is ``workers * games_in_flight``, not ``workers``**
            (KOM-040 b1, review F5): an interleaved worker runs one lane per
            game in flight and each lane owns its own directory, so a healthy
            2-lane run reports twice the directories a 1-lane run does. Only a
            count *above* that product implies restart incarnations, and the
            excess -- not the total -- is what an operator reads as restarts.
            ``restarts`` above says the same thing directly and is the field to
            trust; this one is the corroborating measurement.
        wall_s: Seconds spent inside
            :func:`~komugi.selfplay.orchestrate.run_selfplay`. The
            self-play/train time split is the first question asked of a
            campaign that is slower than it should be, and the training
            record cannot answer it.
        draws: How many of ``games`` the engine adjudicated as draws, the
            engine's ply turn cap included
            (:attr:`komugi.campaign.game_stats.GameStats.draws`). **The field
            the previous architecture died without**: it spiralled into
            all-draws play, and the only record of the spiral was the operator
            noticing. :data:`NOT_MEASURED` on a v1 line.
        turn_cap_games: How many games reached the turn cap
            (:attr:`komugi.campaign.game_stats.GameStats.turn_cap_games`).
            Carried beside ``draws`` because the two together are the
            diagnosis: draws that are all turn caps say the players shuffled
            to the cap, draws that are not say they agreed to a rule.
            :data:`NOT_MEASURED` on a v1 line.
        mean_plies: Mean game length in plies. ``NaN`` on a v1 line -- see
            :data:`NOT_MEASURED`.
        max_plies: The longest game's length in plies, which is how a reader
            tells "games are getting long" from "one game hit the cap".
            :data:`NOT_MEASURED` on a v1 line.
        median_placements_w: The median number of pieces white placed in the
            draft (:attr:`komugi.campaign.game_stats.GameStats.median_placements_w`,
            whose definition -- the 終 row included, booked games reading low
            -- is the one that holds here). ``NaN`` on a v1 or v2 line.
        median_placements_b: The same for black. ``NaN`` on a v1 or v2 line.
        early_shu_games: How many games had a side end its draft on fewer than
            :data:`~komugi.campaign.game_stats.EARLY_SHU_PLACEMENTS`
            placements. **The field campaign 2 died without**: it fell into
            v1's draft attractor -- a median of four placements and 終 by
            iteration 14 -- and the only record of it was an advisor censusing
            the shards by hand. :data:`NOT_MEASURED` on a v1 or v2 line.

    The game-shape four and the draft three have **no defaults**,
    deliberately: they are measured per iteration by
    :func:`komugi.campaign.game_stats.game_stats`, and a default would let a
    construction site that forgot to measure them write a plausible zero to the
    file forever. A forgotten site is a ``TypeError`` at the call, which is the
    loudest a missing measurement can be.

    Raises:
        ValueError: On a negative count or a non-finite/negative wall time.
            Validated here, where the record is built, rather than where it is
            finally plotted -- the discipline of
            :mod:`komugi.telemetry.records`. The game-shape fields additionally
            accept their not-measured sentinels, and nothing else negative.
    """

    iteration: int
    games: int
    rows: int
    restarts: int
    shard_dirs: int
    wall_s: float
    draws: int
    turn_cap_games: int
    mean_plies: float
    max_plies: int
    median_placements_w: float
    median_placements_b: float
    early_shu_games: int

    def __post_init__(self) -> None:
        """Reject a record that could not describe a real self-play run."""
        for name, count in (
            ("iteration", self.iteration),
            ("games", self.games),
            ("rows", self.rows),
            ("restarts", self.restarts),
            ("shard_dirs", self.shard_dirs),
        ):
            if isinstance(count, bool) or not isinstance(count, int):
                raise ValueError(
                    f"SelfPlayTelemetry.{name} must be an integer, got {count!r}"
                )
            if count < 0:
                raise ValueError(
                    f"SelfPlayTelemetry.{name} must be >= 0, got {count!r}"
                )
        for name, count in (
            ("draws", self.draws),
            ("turn_cap_games", self.turn_cap_games),
            ("max_plies", self.max_plies),
            ("early_shu_games", self.early_shu_games),
        ):
            if isinstance(count, bool) or not isinstance(count, int):
                raise ValueError(
                    f"SelfPlayTelemetry.{name} must be an integer, got {count!r}"
                )
            if count < 0 and count != NOT_MEASURED:
                # The sentinel is the ONE negative value these fields accept:
                # a v1 line carries no measurement, and every other negative is
                # a counting bug that must not reach the file.
                raise ValueError(
                    f"SelfPlayTelemetry.{name} must be >= 0 or the "
                    f"not-measured sentinel {NOT_MEASURED}, got {count!r}"
                )
        wall = float(self.wall_s)
        if not math.isfinite(wall):
            # NaN would round-trip as the non-standard `NaN` token if the
            # encoder let it through, and would compare False against every
            # threshold a future gate sets.
            raise ValueError(f"SelfPlayTelemetry.wall_s must be finite, got {wall!r}")
        if wall < 0.0:
            raise ValueError(f"SelfPlayTelemetry.wall_s must be >= 0, got {wall!r}")
        object.__setattr__(self, "wall_s", wall)
        mean = float(self.mean_plies)
        if math.isinf(mean) or mean < 0.0:
            # NaN is admitted here and only here: it is `mean_plies`'s
            # not-measured sentinel. An infinity is not a mean of anything.
            raise ValueError(
                f"SelfPlayTelemetry.mean_plies must be >= 0 or NaN, got {mean!r}"
            )
        object.__setattr__(self, "mean_plies", mean)
        for name in ("median_placements_w", "median_placements_b"):
            median = float(getattr(self, name))
            if math.isinf(median) or median < 0.0:
                # NaN is the not-measured sentinel here too; a side cannot have
                # placed a negative or infinite number of pieces.
                raise ValueError(
                    f"SelfPlayTelemetry.{name} must be >= 0 or NaN, got {median!r}"
                )
            object.__setattr__(self, name, median)

    @property
    def measured(self) -> bool:
        """Return whether this record's game shape was measured at all.

        ``False`` exactly on a record decoded from a v1 line. An auditor
        plotting a draw rate across a campaign's whole history asks this
        instead of comparing against zero, and :func:`encode_selfplay` asks it
        to refuse writing a sentinel back out as if it were a measurement.

        Deliberately **not** widened to cover the draft fields: a v2 line's
        draw rate is a measurement and must keep reading as one. The draft has
        its own flag, :attr:`draft_measured`.
        """
        return not math.isnan(self.mean_plies)

    @property
    def draft_measured(self) -> bool:
        """Return whether this record's draft telemetry was measured at all.

        ``False`` on a record decoded from a v1 **or** a v2 line -- campaign 1
        and campaign 2 both played their whole lives without it. The ramp gate
        asks this before reading :attr:`early_shu_rate`, because "no game
        ended its draft early" and "nobody counted" are the two readings the
        campaign 2 post-mortem had to tell apart by hand.
        """
        return not math.isnan(self.median_placements_w)

    @property
    def draw_rate(self) -> float:
        """Return ``draws / games``, or ``NaN`` if either was never measured.

        ``0.0`` for a measured iteration that played no games, which is the
        same convention :attr:`komugi.campaign.game_stats.GameStats.draw_rate`
        uses and for the same reason: an iteration that played nothing has no
        draw rate, and the ``games`` beside it says so.
        """
        if not self.measured or self.draws == NOT_MEASURED:
            return math.nan
        return 0.0 if self.games == 0 else self.draws / self.games

    @property
    def early_shu_rate(self) -> float:
        """Return ``early_shu_games / games``, or ``NaN`` if never measured.

        **The ramp gate's number.** A draft-minimum curriculum is relaxed a
        rung only while this stays low (v1 held it under ``0.3``), so an
        unmeasured iteration must not be able to satisfy the comparison: the
        ``NaN`` is what makes ``rate < 0.3`` false on a v1 or v2 line instead
        of trivially true. ``0.0`` for a measured iteration that played no
        games, as :attr:`draw_rate` does.
        """
        if not self.draft_measured or self.early_shu_games == NOT_MEASURED:
            return math.nan
        return 0.0 if self.games == 0 else self.early_shu_games / self.games


@dataclass(frozen=True)
class SelfPlayRead:
    """Records read out of a telemetry file, and the lines that were not.

    The unreadable count travels with the records for the reason
    :class:`~komugi.telemetry.journal.JournalRead` gives: "two iterations" and
    "two iterations and a line we could not parse" are different claims about
    a dead campaign, and the second must not be reachable by ignoring a return
    value.
    """

    records: tuple[SelfPlayTelemetry, ...]
    skipped: int

    def __len__(self) -> int:
        """Return the number of records decoded."""
        return len(self.records)

    @property
    def ok(self) -> bool:
        """Return whether every non-blank line decoded."""
        return self.skipped == 0


def encode_selfplay(record: SelfPlayTelemetry) -> str:
    """Return the one-line JSON encoding of `record` (no trailing newline).

    ``allow_nan=False`` is the second of two locks -- the record already
    refuses a non-finite wall time -- because the non-standard ``NaN`` token
    would make the file unparseable by anything but Python.

    Raises:
        ValueError: If `record` carries the not-measured sentinels, i.e. if it
            was decoded from a v1 line (no game shape) or a v2 line (no draft
            telemetry). Such a record is *readable* history, not a
            measurement, and writing it back out as a v3 line would claim a
            campaign measured something it never did. Refused here, naming the
            reason, rather than emerging as a ``NaN`` token failure from
            inside :func:`json.dumps`.
    """
    if not record.measured:
        raise ValueError(
            f"refusing to encode a not-measured {SELFPLAY_KIND} record "
            f"(iteration {record.iteration}): it was decoded from a v1 line, "
            f"whose game shape was never measured, and a v"
            f"{SELFPLAY_SCHEMA_VERSION} line asserts that it was"
        )
    if not record.draft_measured:
        raise ValueError(
            f"refusing to encode a not-measured {SELFPLAY_KIND} record "
            f"(iteration {record.iteration}): it was decoded from a v"
            f"{GAME_SHAPE_SINCE} line, whose draft was never measured, and a v"
            f"{SELFPLAY_SCHEMA_VERSION} line asserts that it was"
        )
    payload: dict[str, object] = {
        "kind": SELFPLAY_KIND,
        "v": SELFPLAY_SCHEMA_VERSION,
        "iteration": record.iteration,
        "games": record.games,
        "rows": record.rows,
        "restarts": record.restarts,
        "shard_dirs": record.shard_dirs,
        "wall_s": record.wall_s,
        "draws": record.draws,
        "turn_cap_games": record.turn_cap_games,
        "mean_plies": record.mean_plies,
        "max_plies": record.max_plies,
        "median_placements_w": record.median_placements_w,
        "median_placements_b": record.median_placements_b,
        "early_shu_games": record.early_shu_games,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _field(payload: Mapping[str, object], key: str) -> object:
    """Return ``payload[key]``, as a :class:`JournalDecodeError` if absent."""
    if key not in payload:
        raise JournalDecodeError(f"selfplay line is missing field {key!r}")
    return payload[key]


def _as_int(payload: Mapping[str, object], key: str) -> int:
    """Return an int field, rejecting bools and JSON floats."""
    value = _field(payload, key)
    # bool is an int subclass; `true` where a count belongs is corruption.
    if isinstance(value, bool) or not isinstance(value, int):
        raise JournalDecodeError(f"field {key!r} must be an integer, got {value!r}")
    return value


def _as_float(payload: Mapping[str, object], key: str) -> float:
    """Return a float field, accepting a JSON integer, rejecting bools."""
    value = _field(payload, key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise JournalDecodeError(f"field {key!r} must be a number, got {value!r}")
    return float(value)


def _as_str(payload: Mapping[str, object], key: str) -> str:
    """Return a string field."""
    value = _field(payload, key)
    if not isinstance(value, str):
        raise JournalDecodeError(f"field {key!r} must be a string, got {value!r}")
    return value


def decode_selfplay_line(line: str) -> SelfPlayTelemetry:
    """Decode one line of a telemetry file into a record.

    Raises :class:`~komugi.telemetry.journal.JournalDecodeError` -- the same
    exception the root journal raises, so a caller reading both files handles
    one error type -- for anything unusable: truncated JSON, a foreign
    ``kind``, a schema version outside :data:`SELFPLAY_READABLE_VERSIONS`, a
    field of the wrong type, or a value the record's own validation rejects.

    **A v1 line decodes** (KOM-150) **and so does a v2 line** (KOM-156b):
    every field a line carried is read as it always was, and the fields it
    predates come back as the not-measured sentinels. That is the whole reason
    the version widens rather than moves: campaign 1's ``selfplay.jsonl`` is v1
    from end to end and campaign 2's is v2, and they are the history the
    draw-spiral and draft-attractor questions are asked of.
    """
    try:
        parsed: object = json.loads(line)
    except ValueError as exc:  # json.JSONDecodeError
        raise JournalDecodeError(f"line is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise JournalDecodeError(f"line is not a JSON object: {parsed!r}")

    version = _as_int(parsed, "v")
    if version not in SELFPLAY_READABLE_VERSIONS:
        raise JournalDecodeError(
            f"unsupported selfplay schema version {version}, expected one of "
            f"{sorted(SELFPLAY_READABLE_VERSIONS)}"
        )
    kind = _as_str(parsed, "kind")
    if kind != SELFPLAY_KIND:
        # This file is single-kind by construction (see the module docstring),
        # so a foreign kind here is a file that is not what it claims to be.
        raise JournalDecodeError(
            f"unknown record kind {kind!r}, expected {SELFPLAY_KIND!r}"
        )
    measured = version >= GAME_SHAPE_SINCE
    drafted = version >= DRAFT_SINCE
    try:
        return SelfPlayTelemetry(
            iteration=_as_int(parsed, "iteration"),
            games=_as_int(parsed, "games"),
            rows=_as_int(parsed, "rows"),
            restarts=_as_int(parsed, "restarts"),
            shard_dirs=_as_int(parsed, "shard_dirs"),
            wall_s=_as_float(parsed, "wall_s"),
            # A v1 line is not missing these fields, it predates them: the
            # sentinels say "never measured", which is a different claim from
            # any number this decoder could invent.
            draws=_as_int(parsed, "draws") if measured else NOT_MEASURED,
            turn_cap_games=(
                _as_int(parsed, "turn_cap_games") if measured else NOT_MEASURED
            ),
            mean_plies=_as_float(parsed, "mean_plies") if measured else math.nan,
            max_plies=_as_int(parsed, "max_plies") if measured else NOT_MEASURED,
            # The same tolerance one version later, for the same reason: a v2
            # line is not missing the draft fields, it predates them.
            median_placements_w=(
                _as_float(parsed, "median_placements_w") if drafted else math.nan
            ),
            median_placements_b=(
                _as_float(parsed, "median_placements_b") if drafted else math.nan
            ),
            early_shu_games=(
                _as_int(parsed, "early_shu_games") if drafted else NOT_MEASURED
            ),
        )
    except ValueError as exc:
        # Covers the record's own __post_init__ as well as the field helpers:
        # a negative row count on disk is as unusable as a truncated line.
        raise JournalDecodeError(f"invalid {SELFPLAY_KIND} record: {exc}") from exc


def append_selfplay(
    path: str | os.PathLike[str], record: SelfPlayTelemetry
) -> None:
    """Append `record` to the telemetry file at `path`, durably.

    Creates the file (but not its parent directory) if it does not exist, and
    returns only once the bytes are on the platter. This is
    :func:`komugi.telemetry.journal.write_record`'s discipline, restated for
    this record type rather than reused, because that function's ``Record``
    union is closed over :mod:`komugi.telemetry`'s two records and widening it
    is outside this module's reach. The three properties that matter are the
    same and are the reason the audit claim holds at all: ``O_APPEND`` (the
    kernel owns the offset, so nothing lands on top of an earlier record and
    no mode here can shorten the file), the short-write loop (a partial
    ``os.write`` still resumes at the end of file), and the ``fsync`` before
    return (one per iteration -- unmeasurable next to an iteration of
    self-play, and the reason a hard kill still leaves the finished
    iterations' numbers behind).
    """
    payload = (encode_selfplay(record) + "\n").encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, _JOURNAL_MODE)
    try:
        written = 0
        while written < len(payload):
            written += os.write(fd, payload[written:])
        os.fsync(fd)
    finally:
        os.close(fd)


def read_selfplay(
    path: str | os.PathLike[str], *, missing_ok: bool = False
) -> SelfPlayRead:
    """Read every self-play record from the telemetry file at `path`.

    Lines that cannot be decoded are skipped and counted in
    :attr:`SelfPlayRead.skipped` rather than raised: a campaign killed between
    the first byte and the newline leaves a partial trailing line, and losing
    the whole run's history to it would defeat the purpose of writing it down.

    With ``missing_ok=True`` a file that does not exist reads as empty, for
    the case where an audit runs against a campaign that died before its first
    iteration finished self-play.
    """
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        if missing_ok:
            return SelfPlayRead(records=(), skipped=0)
        raise

    records: list[SelfPlayTelemetry] = []
    skipped = 0
    for line in text.split("\n"):
        if not line.strip():
            # The trailing empty string after the final newline, and any blank
            # padding: absence of a record, not a damaged one.
            continue
        try:
            records.append(decode_selfplay_line(line))
        except JournalDecodeError:
            skipped += 1
    return SelfPlayRead(records=tuple(records), skipped=skipped)
