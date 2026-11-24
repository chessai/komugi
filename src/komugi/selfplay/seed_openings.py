"""Seeded self-play openings: the conversion curriculum's file and its draw.

KOM-254, designed at ``docs/design/kom-254-conversion-curriculum.md`` and
amended by ``docs/reviews/kom-254-design-review.md``. The mechanism is KOM-014's
draft book with one thing changed -- the prefix is 52-384 plies of a *harvested*
game rather than 4-12 plies of a mined opening -- so everything structurally
hard is already paid for: a per-game decision that is a pure function of
``(run_seed, worker_id, incarnation, game_index)``, a replay that records no
row, rows that begin at the cut ply, and loud failure on a prefix this engine
cannot replay.

What this module owns is the *file* and the *draw*. The replay is the worker's
(:meth:`komugi.selfplay.worker.SelfPlayWorker._replay_actions`, shared with the
book path), and the harvester that writes the file is advisor-side and out of
this ticket's scope.

Why the curriculum exists (KOM-253): positions Komugi's value head scores
``>= .85`` for 20+ plies occur in every external-anchor game but only 4-12% of self-play
games, and when force is depleted they *erode* rather than convert -- a
6400-playout search still calls them won and the policy cannot finish them. The
regime is under-trained, so it is trained on (KataGo's ``startPoses``
precedent).

The four guards a seed file is refused on are the book's own (``load_book``,
``book.py:1163-1227``) plus one the book does not need: a book entry is mined
from ply 0 by construction, while a *harvested* prefix comes from a window ~10%
of which is itself booked -- and a booked game's rows begin at ply ``L``, so an
action list taken from one replays legally from ply 0 **into a different
position** (design review F2). :data:`SeedEntry.start_ply` is that guarantee,
and the loader refuses anything but ``0``.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

from komugi.engine.observation import MAX_TURNS
from komugi.selfplay.book import running_gungi_mode
from komugi.selfplay.records import game_seed, num_actions

SEED_OPENINGS_VERSION: Final[int] = 1
"""The seed-file format this reader reads (the book's ``book_version`` rule).

Checked **first**, before any other key, because it is the only field that says
whether the rest of the file means what this reader thinks it means."""

SEED_OPENINGS_DOMAIN: Final[int] = 0x9254_0001
"""Spawn-key domain for a worker's per-game **seed-opening draw** (KOM-254 D4).

Composed exactly as the book and draft-threshold domains are,
``(*game_seed.spawn_key, SEED_OPENINGS_DOMAIN)``, and disjoint from every other
stream a game owns: from :func:`~komugi.selfplay.records.move_seed`'s
``(*key, ply)`` because no ply the engine can reach comes near ``0x9254_0001``
(it adjudicates at :data:`~komugi.engine.observation.MAX_TURNS`), and from
:data:`~komugi.selfplay.records.BOOK_BUILD_DOMAIN`,
:data:`~komugi.selfplay.records.BOOK_SELECT_DOMAIN`,
:data:`~komugi.selfplay.records.DRAFT_THRESHOLD_DOMAIN` and the campaign probe's
``0x9022_0001``/``0x9022_0002`` **by value**.

Below ``2**32``, which the whole scheme rests on: numpy splits a larger key
element into two 32-bit words, so a domain at or above it would hash exactly
like ``(worker_id, incarnation, game_index, DOMAIN)`` and the disjointness
argument would be false (:data:`~komugi.selfplay.records.DRAFT_THRESHOLD_DOMAIN`
spells this out). That disjointness is what makes the R1 claim true: at
``seed_openings_prob = 0.0`` a run with a seed file draws the identical
full/fast sequence and the identical per-move Gumbel vectors as a run without
one, so the mechanism lands **inert**.
"""

CONVERSION_HORIZON_PLIES: Final[int] = MAX_TURNS // 4
"""Plies a seeded game must have left to play after its cut (= 128 today).

**Derived** from :data:`~komugi.engine.observation.MAX_TURNS` rather than
written as a literal, the way :data:`~komugi.selfplay.records.MLH_SCALE` and
:data:`~komugi.selfplay.book.TARGET_SHARD_BUDGET` are -- KOM-165 already moved
that cap once (256 → 512), and a literal here would have silently kept a
256-ply run's horizon.

Sized on KOM-253's measurements, which is the whole point of the number: eroded
plateaus fall only after ~71 plies, converted episodes plateau for ~27 plies and
the external-anchor leg's wins convert ~24-27 plies after the first ``.85`` reading. A cut that
leaves less than that is a game whose result is decided by the engine's turn
cap rather than by the position -- and a capped game's rows carry KOM-245's
material-cap label ``(p, 1-p, 0)``, which is precisely the "material lead ⇒
won" label this curriculum exists to contradict (design review F4).
"""

MIN_SEED_ENTRIES: Final[int] = 32
"""Fewest entries a seed file may hold, mirroring the book's
:data:`~komugi.selfplay.book.MIN_ENTRIES` and for the same reason with more
force: a 3-entry book would send every booked game of a generation down one of
three lines, and a seed prefix is 4-30x longer, hence far more specific. At the
permitted ceiling ``seed_openings_prob = 0.5`` a one-entry file would send half
of self-play into a single position (design review F8)."""

MAX_SEED_ENTRIES: Final[int] = 4096
"""Most entries a seed file may hold (D3). Uniform selection over this many
entries needs no weighting, which v1 therefore does not have."""

_MAX_DRAFT_THRESHOLD: Final[int] = 25
"""Highest threshold a draft can reach -- a hand (``komugi.config``'s own
``_MAX_MIN_PLACEMENTS``). Restated rather than imported for the reason
:mod:`komugi.selfplay.records` restates it in turn: it is that module's
*private* name, and importing a private across modules couples this loader to
seed-discipline internals that are documented as restatable. The tripwire that
keeps the two numbers equal is a test, not an import."""

MIN_CUT: Final[int] = 2 * _MAX_DRAFT_THRESHOLD + 2
"""Lowest cut ply a seed entry may name (= 52): **no cut lands mid-draft**.

A full draft is at most one hand per side plus each side's 終
(:data:`_MAX_DRAFT_THRESHOLD` placements, twice, plus two finishes), so a cut
at or above this is past both drafts by construction. Derived rather than
written as ``52`` because the hand is :mod:`komugi.config`'s
``_MAX_MIN_PLACEMENTS``.

What it buys (design review F10): the prefix plies were played under the
*source* run's DRAFTMIN thresholds, not this game's drawn ``(N_w, N_b)``, so a
cut inside a draft would leave this run's threshold silently vacuous for that
game. Above :data:`MIN_CUT` a seeded game's draft is **entirely** the source's
and this run's thresholds are vacuous for it *by construction* -- stated, not
hidden.
"""


@dataclass(frozen=True, slots=True)
class SeedEntry:
    """One harvested prefix, and the provenance that makes it auditable.

    ``actions`` replays in order from a bare
    :class:`~komugi.engine.game.Game`, exactly as a
    :class:`~komugi.selfplay.book.BookEntry`'s does -- a position is not a
    portable artefact and an action list is.

    Args:
        id: Non-blank and unique within the file; it is what the worker's log
            line and every failure message name.
        actions: The source game's recorded actions, from its ply ``0``.
        cut_min: Lowest ply the seeded game may start searching from.
        cut_max: Highest such ply, inclusive, and **strictly below**
            ``len(actions)``: the loader refuses equality, because the last
            recorded action of a source game is the move that ended it. The cut
            is drawn uniformly in ``[cut_min, cut_max]`` per game, so one entry
            covers a *band* of positions rather than a single one.
        start_ply: Which ply of the source game ``actions[0]`` was played at.
            Required, and required to be ``0``: see the module docstring.
        source_iteration: The campaign iteration the prefix was harvested from.
        source_game_id: The ``game_id`` of the source game in that iteration.
        cls: The harvester's class string for the position (e.g. its plateau
            band). Recorded, never read here -- it is the advisor's label for a
            census, and this module's business is only that it is present.
    """

    id: str
    actions: tuple[int, ...]
    cut_min: int
    cut_max: int
    start_ply: int
    source_iteration: int
    source_game_id: int
    cls: str


@dataclass(frozen=True, slots=True)
class SeedChoice:
    """What the per-game draw decided (the book's :class:`BookChoice` shape).

    Args:
        seeded: Whether this game starts from a harvested prefix.
        entry_index: Index into the loaded entries, or ``None`` when the game
            is unseeded -- ``None`` rather than ``-1``, because an unseeded
            game has no entry at all and a sentinel index reads back as "the
            last entry".
        cut: The ply the seeded game starts searching from, or ``None``.
    """

    seeded: bool
    entry_index: int | None
    cut: int | None


def _entry_from_json(path: str | os.PathLike[str], payload: Any) -> SeedEntry:
    """Return one :class:`SeedEntry`, naming the file on a missing key."""
    if not isinstance(payload, dict):
        raise ValueError(
            f"seed openings {path} holds an entry that is not a JSON object, "
            f"got {type(payload)}"
        )
    try:
        return SeedEntry(
            id=str(payload["id"]),
            actions=tuple(int(action) for action in payload["actions"]),
            cut_min=int(payload["cut_min"]),
            cut_max=int(payload["cut_max"]),
            start_ply=int(payload["start_ply"]),
            source_iteration=int(payload["source_iteration"]),
            source_game_id=int(payload["source_game_id"]),
            cls=str(payload["class"]),
        )
    except KeyError as exc:
        raise ValueError(
            f"seed openings {path} has an entry missing key {exc}: "
            f"{payload.get('id', '<no id>')!r}"
        ) from exc


def load_seed_openings(
    path: str | os.PathLike[str],
) -> tuple[tuple[SeedEntry, ...], str]:
    """Parse a seed file, refusing one this process cannot replay.

    The refusals are :func:`~komugi.selfplay.book.load_book`'s, argument for
    argument, plus the two a harvested prefix needs and a mined one does not:

    * ``version`` -- checked **first**, before any other key is read, because
      it is the only field that says whether the rest of the file means what
      this reader thinks it means.
    * ``action_size`` -- the guard that actually protects a prefix: an action
      id means nothing across an encoding change, so a stale file would replay
      a plausible wrong line rather than fail.
    * ``gungi_mode`` -- a prefix is replayed from a bare
      :class:`~komugi.engine.game.Game`, and ``GUNGI_MODE`` decides which
      position that is.
    * ``start_ply != 0`` -- the harvested-prefix guard (design review F2). Most
      of a draft's placements are legal from most positions, so a prefix lifted
      out of a *booked* source game replays legally from ply 0 into a different
      position and the curriculum trains on plateaus that never occurred. The
      worker's legality check provably cannot catch it; only the field can.
    * the entry bounds ``MIN_CUT <= cut_min <= cut_max <= MAX_TURNS -
      CONVERSION_HORIZON_PLIES`` **and** ``cut_max < len(actions)``, each
      violation naming its numbers -- a cut below :data:`MIN_CUT` lands
      mid-draft, one above the conversion horizon is a game the turn cap
      decides, and one at ``len(actions)`` is a cut the replay is guaranteed to
      refuse (impl review F2): ``actions[cut]`` existing is what proves the
      position at the cut was non-terminal in the source game, so at equality
      the top of the band is terminal by construction and kills a worker at
      draw time rather than at load.

    ``plane_map_version`` is deliberately **not** required: a prefix is action
    ids and a seeded game's rows are written fresh by this process's encoder,
    so there is no cross-generation observation to protect (unlike a book,
    whose campaign-side check is the only place a stale plane map is named).

    Args:
        path: The seed file, as
            :attr:`komugi.config.SelfPlayConfig.seed_openings_path` names it.

    Returns:
        ``(entries, sha256)`` -- the entries in file order, and the SHA-256 of
        the **exact file bytes**. The hash is the answer to design review F7:
        the file is refreshed per harvest window, so "which entry did game
        783/12/0/44 use" is reconstructible only against the file the worker
        actually read, and the worker's per-game log line carries this hash's
        prefix.

    Raises:
        ValueError: On unreadable JSON, a missing key, a format version this
            reader does not read, a mismatched action space or start mode, an
            entry that does not start at ply 0, an entry count outside
            ``[MIN_SEED_ENTRIES, MAX_SEED_ENTRIES]``, a duplicate or blank id,
            an empty action list, an action id outside the action space, or a
            cut band that violates its own bounds.
    """
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    try:
        payload: Any = json.loads(raw.decode("utf-8"))
    except ValueError as exc:
        raise ValueError(f"seed openings {path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(
            f"seed openings {path} must be a JSON object, got {type(payload)}"
        )

    try:
        version = int(payload["version"])
    except KeyError as exc:
        raise ValueError(f"seed openings {path} is missing key {exc}") from exc
    if version != SEED_OPENINGS_VERSION:
        raise ValueError(
            f"seed openings {path} declares version {version}, but this "
            f"process reads version {SEED_OPENINGS_VERSION}: the constant is "
            f"bumped exactly when a reader must change, so parsing this file "
            f"under the running version's rules would read one format as if it "
            f"were another"
        )

    try:
        action_size = int(payload["action_size"])
        mode = str(payload["gungi_mode"])
        raw_entries = payload["entries"]
    except KeyError as exc:
        raise ValueError(f"seed openings {path} is missing key {exc}") from exc

    running_actions = num_actions()
    if action_size != running_actions:
        raise ValueError(
            f"seed openings {path} was harvested from an action space of "
            f"{action_size} actions, but this process encodes "
            f"{running_actions}: an action id means nothing across an encoding "
            f"change, so the prefixes cannot be replayed"
        )
    running_mode = running_gungi_mode()
    if mode != running_mode:
        raise ValueError(
            f"seed openings {path} was harvested under GUNGI_MODE={mode!r}, "
            f"but this process runs {running_mode!r}: a prefix replayed from a "
            f"different start position is a different line, or illegal"
        )

    entries = tuple(_entry_from_json(path, item) for item in raw_entries)
    if not MIN_SEED_ENTRIES <= len(entries) <= MAX_SEED_ENTRIES:
        raise ValueError(
            f"seed openings {path} holds {len(entries)} entries, outside "
            f"[{MIN_SEED_ENTRIES}, {MAX_SEED_ENTRIES}]: the draw is uniform "
            f"over the file, so too few entries send a whole generation's "
            f"seeded games down a handful of lines"
        )

    seen: set[str] = set()
    # The horizon bound is a property of the running engine, not of an entry,
    # so it is computed once here and named in every message that refuses one.
    horizon_bound = MAX_TURNS - CONVERSION_HORIZON_PLIES
    for entry in entries:
        if not entry.id:
            raise ValueError(f"seed openings {path} holds an entry with a blank id")
        if entry.id in seen:
            raise ValueError(
                f"seed openings {path} holds two entries with id {entry.id}: "
                f"an id is what a log line and a census name a prefix by"
            )
        seen.add(entry.id)
        if entry.start_ply != 0:
            raise ValueError(
                f"seed openings {path} entry {entry.id} has start_ply "
                f"{entry.start_ply}, not 0: its actions are "
                f"replayed from a bare Game, so a list that began mid-game "
                f"replays legally into a DIFFERENT position rather than failing"
            )
        if not entry.actions:
            raise ValueError(
                f"seed openings {path} entry {entry.id} carries no actions: an "
                f"empty prefix is an unseeded game wearing an entry's id"
            )
        if any(not 0 <= action < action_size for action in entry.actions):
            raise ValueError(
                f"seed openings {path} entry {entry.id} carries an action id "
                f"outside [0, {action_size})"
            )
        if entry.cut_min < MIN_CUT:
            raise ValueError(
                f"seed openings {path} entry {entry.id} has cut_min "
                f"{entry.cut_min}, below {MIN_CUT}: a cut below that can land "
                f"mid-draft, where the prefix played under the SOURCE run's "
                f"draft thresholds and this run's would be silently vacuous"
            )
        if entry.cut_min > entry.cut_max:
            raise ValueError(
                f"seed openings {path} entry {entry.id} has cut_min "
                f"{entry.cut_min} above cut_max {entry.cut_max}: the cut is "
                f"drawn in [cut_min, cut_max], and an inverted band draws "
                f"nothing"
            )
        if entry.cut_max > horizon_bound:
            raise ValueError(
                f"seed openings {path} entry {entry.id} has cut_max "
                f"{entry.cut_max}, above {horizon_bound} = MAX_TURNS "
                f"({MAX_TURNS}) - CONVERSION_HORIZON_PLIES "
                f"({CONVERSION_HORIZON_PLIES}): a game cut that deep reaches "
                f"the engine's turn cap by construction, and a capped game's "
                f"rows carry the material-cap label this curriculum exists to "
                f"contradict"
            )
        if entry.cut_max >= len(entry.actions):
            raise ValueError(
                f"seed openings {path} entry {entry.id} has cut_max "
                f"{entry.cut_max} but carries {len(entry.actions)} actions: "
                f"cut_max must be strictly below len(actions), because "
                f"actions[cut] existing is the proof that the position at the "
                f"cut was NON-terminal when the source game was played -- at "
                f"equality the top of the band is terminal by construction "
                f"(the last recorded action is the move that ended the game), "
                f"so it kills a worker at draw time, at whichever game happens "
                f"to draw it, and the pre-flight cannot see it because it "
                f"replays entry 0 at cut_min only"
            )
    return entries, digest


def seed_openings_rng(
    run_seed: int, worker_id: int, incarnation: int, game_index: int
) -> np.random.Generator:
    """Return one game's **seed-openings** stream -- disjoint from every other.

    ``SeedSequence(entropy=game_seed(...).entropy, spawn_key=(*spawn_key,
    SEED_OPENINGS_DOMAIN))``, i.e.
    :func:`~komugi.selfplay.book.book_select_rng`'s composition with this
    module's domain where a ply would be. Drawn from the game's own ``rng``
    instead, this draw would shift every subsequent full/fast draw of *every*
    game and the "lands inert at prob 0" claim would be false.

    ``Generator(PCG64(...))`` with the bit generator named rather than
    ``default_rng``, for the reason
    :data:`~komugi.selfplay.book.BIT_GENERATOR` gives: this draw is recomputed
    offline, in another process, by :func:`recompute_seed_choice`.
    """
    seeds = game_seed(run_seed, worker_id, incarnation, game_index)
    return np.random.Generator(
        np.random.PCG64(
            np.random.SeedSequence(
                entropy=seeds.entropy,
                spawn_key=(*seeds.spawn_key, SEED_OPENINGS_DOMAIN),
            )
        )
    )


def _draw(
    rng: np.random.Generator,
    *,
    n_entries: int,
    prob: float,
    bounds: Callable[[int], tuple[int, int]],
) -> SeedChoice:
    """The draw itself, in this order and no other (design review Q2).

    ``random() < prob``; then -- **only if** that came up seeded --
    ``integers(0, n_entries)``; then ``integers(cut_min, cut_max + 1)``,
    inclusive on ``cut_max`` because the band is ``[cut_min, cut_max]``. The
    order is part of the answer, because the audit replays it, and the two
    later draws are conditional so an unseeded game consumes exactly one
    number -- which is what makes the seeded *share* a property of the first
    draw alone.

    ``integers(lo, hi)``, never ``% n``: a modulo over a machine word is biased
    for any ``n`` that is not a power of two. ``cut_min == cut_max == k``
    therefore always yields ``k``.

    `bounds` resolves the drawn entry's cut band. It is a callable rather than
    a pair because the band belongs to an entry that is not known until the
    second draw -- and it is the only difference between the worker's draw
    (:func:`select_seed_entry`, which holds the entries) and the offline one
    (:func:`recompute_seed_choice`, which is handed one entry's band).
    """
    if n_entries < 1:
        raise ValueError(
            f"cannot select from {n_entries} seed entries: an empty file is "
            f"not an unseeded run, and reporting it as one would hide it"
        )
    if not bool(rng.random() < prob):
        return SeedChoice(seeded=False, entry_index=None, cut=None)
    index = int(rng.integers(0, n_entries))
    cut_min, cut_max = bounds(index)
    return SeedChoice(
        seeded=True,
        entry_index=index,
        cut=int(rng.integers(cut_min, cut_max + 1)),
    )


def select_seed_entry(
    run_seed: int,
    worker_id: int,
    incarnation: int,
    game_index: int,
    *,
    entries: Sequence[SeedEntry],
    prob: float,
) -> SeedChoice:
    """Return whether one game is seeded, with which entry, and at which cut.

    The worker's call. A pure function of the four integers that name the game
    plus the file, on the game's own :func:`seed_openings_rng` stream, so the
    answer is reconstructible offline from a shard's ``game_id`` and the file
    alone -- :func:`recompute_seed_choice` is that reconstruction.

    Args:
        run_seed: The seed **the worker was handed** -- in a campaign that is
            :func:`komugi.campaign.loop.iteration_seed` of the campaign's own,
            not the campaign's own.
        worker_id: The worker's index within the run.
        incarnation: The worker's respawn counter.
        game_index: The game's index within the worker's workload.
        entries: The loaded file, in file order.
        prob: :attr:`komugi.config.SelfPlayConfig.seed_openings_prob`.

    Returns:
        The :class:`SeedChoice` this game's stream decided.

    Raises:
        ValueError: If `entries` is empty -- a file with no entry cannot seed a
            game, and returning "unseeded" would hide it behind a perfectly
            ordinary-looking 0% seeding rate (the book's own argument).
    """
    return _draw(
        seed_openings_rng(run_seed, worker_id, incarnation, game_index),
        n_entries=len(entries),
        prob=prob,
        bounds=lambda index: (entries[index].cut_min, entries[index].cut_max),
    )


def recompute_seed_choice(
    run_seed: int,
    worker_id: int,
    incarnation: int,
    game_index: int,
    *,
    n_entries: int,
    prob: float,
    cut_min: int,
    cut_max: int,
) -> tuple[bool, int | None, int | None]:
    """Reproduce one game's seed draw **out of process** (AM6 / design review F6).

    Nothing in a shard records that a game was seeded: the schema is unchanged
    (no new field, no reordering, so window mixing with pre-KOM-254 shards is
    unaffected), so "was game X seeded, from which entry, cut where" is
    answerable only by recomputation -- and by a function that does not need a
    worker, a config or an engine. This is it. It also settles the reading the
    book audit gets wrong: a seeded game presents ``min(ply) = cut >= 52``, so
    :func:`komugi.campaign.book_audit.recompute_choice` would file it under
    *booked* or *truncated*; this function is what says otherwise.

    `cut_min`/`cut_max` are the **drawn entry's** band. A caller who does not
    yet know which entry was drawn calls this once with any band to read
    `entry_index` back -- the first two draws do not depend on the band -- and
    then again with that entry's own ``cut_min``/``cut_max`` to get the cut.
    :func:`select_seed_entry` does the same thing in one pass because it holds
    the entries.

    Args:
        run_seed: The seed the worker was handed (see
            :func:`select_seed_entry`).
        worker_id: The worker's index within the run.
        incarnation: The worker's respawn counter.
        game_index: The game's index within the worker's workload.
        n_entries: How many entries the file the worker read held.
        prob: The ``seed_openings_prob`` that worker ran under.
        cut_min: The drawn entry's low cut edge.
        cut_max: The drawn entry's high cut edge, inclusive.

    Returns:
        ``(seeded, entry_index, cut)`` -- the second and third are ``None``
        exactly when the game was not seeded.
    """
    choice = _draw(
        seed_openings_rng(run_seed, worker_id, incarnation, game_index),
        n_entries=n_entries,
        prob=prob,
        bounds=lambda _index: (cut_min, cut_max),
    )
    return (choice.seeded, choice.entry_index, choice.cut)
