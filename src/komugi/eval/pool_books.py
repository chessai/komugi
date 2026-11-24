"""Epoch books: the frozen, versioned openings one pool epoch is played on.

``docs/design/kom-011d-ladder-pool.md`` §D3 is the authority. What this module
adds to KOM-014's draft books is **lifecycle**, not format: a draft book is a
self-play initial-state device *refreshed every generation*
(``docs/design/kom-014-draft-books.md`` §5.2), and a match book must be
*"versioned and frozen per ladder epoch -- refreshing books mid-series breaks
cross-generation comparability"* (``docs/DESIGN.html:260``). A
:class:`PoolBook` is that frozen artefact: a selection of :data:`POOL_BOOK_K`
prefixes from **one named KOM-014 book file**, hashed over its own canonical
bytes, written once and thereafter read-only.

Four properties, each a refusal rather than a convention:

* **A source that cannot supply ``K`` is refused, not stretched** (§D3.2).
  ``book.MIN_ENTRIES`` permits a legal book with 32 entries, so drawing 128
  without replacement is not always possible. Drawing *with* replacement would
  repeat openings inside one matchup -- the correlated games ``K`` exists to
  prevent -- and silently shrinking ``K`` would change an epoch's opening count
  without changing anything a reader would notice. The remedy is to name a
  different source book, and the refusal happens at build time, before a game
  is paid for.
* **A build over an existing path refuses and leaves the bytes alone** (§D3.3).
  An epoch book is written once; a rebuild that overwrote one would silently
  re-define an id that stored records already carry.
* **Every prefix is re-verified by replay before the book exists at all.** The
  source book's legality claim was made under whatever engine built it, and
  :func:`komugi.selfplay.book.load_book` already refuses a stale
  ``action_size`` or ``gungi_mode`` for exactly this reason. The verification
  lives in :meth:`PoolBook.__post_init__`, so *no* code path -- build, load, or
  a hand-constructed fixture -- can produce a book whose prefix is illegal or
  leaves a terminal position. That is what makes "a prefix that ends in a
  terminal position is a construction error" (§6 node 14) true of the type
  rather than of one function.
* **Pair ``k`` uses entry ``k``, totally and deterministically** (§D3.2). Both
  games of the pair replay it, so the two entrants see the identical opening
  from opposite seats and :func:`komugi.eval.seats.check_paired_books` -- which
  audits *"two games per pair_id, mirrored seats, same book_id"* -- stops being
  vacuous the moment a book lands. A matchup with more pairs than the book has
  entries is refused; the epoch book is not stretched.

**Where the torch boundary actually is.** This module reaches the engine (a
prefix is verified by replaying it) **and, transitively, torch**: ``num_actions``
comes from :mod:`komugi.selfplay.records` and :class:`GameState` from
:mod:`komugi.selfplay.adapter`, both of which import torch at module level. So an
epoch book is *not* mintable on a box that has no torch installed, and this
docstring no longer says it is. What the module does not reach is the **arena**:
:mod:`komugi.eval.arena` imports *this*, never the other way round, which is what
keeps the book seam a leaf of the eval package rather than a cycle through the
match driver. Both halves are pinned by
``tests/eval/test_pool_books.py::test_the_book_module_pulls_torch_only_through_the_engine``,
in node 29's spawned-child idiom, so neither sentence can silently become false.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

from komugi.engine.game import Game
from komugi.selfplay.adapter import GameState
from komugi.selfplay.book import Book, load_book, running_gungi_mode
from komugi.selfplay.book import _write_durably as write_durably
from komugi.selfplay.records import num_actions

POOL_BOOK_VERSION: Final[int] = 1
"""The epoch book's format version. Bumped when a *reader* must change.

Checked first by :func:`load_pool_book`, before any other manifest key, for
:func:`komugi.selfplay.book.load_book`'s reason: it is the only field that says
whether the rest of the manifest means what this reader thinks it means.
"""

POOL_BOOK_K: Final[int] = 128
"""``K``: how many prefixes an epoch book carries (§D3.2), derived not chosen.

One entry per pair of the **largest matchup it must serve**: a 128-game matchup
is 64 seat-swapped pairs and §D2.2(b)1's in-family link is 128 pairs. ``K``
below the pair count repeats openings inside a matchup (the correlated games
``DESIGN.html:169``'s "two identical 32-game pits differed by 41 points" is
about); ``K`` above it leaves entries unused and needs a selection rule the
artefact does not carry, while the pair-to-entry map must be total and
deterministic for two builds of one epoch to play the same games.
"""

DIGEST_CHARS: Final[int] = 8
"""Hex characters of a sha256 that reach an id -- §D3.2's ``sha8``."""

_REPLAY_DRAW_SCORE: Final[float] = 0.0
"""``draw_score`` for the verification replay.

Contempt cannot change legality or termination, and a book is not a search --
:data:`komugi.selfplay.book._REPLAY_DRAW_SCORE` makes the same choice for the
same reason.
"""

_JSON_SEPARATORS: Final[tuple[str, str]] = (",", ":")
"""KOM-014's canonical-JSON rules (``book.py:900-902``), so two builds of one
epoch from one source are **byte-identical** -- which is what the version hash
means."""


def epoch_prefix(epoch: int) -> str:
    """Return an epoch's id prefix: ``E007`` at epoch 7.

    Every entry id and the ``epoch_book_id`` itself carry it, which is what
    makes a record set that mixes two book epochs detectable by inspection of
    :meth:`komugi.eval.records.MatchSet.book_ids` (§D3.3) and what
    :func:`komugi.eval.divergence.book_era` parses.

    Raises:
        ValueError: If `epoch` is below 1; a pool epoch is numbered from 1.
    """
    if epoch < 1:
        raise ValueError(f"a pool epoch is numbered from 1, got {epoch!r}")
    return f"E{epoch:03d}"


def _sha8(payload: str) -> str:
    """Return the first :data:`DIGEST_CHARS` hex characters of sha256(`payload`)."""
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:DIGEST_CHARS]


def canonical_json(payload: Any) -> str:
    """Return `payload` as the bytes an epoch book is hashed and stored as.

    ``sort_keys``, no whitespace, and ``allow_nan=False`` -- KOM-014's rules
    (``book.py:900-902``). ``allow_nan`` is not decoration: the default emits
    the bare tokens ``NaN``/``Infinity``, which are not JSON and which a strict
    reader refuses, so a non-finite value is a build-time exception here rather
    than an unparseable artefact.
    """
    return json.dumps(
        payload, sort_keys=True, separators=_JSON_SEPARATORS, allow_nan=False
    )


def entry_id_for(epoch: int, index: int, actions: Sequence[int]) -> str:
    """Return entry `index`'s id: ``E007-3f9c1a20`` (§D3.2).

    The digest covers the index **and** the actions, not the actions alone: two
    source entries can carry the same prefix (a KOM-014 book dedups on the whole
    action tuple, not across builds), and two entries of one epoch book sharing
    an id would make ``book_id`` ambiguous about which pair was played -- while
    :func:`komugi.eval.seats.check_paired_books` compares the string for
    equality and would never notice.
    """
    return f"{epoch_prefix(epoch)}-{_sha8(canonical_json([index, list(actions)]))}"


@dataclass(frozen=True, slots=True)
class PoolBookEntry:
    """One frozen opening: the prefix both games of a pair replay.

    ``actions`` replays in order from a bare :class:`~komugi.engine.game.Game`.
    There is no position blob for :class:`komugi.selfplay.book.BookEntry`'s own
    reason: a position is not a portable artefact and an action list is.

    ``plies`` is not constrained to KOM-014's ``[MIN_PLIES, MAX_PLIES]`` here.
    That range is the *source* book's rule, enforced by
    :func:`komugi.selfplay.book.load_book` where the source is read; what an
    epoch book requires of a prefix is that it is legal and leaves a position
    with a move to make, which :class:`PoolBook` verifies by replay.

    Args:
        id: ``E<NNN>-<sha8>``, epoch-prefixed (:func:`entry_id_for`) and already
            a legal ``ArenaConfig.book_id``.
        actions: The prefix, in play order.
        plies: ``len(actions)``, carried so a hand-edited file disagrees with
            itself loudly rather than replaying a truncated line.
        source_entry_id: The KOM-014 entry this was selected from, so a line in
            a published pool can be traced back to the self-play game it came
            from.
    """

    id: str
    actions: tuple[int, ...]
    plies: int
    source_entry_id: str

    def __post_init__(self) -> None:
        """Reject an entry that cannot describe a prefix."""
        if not self.id.strip():
            raise ValueError("a pool book entry needs a non-empty id")
        if not self.source_entry_id.strip():
            raise ValueError(
                f"pool book entry {self.id!r} names no source entry: an epoch "
                f"book's only source is a KOM-014 book file (§D3.4), and the "
                f"provenance is what says which one"
            )
        if self.plies != len(self.actions):
            raise ValueError(
                f"pool book entry {self.id!r} carries {len(self.actions)} "
                f"actions but claims {self.plies} plies"
            )
        if self.plies < 1:
            raise ValueError(
                f"pool book entry {self.id!r} carries no actions: an epoch book "
                f"entry is an opening, and a zero-ply one is the empty board "
                f"under a book id that says otherwise"
            )

    def as_json(self) -> dict[str, Any]:
        """Return this entry as the manifest's object (§D3.2)."""
        return {
            "id": self.id,
            "actions": list(self.actions),
            "plies": self.plies,
            "source_entry_id": self.source_entry_id,
        }


def verify_prefix(actions: Sequence[int], *, label: str) -> None:
    """Replay `actions` from a bare :class:`~komugi.engine.game.Game`, or raise.

    Three failures, all of them construction errors and none of them a game:

    * an action outside the running encoding's range, or illegal in the position
      it is played from -- the source book's legality claim was made under
      whatever engine built it (§D3.2);
    * a prefix that passes *through* a terminal position, which is the same
      thing one ply earlier;
    * a prefix that **leaves** a terminal position, which would hand the arena a
      game with no moves to play. §6 node 14: refused before the match, rather
      than discovered at the loop's first ``is_terminal()`` check and scored as
      an empty game.

    Args:
        actions: The prefix, in play order.
        label: What to name in the message -- an entry id, or a source entry.

    Raises:
        ValueError: On any of the three.
    """
    size = num_actions()
    state = GameState(Game(), _REPLAY_DRAW_SCORE)
    for ply, action in enumerate(actions):
        if state.is_terminal():
            raise ValueError(
                f"prefix {label} passes through a terminal position at ply "
                f"{ply}: the engine had already adjudicated the game"
            )
        if not 0 <= action < size:
            raise ValueError(
                f"prefix {label} carries action {action!r} at ply {ply}, "
                f"outside [0, {size}) -- this process's action encoding"
            )
        if not bool(state.legal_mask()[action]):
            raise ValueError(
                f"prefix {label} plays illegal action {action} at ply {ply}: "
                f"the source book's legality claim does not hold under this "
                f"engine, so the line cannot be replayed"
            )
        state.play(action)
    if state.is_terminal():
        raise ValueError(
            f"prefix {label} leaves a terminal position after "
            f"{len(actions)} plies: a game started from it has no move to "
            f"play, which is a construction error and is refused before any "
            f"game is played (§6 node 14)"
        )


@dataclass(frozen=True, slots=True)
class PoolBook:
    """One frozen ladder book epoch (§D3.2), verified at construction.

    Every field is part of the version: :attr:`epoch_book_id` is ``sha8`` over
    the canonical bytes of all of them, so the entry list, the source book's
    bytes, its path, ``gungi_mode``, ``action_size``, ``engine_rules_version``,
    ``selection_seed``, ``k`` and ``pool_book_version`` each mint a new epoch
    and a new id when they change (§D3.3).

    Args:
        epoch: The pool epoch this book was minted for, numbered from 1.
        source_book: The KOM-014 book file it was selected from, as named.
        source_book_sha256: That **file's bytes**, not its manifest -- a book
            rebuilt from a different shard tree under the same name is a
            different source.
        gungi_mode: The start mode the prefixes are legal under
            (``book.py:242-252``); a prefix replayed under another *"is a
            different line, or illegal"*.
        action_size: The action encoding the ids mean something in
            (``book.py:1166-1173``'s guard, restated).
        engine_rules_version: §D1.3's rules revision. A repetition-rule change
            changes which games are drawn, so it mints a new book.
        selection_seed: Entropy of the without-replacement draw.
        k: How many entries were asked for, carried so a truncated file
            disagrees with itself.
        entries: The prefixes, in draw order.
        epoch_book_id: ``E<NNN>-<sha8>``; :func:`epoch_book_id_for` derives it
            and :func:`load_pool_book` re-derives it as a checksum.
    """

    epoch: int
    source_book: str
    source_book_sha256: str
    gungi_mode: str
    action_size: int
    engine_rules_version: int
    selection_seed: int
    k: int
    entries: tuple[PoolBookEntry, ...]
    epoch_book_id: str
    pool_book_version: int = POOL_BOOK_VERSION

    def __post_init__(self) -> None:
        """Verify the whole book: ids, counts, contents hash, and every prefix.

        **Every id is bound to the contents it claims to hash** (review F2).
        Carrying the ``E<NNN>-`` prefix is not enough: §D3.3's whole claim is
        *"the id **is** the version"*, and a :class:`PoolBook` assembled in code
        -- a fixture, a driver building one from a manifest it already parsed, a
        merge of two loaders -- could otherwise carry one book's
        :attr:`epoch_book_id` over a different entry list, pass the rated
        declaration check on the cross-family path
        (``v1bridge._assert_declared_book``), and store records whose entry ids
        name openings nothing in the artefact holds. Re-deriving here rather
        than in :func:`load_pool_book` alone is what makes the binding a
        property of the **type**: no path into it -- build, load or fixture --
        can produce a book whose id is not its own contents' hash.
        """
        prefix = epoch_prefix(self.epoch)
        if not self.epoch_book_id.startswith(f"{prefix}-"):
            raise ValueError(
                f"epoch book id {self.epoch_book_id!r} is not prefixed with "
                f"{prefix!r}: the epoch prefix is what makes a record set that "
                f"mixes two book epochs detectable (§D3.3)"
            )
        if not self.entries:
            raise ValueError(
                f"epoch book {self.epoch_book_id!r} carries no entries: a book "
                f"with nothing in it is not a bookless run, it is a run whose "
                f"records would claim an opening that does not exist"
            )
        if len(self.entries) != self.k:
            raise ValueError(
                f"epoch book {self.epoch_book_id!r} declares k={self.k} but "
                f"carries {len(self.entries)} entries"
            )
        seen: set[str] = set()
        for index, entry in enumerate(self.entries):
            if not entry.id.startswith(f"{prefix}-"):
                raise ValueError(
                    f"epoch book {self.epoch_book_id!r} carries entry "
                    f"{entry.id!r}, which is not prefixed with {prefix!r}"
                )
            if entry.id in seen:
                raise ValueError(
                    f"epoch book {self.epoch_book_id!r} carries entry id "
                    f"{entry.id!r} twice: a book_id must name one opening"
                )
            seen.add(entry.id)
            derived_entry = entry_id_for(self.epoch, index, entry.actions)
            if entry.id != derived_entry:
                raise ValueError(
                    f"epoch book {self.epoch_book_id!r} carries entry "
                    f"{entry.id!r} at index {index}, but that index and those "
                    f"actions hash to {derived_entry!r}: an entry id is a hash "
                    f"of the opening it names (§D3.2), so an id that is not one "
                    f"is a record-level claim about a line this book does not "
                    f"carry"
                )
            verify_prefix(entry.actions, label=entry.id)
        derived_book = epoch_book_id_for(self.versioned_fields())
        if self.epoch_book_id != derived_book:
            raise ValueError(
                f"epoch book id {self.epoch_book_id!r} is not the hash of its "
                f"own contents, which are {derived_book!r}: the id **is** the "
                f"version (§D3.3), so a book whose id disagrees with its entry "
                f"list, its source or its selection is an artefact every stored "
                f"record naming it describes wrongly"
            )

    def __len__(self) -> int:
        """Return how many entries -- and therefore how many pairs -- this serves."""
        return len(self.entries)

    def entry_for_pair(self, pair: int) -> PoolBookEntry:
        """Return the entry pair `pair` of a matchup plays (§D3.2: entry ``k``).

        Total and deterministic on purpose: it is what makes two builds of one
        epoch play the same games, and what gives every pair of a matchup its
        own distinct opening seen exactly twice, once from each seat.

        Raises:
            ValueError: If `pair` is negative, or beyond the book's ``K``. A
                matchup played at more pairs than the epoch book carries needs a
                **new epoch book** minted at a larger ``K`` (§D3.2); the book is
                not stretched and the openings are not repeated.
        """
        if pair < 0:
            raise ValueError(f"a pair index is non-negative, got {pair!r}")
        if pair >= len(self.entries):
            raise ValueError(
                f"epoch book {self.epoch_book_id!r} carries {len(self.entries)} "
                f"entries and pair {pair} was asked for: a matchup wider than K "
                f"repeats openings inside itself, so it needs a new epoch book "
                f"minted at a larger K rather than this one stretched (§D3.2)"
            )
        return self.entries[pair]

    def manifest(self) -> dict[str, Any]:
        """Return the whole artefact as §D3.2's JSON object, id included."""
        return {**self.versioned_fields(), "epoch_book_id": self.epoch_book_id}

    def versioned_fields(self) -> dict[str, Any]:
        """Return every field the :attr:`epoch_book_id` is a hash of (§D3.3)."""
        return {
            "pool_book_version": self.pool_book_version,
            "epoch": self.epoch,
            "source_book": self.source_book,
            "source_book_sha256": self.source_book_sha256,
            "gungi_mode": self.gungi_mode,
            "action_size": self.action_size,
            "engine_rules_version": self.engine_rules_version,
            "selection_seed": self.selection_seed,
            "k": self.k,
            "entries": [entry.as_json() for entry in self.entries],
        }


def epoch_book_id_for(versioned_fields: dict[str, Any]) -> str:
    """Return the ``epoch_book_id`` of a manifest's versioned fields (§D3.3).

    ``E<NNN>-<sha8 over the canonical bytes of everything above>``, where "above"
    is :meth:`PoolBook.versioned_fields` -- the whole artefact except the id
    itself. Any change to any of them mints a new epoch and a new id, which is
    the property that lets a stored record's ``book_id`` be trusted to name the
    openings it was actually played on.
    """
    prefix = epoch_prefix(int(versioned_fields["epoch"]))
    return f"{prefix}-{_sha8(canonical_json(versioned_fields))}"


def _select(source: Book, *, k: int, selection_seed: int) -> tuple[int, ...]:
    """Return the indices of `k` source entries, drawn without replacement.

    The bit generator is pinned **by name** for
    :data:`komugi.selfplay.book.BIT_GENERATOR`'s reason: this artefact is
    compared byte-wise across processes and machines, and ``default_rng``'s
    algorithm is documented as an implementation choice.

    Entries are kept in **draw order** rather than sorted back into file order,
    which is KOM-014's own rule (``book.py:746-751``): sorting would be equally
    deterministic and would hide a sampler that drew badly.
    """
    rng = np.random.Generator(
        np.random.PCG64(np.random.SeedSequence(selection_seed))
    )
    drawn = rng.choice(len(source.entries), size=k, replace=False)
    return tuple(int(index) for index in drawn)


def build_epoch_book(
    source_book: str | os.PathLike[str],
    target: str | os.PathLike[str],
    *,
    epoch: int,
    selection_seed: int,
    engine_rules_version: int,
    k: int = POOL_BOOK_K,
) -> PoolBook:
    """Mint one epoch book from one named KOM-014 book file, and publish it.

    The build is a **selection, not a sampler** (§D3.2): the source's entries in
    file order are the candidate pool, ``k`` are drawn without replacement, and
    every drawn prefix is re-verified by replay before the artefact exists
    (:class:`PoolBook`'s own constructor does the verifying, so no path into the
    type skips it).

    Args:
        source_book: The KOM-014 book file. **The only source there is** -- the
            same mechanical guarantee that makes a hand-curated match book
            impossible (``book.py:752-757``, ``DESIGN.html:250``) is inherited
            here by having no other argument to hand a line through.
        target: Where to publish, e.g.
            ``gungi-az-runs/pool/books/epoch_007.json``.
        epoch: The pool epoch this book is minted for.
        selection_seed: Entropy of the draw; part of the version.
        engine_rules_version: §D1.3's rules revision the book is frozen against.
            Required, with no default, because it is a *declaration* about the
            engine the games will be played under and inheriting it from a
            constant in another module would let it drift silently.
        k: How many entries to select. Defaults to :data:`POOL_BOOK_K`.

    Returns:
        The published :class:`PoolBook`.

    Raises:
        FileExistsError: If `target` already exists. An epoch book is written
            once and is thereafter read-only (§D3.3): the existing bytes are
            **not** touched, because a stored record already names them.
        ValueError: If `k` is not positive, if the source carries fewer than `k`
            entries (§6 node 47 -- refused rather than drawn with replacement or
            silently shrunk), or if any drawn prefix fails to replay.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k!r}")
    target_path = Path(target)
    # Checked before the source is even read: the refusal is about the artefact
    # that already exists, and reading a 300 KB book first would only make the
    # message later.
    if target_path.exists():
        raise FileExistsError(
            f"{target_path} already holds an epoch book: a book epoch is "
            f"written once and is thereafter read-only (§D3.3), because stored "
            f"records already carry the ids inside it. Refusing to overwrite; "
            f"mint the next epoch at its own path instead"
        )

    source_path = Path(source_book)
    source_bytes = source_path.read_bytes()
    source = load_book(source_path)
    if len(source.entries) < k:
        raise ValueError(
            f"source book {source_path} carries {len(source.entries)} entries "
            f"and K is {k}: a book epoch selects K prefixes without "
            f"replacement, and this source cannot supply them. Drawing with "
            f"replacement would repeat openings inside a matchup and shrinking "
            f"K would change the epoch's opening count invisibly, so neither is "
            f"done -- name a larger source book (§D3.2)"
        )

    entries: list[PoolBookEntry] = []
    for position, index in enumerate(
        _select(source, k=k, selection_seed=selection_seed)
    ):
        chosen = source.entries[index]
        entries.append(
            PoolBookEntry(
                id=entry_id_for(epoch, position, chosen.actions),
                actions=tuple(int(action) for action in chosen.actions),
                plies=len(chosen.actions),
                source_entry_id=chosen.id,
            )
        )

    fields: dict[str, Any] = {
        "pool_book_version": POOL_BOOK_VERSION,
        "epoch": epoch,
        # **Resolved**, not as passed (review F9): the path is a versioned field
        # (§D3.3), so the same source file named relatively from two working
        # directories would otherwise mint two ids for one artefact -- which
        # defeats §D3.2's "two builds of one epoch from one source are
        # byte-identical" for every build a script runs with a relative path.
        "source_book": os.fspath(source_path.resolve()),
        # The file's bytes, not its manifest: two books built from different
        # shard trees under one name are two different sources (§D3.2).
        "source_book_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "gungi_mode": source.gungi_mode,
        "action_size": source.action_size,
        "engine_rules_version": engine_rules_version,
        "selection_seed": selection_seed,
        "k": k,
        "entries": [entry.as_json() for entry in entries],
    }
    book = PoolBook(
        epoch=epoch,
        source_book=str(fields["source_book"]),
        source_book_sha256=str(fields["source_book_sha256"]),
        gungi_mode=source.gungi_mode,
        action_size=source.action_size,
        engine_rules_version=engine_rules_version,
        selection_seed=selection_seed,
        k=k,
        entries=tuple(entries),
        epoch_book_id=epoch_book_id_for(fields),
    )
    # KOM-014's own publish idiom, called rather than copied (``book.py:696-714``):
    # dot-prefixed temp file, fsync the bytes, os.replace, fsync the directory,
    # so a reader sees an epoch book either not at all or entirely. A second
    # copy of an atomicity sequence is a second thing to get subtly wrong.
    write_durably(target_path, canonical_json(book.manifest()) + "\n")
    return book


def _entry_from_json(payload: Any) -> PoolBookEntry:
    """Return one :class:`PoolBookEntry` from its manifest object.

    A missing key is a :class:`ValueError` here rather than a bare
    ``KeyError`` (review F4): :func:`load_pool_book` documents *"a missing
    manifest key"* as a ``ValueError``, and a caller that wrote
    ``except ValueError`` around the load exactly as documented would otherwise
    watch a truncated entry escape as an unhandled ``KeyError``.
    """
    if not isinstance(payload, dict):
        raise ValueError(f"an epoch book entry must be an object, got {payload!r}")
    try:
        return PoolBookEntry(
            id=str(payload["id"]),
            actions=tuple(int(action) for action in payload["actions"]),
            plies=int(payload["plies"]),
            source_entry_id=str(payload["source_entry_id"]),
        )
    except KeyError as exc:
        raise ValueError(
            f"an epoch book entry is missing manifest key {exc}"
        ) from exc


def load_pool_book(path: str | os.PathLike[str]) -> PoolBook:
    """Parse an epoch book, refusing one this process cannot replay (§D3.2).

    The refusals are :func:`komugi.selfplay.book.load_book`'s, for the same
    reasons and in the same order, plus the one an epoch book adds:

    * ``pool_book_version`` -- checked **first**, before any other manifest key
      is read, because it is the only field that says whether the rest of the
      manifest means what this reader thinks it means;
    * ``action_size`` -- the guard that actually protects a prefix: an action id
      means nothing across an encoding change, so a stale book would replay a
      plausible **wrong** line rather than fail;
    * ``gungi_mode`` -- a prefix is replayed from a bare
      :class:`~komugi.engine.game.Game`, and ``GUNGI_MODE`` decides which
      position that is;
    * ``epoch_book_id`` -- re-derived from the manifest's own versioned fields
      and compared. The id is the version (§D3.3); a file whose id does not
      match its contents has been edited since it was published, and every
      record naming that id is a claim about openings this file no longer holds.
      That derivation now lives in :meth:`PoolBook.__post_init__` (review F2),
      so this function's job is to **name the file** the refusal is about: the
      construction below is wrapped for exactly that, and for nothing else.

    Every prefix is replayed on the way in, and every id is re-derived, because
    :class:`PoolBook` verifies both at construction.

    **Every manifest key is read inside a guard** (review F4). Five of the nine
    used to be read bare, so a truncated or half-written file raised ``KeyError``
    while this docstring promised ``ValueError`` -- and a caller that handled the
    documented exception would have seen the undocumented one instead.

    Raises:
        ValueError: On unparseable JSON, a missing manifest key, a format
            version this reader does not read, a mismatched action space or
            start mode, an id that is not its own contents' hash, or an entry
            that will not replay.
    """
    book_path = Path(path)
    text = book_path.read_text(encoding="utf-8")
    try:
        payload: Any = json.loads(text)
    except ValueError as exc:
        raise ValueError(f"epoch book {book_path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(
            f"epoch book {book_path} must be a JSON object, got {type(payload)}"
        )

    try:
        version = int(payload["pool_book_version"])
    except KeyError as exc:
        raise ValueError(
            f"epoch book {book_path} is missing manifest key {exc}"
        ) from exc
    if version != POOL_BOOK_VERSION:
        raise ValueError(
            f"epoch book {book_path} declares pool_book_version {version}, but "
            f"this process reads pool_book_version {POOL_BOOK_VERSION}: the "
            f"constant is bumped exactly when a reader must change, so parsing "
            f"this file under the running version's rules would read one format "
            f"as if it were another"
        )

    try:
        action_size = int(payload["action_size"])
        mode = str(payload["gungi_mode"])
        stored_id = str(payload["epoch_book_id"])
        epoch = int(payload["epoch"])
        source_book = str(payload["source_book"])
        source_book_sha256 = str(payload["source_book_sha256"])
        engine_rules_version = int(payload["engine_rules_version"])
        selection_seed = int(payload["selection_seed"])
        k = int(payload["k"])
        raw_entries = payload["entries"]
    except KeyError as exc:
        raise ValueError(
            f"epoch book {book_path} is missing manifest key {exc}"
        ) from exc
    if not isinstance(raw_entries, list):
        raise ValueError(
            f"epoch book {book_path} carries entries of type "
            f"{type(raw_entries)}: the manifest's 'entries' is a list of "
            f"objects (§D3.2)"
        )

    running_actions = num_actions()
    if action_size != running_actions:
        raise ValueError(
            f"epoch book {book_path} was built for an action space of "
            f"{action_size} actions, but this process encodes "
            f"{running_actions}: an action id means nothing across an encoding "
            f"change, so the prefixes cannot be replayed"
        )
    running_mode = running_gungi_mode()
    if mode != running_mode:
        raise ValueError(
            f"epoch book {book_path} was built under GUNGI_MODE={mode!r}, but "
            f"this process runs {running_mode!r}: a prefix replayed from a "
            f"different start position is a different line, or illegal"
        )

    try:
        entries = tuple(_entry_from_json(item) for item in raw_entries)
        return PoolBook(
            epoch=epoch,
            source_book=source_book,
            source_book_sha256=source_book_sha256,
            gungi_mode=mode,
            action_size=action_size,
            engine_rules_version=engine_rules_version,
            selection_seed=selection_seed,
            k=k,
            entries=entries,
            epoch_book_id=stored_id,
            pool_book_version=version,
        )
    except ValueError as exc:
        # The type's own refusals -- the re-derived ids (§D3.3), the entry
        # counts, and every prefix replayed -- with the **file** named. A
        # message that says only "the id is not the hash of its contents" is
        # not actionable when a driver loads a directory of epoch books.
        raise ValueError(f"epoch book {book_path} is refused: {exc}") from exc


__all__ = [
    "DIGEST_CHARS",
    "POOL_BOOK_K",
    "POOL_BOOK_VERSION",
    "PoolBook",
    "PoolBookEntry",
    "build_epoch_book",
    "canonical_json",
    "entry_id_for",
    "epoch_book_id_for",
    "epoch_prefix",
    "load_pool_book",
    "verify_prefix",
]
