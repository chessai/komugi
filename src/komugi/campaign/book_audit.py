"""The draft-book audit: what a campaign's shards say about its books (KOM-014).

``docs/design/kom-014-draft-books.md`` §6.1(b) and §6.7 are the authority. Two
questions are answered here, and neither of them needs a record field that the
worker does not already write:

**"What share of the games were booked?"** -- :func:`booked_fraction`, off
§4.3's *three-way* ``min(ply)`` classification. A booked game's rows begin at
``ply == L ∈ [4, 12]`` because the prefix is played and not recorded, so
booked-ness is derivable and a ``booked`` column would be a second source of
truth for a fact the data already determines. But ``min(ply) > 0`` is **not** a
booked-ness test: a game one of whose non-final shards was skipped presents
``min(ply) > 0`` with no book involved, and *"one entry after a kill is the
expected shape of a preemption"*. Such a game is **truncated** -- excluded from
the numerator *and* the denominator, reported separately, never silently
counted as booked (N2). The interval reported beside the fraction is the
**exact-binomial** two-sided 99% one, because the whole argument of §6.1 is that
``[8%, 12%]`` at ``n = 200`` is a ±1σ band that fails one honest run in three,
and an interval that was itself an approximation could not have made it.

**"Which entry did *that* game use?"** -- :func:`recompute_choice`, and the
answer is recomputed rather than looked up. §4.2's draw is a pure function of
``(run_seed, worker_id, incarnation, game_index)``; a row carries ``game_id``,
whose packing is invertible (:func:`~komugi.selfplay.worker.unpack_game_id`);
and the seed a *worker* was handed is not the campaign's own but
``iteration_seed(campaign.run_seed, iteration)``
(:func:`~komugi.campaign.loop.iteration_seed`). That two-level derivation is the
part a worker-level test cannot see (N3): every worker-level derivation is
correct *relative to* whatever seed it was handed, so an audit that fed
``campaign.run_seed`` straight into :func:`~komugi.selfplay.book.book_select_rng`
would be self-consistently wrong.

**The residual this module cannot fix, stated rather than hidden (R9).** When
data loss keeps ``min(ply)`` inside ``[4, 12]`` -- a game booked at ``L = 4``
that also lost plies 4…7 presents ``m = 8`` -- it reads as *booked with a
fabricated ``L``*, and nothing in the row set distinguishes the two. The
classification reports what it can defend and no more; a non-zero
``truncated`` count is the signal to read next to ``ShardReader.skipped``.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from komugi.campaign.loop import iteration_seed
from komugi.selfplay.book import (
    Book,
    BookChoice,
    BookedFraction,
    booked_fraction as _shard_booked_fraction,
    clopper_pearson,
    select_book_entry,
)
from komugi.selfplay.worker import unpack_game_id

__all__ = [
    "BookedFraction",
    "RecomputedChoice",
    "booked_fraction",
    "clopper_pearson",
    "recompute_choice",
    "recompute_choices",
]


def booked_fraction(
    root: str | os.PathLike[str], iterations: Iterable[int]
) -> BookedFraction:
    """Classify every game of `iterations` and report the booked share.

    A thin re-export of :func:`komugi.selfplay.book.booked_fraction`, which is
    where the classification lives because the *builder* uses the identical
    three-way rule to decide which source games it may mine (§3.2 step 2). One
    rule, one implementation: an audit that classified games differently from
    the builder would disagree with it about which games were booked, and the
    disagreement would look like a booking-rate anomaly.

    Args:
        root: The campaign root.
        iterations: Which iterations to classify. Pass a range to accumulate:
            §9's R4 is a claim about ≥ 1500 games, which is ~18 production
            iterations, and the per-iteration number is noise (§6.1(c)).

    Returns:
        The counts, the fraction over ``booked + unbooked``, and the exact
        99% interval for that denominator.
    """
    return _shard_booked_fraction(root, iterations)


@dataclass(frozen=True, slots=True)
class RecomputedChoice:
    """What §4.2's draw *would* have decided for one game (§6.7).

    Args:
        game_id: The id as it appears on the game's rows.
        worker_id: Recovered from `game_id`.
        incarnation: Recovered from `game_id`.
        game_index: Recovered from `game_id`.
        booked: Whether that game started from a book prefix.
        entry_id: The entry's :attr:`~komugi.selfplay.book.BookEntry.id`, or
            ``None`` when the game was unbooked. The *id* rather than the index
            because an index means nothing without the book it indexes, and the
            id is stable across a re-read of the same file.
        entry_index: The index within :attr:`~komugi.selfplay.book.Book.entries`,
            or ``None`` -- carried so a caller can reach the prefix itself.
    """

    game_id: int
    worker_id: int
    incarnation: int
    game_index: int
    booked: bool
    entry_id: str | None
    entry_index: int | None


def recompute_choice(
    run_seed: int,
    iteration: int,
    game_id_value: int,
    book: Book,
    fraction: float,
) -> RecomputedChoice:
    """Recompute one game's book decision from the **campaign's** seed.

    The full two-level derivation of §6.7, in three lines and no more:

    ``iteration_seed(run_seed, iteration)`` is the seed the worker was handed
    (``loop.py``'s ``_orchestrator_config`` puts it on
    :class:`~komugi.selfplay.orchestrate.OrchestratorConfig.run_seed`, the
    orchestrator hands it to ``worker_main``, and the worker derives every game
    seed from it); ``unpack_game_id`` recovers the spawn key; and
    :func:`~komugi.selfplay.book.select_book_entry` is the worker's own
    function, called rather than re-implemented -- a second copy of the two
    draws would agree with the worker exactly until the day one of them changed.

    Args:
        run_seed: The **campaign's** ``run_seed``, not a worker's.
        iteration: Which iteration the game belongs to.
        game_id_value: The ``game_id`` on the game's rows.
        book: The book that iteration played from.
        fraction: The ``book_fraction`` it played at.

    Returns:
        The recomputed :class:`RecomputedChoice`.
    """
    worker_id, incarnation, game_index = unpack_game_id(game_id_value)
    choice: BookChoice = select_book_entry(
        iteration_seed(run_seed, iteration),
        worker_id,
        incarnation,
        game_index,
        entries=len(book),
        fraction=fraction,
    )
    index = choice.entry_index if choice.booked else None
    return RecomputedChoice(
        game_id=game_id_value,
        worker_id=worker_id,
        incarnation=incarnation,
        game_index=game_index,
        booked=choice.booked,
        entry_id=None if index is None else book.entries[index].id,
        entry_index=index,
    )


def recompute_choices(
    run_seed: int,
    iteration: int,
    game_ids: Sequence[int],
    book: Book,
    fraction: float,
) -> tuple[RecomputedChoice, ...]:
    """Recompute :func:`recompute_choice` for every id in `game_ids`, in order.

    Order-preserving and duplicate-preserving: the caller's list is usually the
    ``game_id`` column of a shard read in row order, and silently deduplicating
    it would make the result unjoinable with the rows it came from.
    """
    return tuple(
        recompute_choice(run_seed, iteration, int(value), book, fraction)
        for value in game_ids
    )
