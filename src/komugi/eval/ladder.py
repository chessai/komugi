"""The ladder driver: a checkpoint directory in, one rated ladder out.

:mod:`komugi.eval.arena` knows how to play *one* match and
:mod:`komugi.eval.elo` how to fit *one* match set. This module is the thin,
opinionated thing between them -- it decides which checkpoints meet, on what
entropy, and how the games are pooled -- and it is thin on purpose: everything
here is a scheduling decision, and the two decisions that are not obvious are
the two the arena's docstrings warn about.

Why adjacent rungs (and not a round robin)
==========================================
``n`` checkpoints played round-robin is ``n*(n-1)/2`` matches; the chain is
``n-1``. What the chain buys for that is what a training curve actually needs:
consecutive checkpoints are the *closest* matched pair available, so their games
are the most informative per game played, and the chain is connected, which is
the condition the Davidson MLE needs to be identified at all (see
:mod:`komugi.eval.elo`'s identifiability note). What it costs is redundancy:
one badly-behaved match on the chain is a single bridge, and a rung whose games
are all draws leaves the two sides of it only loosely tied together.
:attr:`LadderConfig.adjacency` is the dial between the two -- ``2`` also plays
``i`` against ``i+2``, which double-links the chain for ``n-1`` extra matches --
and there is no round-robin setting because "``adjacency`` = the whole field"
already is one.

The two arena constraints this module exists to honour
======================================================
* **One matchup, one match.** :func:`~komugi.eval.arena.play_match` restarts its
  pair indices at ``00000`` every call and puts no seed in the ``pair_id``, so
  two matches between the same two checkpoints collide on ``pair_id`` and fail
  :func:`~komugi.eval.seats.check_paired_books` with ``GAME_COUNT`` violations
  once pooled. Every matchup here is therefore played exactly once, at its full
  game count, and the pooled :class:`~komugi.eval.records.MatchSet` stays
  auditable.
* **One matchup, one seed.** An :class:`~komugi.eval.arena.ArenaConfig`'s seed
  key does not include the entrants, so two matchups sharing a seed replay the
  *identical* per-ply Gumbel streams -- correlated games, and a joint fit over
  them is quietly overconfident. :func:`matchup_seed` gives each matchup its own
  seed, derived from the run seed and the matchup index, so a ladder run is
  still named by one integer and two of its matches share a stream only if a
  128-bit sequence collides under a 64-bit reduction -- a probability, not a
  guarantee; :func:`matchup_seed` states it.

No file I/O, and no plotting: :meth:`LadderResult.report` returns a string and
the caller decides where it goes. Like :mod:`komugi.eval.arena`, this module is
**not** re-exported from :mod:`komugi.eval` -- it reaches the engine, the search
and torch through the arena.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from komugi.eval.arena import (
    EMPTY_BOOK,
    LADDER_FAMILY,
    ArenaConfig,
    ArenaPlayer,
    ladder_players,
    play_match,
)
from komugi.eval.elo import EloResult, fit_elo
from komugi.eval.pool import EpochManifest
from komugi.eval.pool_books import PoolBook
from komugi.eval.records import GameRecord, MatchSet
from komugi.selfplay.records import move_seed

MIN_PLAYERS: int = 2
"""Fewest checkpoints a ladder can be fit to: one match needs two entrants."""

MIN_GAMES_PER_MATCHUP: int = 2
"""Fewest games a ladder matchup can be: one seat-swapped pair.

Stricter than the arena, which allows ``games=0`` -- a coherent request for one
match and an incoherent one for a ladder, where it plays every matchup at zero
games and then dies inside :func:`~komugi.eval.elo.fit_elo` on an empty match
set, an error about the data for what was a config mistake.
"""


def matchup_seed(run_seed: int, index: int) -> int:
    """Return the arena seed for matchup number `index` of a ladder run.

    ``SeedSequence(run_seed, spawn_key=(index,))`` reduced to one 64-bit
    integer, spelled as :func:`komugi.selfplay.records.move_seed` over the run's
    bare sequence -- the same spawn-key composition self-play uses to name a
    ply, applied one level up to name a match. Composing the existing helper
    rather than hand-rolling a ``SeedSequence`` keeps a single derivation in the
    codebase; ``move_seed`` is spelled for plies but is exactly "child `k` of
    this sequence", which is what a matchup index is here.

    Pure, like everything else in the seed stack: asking twice for matchup 3's
    seed answers the same both times, so a ladder run is fully named by
    ``(config.seed, the checkpoint set, the config)``.

    Distinct per index only *probabilistically*, and the docstrings elsewhere
    should not be read as promising more: ``generate_state`` reduces a 128-bit
    sequence to one ``uint64``, so two indices colliding is a birthday argument
    (~``n**2 / 2**65``: about ``3e-16`` over the 97 matchups of a 50-checkpoint
    ladder at ``adjacency=2``), not an injectivity theorem. A collision would be
    silent -- two matchups replaying identical Gumbel streams, with nothing in
    the report to show it -- which is why the number is written down here.
    """
    return move_seed(np.random.SeedSequence(run_seed), index)


@dataclass(frozen=True, slots=True)
class LadderConfig:
    """What a ladder run plays, from what entropy, and against what anchor.

    Args:
        games_per_matchup: Games in **each** matchup. Must be even, which is
            checked here by constructing the :class:`ArenaConfig` the matches
            will be played under -- so the rule cannot drift from the arena's
            (a match is a whole number of seat-swapped pairs) -- and at least
            ``2``, which is the ladder's own rule and *stricter* than the
            arena's: the arena reads ``games=0`` as "play nothing", which is
            coherent for one match and not for a ladder, where it means every
            matchup plays nothing and the run dies in :func:`fit_elo` with
            "cannot fit Elo to an empty match set".
        seed: The run's entropy. Every matchup gets its own seed derived from
            this and its index (:func:`matchup_seed`), never this value itself.
            Validated here against :class:`numpy.random.SeedSequence` -- the
            thing that will actually consume it -- because the arena does not
            validate its seed and a negative one would otherwise surface as a
            numpy error from inside the game loop, hours in.
        draw_score: The contempt every game is played under. **Required, no
            default**, for design review D3's reason: it reaches both the
            terminal and the network values, and a ladder fit from games played
            at a different contempt from the self-play that trained the
            checkpoints is a different experiment.
        every_n: Sample stride over the checkpoints, oldest-first
            (:func:`~komugi.eval.arena.ladder_players`). ``1`` rates every
            saved checkpoint.
        anchor: Player id pinned to 0 Elo. ``None`` (the default) means the
            **oldest** rung, which makes every rating a "gained since the start
            of training" number and makes two runs of the same directory
            comparable; anchoring on the newest would move the whole table every
            time a checkpoint was written.
        adjacency: How far along the chain each rung plays. ``1`` is the pure
            chain (``i`` vs ``i+1``); ``k`` adds ``i`` vs ``i+2 .. i+k``.
        book: The frozen epoch book every matchup is played on
            (:class:`~komugi.eval.pool_books.PoolBook`), or ``None`` -- the
            default, and the empty-board ladder every stored run was played as.
            It is here for the same reason `draw_score` is (kom-011d §D4.2): for
            the in-family series to be *comparable* with the pool's rather than
            merely adjacent to it, the two must be played under the same search
            budget, the same contempt **and** the same openings, and
            :func:`require_pool_agreement` is what refuses to publish them
            together when they were not.
        pair_namespace: Prefixed onto every ``pair_id`` this run writes
            (:func:`~komugi.eval.arena.pair_id_for`), so a ladder replayed as
            part of a later pool epoch does not collide with its own earlier
            records. ``""`` -- the default -- is today's ids byte for byte.

    Raises:
        ValueError: On `adjacency` below 1, on `games_per_matchup` below 2, on
            a `seed` :class:`numpy.random.SeedSequence` will not accept, on a
            non-positive `every_n`, on a blank `anchor`, or on anything the
            arena refuses (odd `games_per_matchup`, `draw_score` outside
            ``[-1, 1]``, a `pair_namespace` containing ``":"``, a `book` with
            fewer entries than a matchup has pairs).
    """

    games_per_matchup: int
    seed: int
    draw_score: float
    every_n: int = 1
    anchor: str | None = None
    adjacency: int = 1
    book: PoolBook | None = None
    pair_namespace: str = ""

    def __post_init__(self) -> None:
        """Reject a config that could not describe a ladder, before any loading."""
        if self.adjacency < 1:
            raise ValueError(
                f"adjacency must be >= 1, got {self.adjacency!r}: a ladder with "
                f"no adjacent matches plays no games at all"
            )
        if self.every_n < 1:
            raise ValueError(f"every_n must be >= 1, got {self.every_n!r}")
        if self.games_per_matchup < MIN_GAMES_PER_MATCHUP:
            raise ValueError(
                f"games_per_matchup must be >= {MIN_GAMES_PER_MATCHUP}, got "
                f"{self.games_per_matchup!r}: a ladder matchup is at least one "
                f"seat-swapped pair, and a zero-game ladder does not fail until "
                f"the Elo fit, as an empty match set"
            )
        # The seed is validated against the thing that will consume it, not
        # against the arena: ArenaConfig does not check its seed at all, so a
        # negative one would otherwise reach numpy from inside the game loop.
        try:
            np.random.SeedSequence(self.seed)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"seed must be usable as numpy entropy, got {self.seed!r}: {error}"
            ) from error
        # Validation by derivation: the arena owns the rules about game counts
        # and contempt, and this is the config the matches will actually carry
        # -- the run's own seed included, so the probe cannot pass a config the
        # matches would not.
        self.arena_config(seed=self.seed)
        if self.anchor is not None and not self.anchor.strip():
            raise ValueError("anchor must be a non-empty player id, or None")

    def arena_config(self, seed: int) -> ArenaConfig:
        """Return the :class:`ArenaConfig` one matchup is played under.

        With no book and the default namespace this is the exact config every
        stored ladder run was played under -- the empty-board ``book_id``, no
        pair prefix -- which is what §6 node 60 pins. With a book, ``book_id``
        names the **epoch book** and each pair's records carry their entry's id
        (``ArenaConfig.book_id_for_pair``), the same split the bridged path
        declares (KOM-023 §D4 amendment 4).
        """
        return ArenaConfig(
            games=self.games_per_matchup,
            seed=seed,
            draw_score=self.draw_score,
            book_id=EMPTY_BOOK if self.book is None else self.book.epoch_book_id,
            pair_namespace=self.pair_namespace,
            book=self.book,
        )


POOL_DRAW_SCORE: float = 0.0
"""The contempt every pool game is played at -- kom-011d §D7.3, frozen.

``0.0`` is KOM-023 §D3.4's pre-registration for every cross-family match
(``r5.py:97-98``); self-play's ``0.20`` (``DESIGN.html:165``) is a *training*
device chosen from v1's draw-collapse evidence, not a measurement setting. It is
a constant here rather than a parameter of :func:`require_pool_agreement` for
``r5.py:100-101``'s reason: a tolerance that can be passed in is a tolerance
that will be passed in.
"""


def manifest_contempt(manifest: EpochManifest) -> float | None:
    """Return the contempt a manifest says its **pool** was played at, if any.

    §D7.3's predicate is *the two series were played at the same contempt*, and
    §D9's manifest carries `contempt` as a **map per family** with its scale
    (§6 node 51: komugi's ``0.0`` on ``[-1, 1]``, v1's ``0.5`` on ``[0, 1]``,
    the external-anchor leg's ``null``). Until T4 lands that field, :class:`EpochManifest` has no
    contempt to compare with -- which is exactly why
    :func:`require_pool_agreement` could only ever check the ladder against
    :data:`POOL_DRAW_SCORE`, and why a pool epoch played at a non-neutral
    contempt would have been published beside a ``0.0`` ladder without the
    predicate ever being evaluated (review F7).

    So the field is read **structurally** rather than declared here: when a
    manifest carries a ``contempt`` map with a numeric ``value`` for
    :data:`~komugi.eval.arena.LADDER_FAMILY`, that number is the pool's own
    contempt and the caller compares against it; when it does not, the answer is
    ``None`` and the pre-registered constant is all there is to check. Reading it
    this way means T4's manifest is compared the day it lands, rather than
    passing silently until someone remembers this function.

    Returns:
        The pool's komugi-side contempt, or ``None`` when the manifest carries
        none (every manifest before T4's `contempt` map).
    """
    carried: object = getattr(manifest, "contempt", None)
    if not isinstance(carried, Mapping):
        return None
    family: object = carried.get(LADDER_FAMILY)
    if not isinstance(family, Mapping):
        return None
    value: object = family.get("value")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def require_pool_agreement(config: LadderConfig, manifest: EpochManifest) -> None:
    """Refuse to publish an in-family ladder beside a pool epoch it disagrees with.

    kom-011d §D4.2 rule 2 publishes the in-family ladder as its own series in
    its own units, and §D7.3 says what makes the two series *comparable* rather
    than merely adjacent: the same contempt and the same openings. A ladder run
    at self-play's ``0.20`` beside a pool run at ``0.0`` is two different
    experiments plotted on one page -- a contempt difference moves draw rates,
    which moves the Davidson ``nu``, which moves every rating.

    **What this no longer guards, stated because rev 2 narrowed it:** §D5.1 fits
    ``G_in`` over the pool's *own* in-family records, so a ladder
    misconfiguration can no longer corrupt the gate statistic ``D``. It can only
    make the two published series incomparable -- which is still a refusal, and
    still this function.

    Both values are named in every message: "the ladder is misconfigured" is not
    an actionable sentence.

    **The contempt is checked twice, against two different things** (review F7).
    :data:`POOL_DRAW_SCORE` is the *pre-registration* -- what §D7.3 says every
    pool game is played at -- and the ladder is held to it. But an epoch that was
    itself run at another contempt (a driver bug, or a deliberate experiment)
    satisfies that check and still fails §D7.3's actual predicate, so when the
    manifest carries its own contempt (:func:`manifest_contempt`) the ladder is
    compared against **that number too**, and the refusal names it.

    Args:
        config: The ladder run's configuration.
        manifest: The pool epoch the series would be published beside.

    Raises:
        ValueError: If the ladder's contempt differs from the pre-registered
            constant or from the manifest's own, or if the ladder was played on
            a different epoch book from the pool's -- including on none at all.
    """
    if config.draw_score != POOL_DRAW_SCORE:
        raise ValueError(
            f"the ladder was played at draw_score={config.draw_score!r} and the "
            f"pool plays every game at {POOL_DRAW_SCORE!r} (kom-011d §D7.3): a "
            f"contempt difference moves draw rates, which moves the Davidson "
            f"draw parameter, which moves every rating -- so the two series are "
            f"not comparable and may not be published as though they were"
        )
    pool_contempt = manifest_contempt(manifest)
    if pool_contempt is not None and config.draw_score != pool_contempt:
        raise ValueError(
            f"the ladder was played at draw_score={config.draw_score!r} and pool "
            f"epoch {manifest.epoch} records its own contempt as "
            f"{pool_contempt!r} (kom-011d §D7.3, §D9): the pre-registered "
            f"{POOL_DRAW_SCORE!r} is what the pool was *meant* to be played at, "
            f"and this epoch says it was not -- so the two series are not "
            f"comparable whichever of the two is wrong"
        )
    ladder_book = None if config.book is None else config.book.epoch_book_id
    if ladder_book != manifest.epoch_book_id:
        raise ValueError(
            f"the ladder was played on epoch book {ladder_book!r} and pool "
            f"epoch {manifest.epoch} on {manifest.epoch_book_id!r} (kom-011d "
            f"§D3, §D4.2): a milestone's gain over its predecessor is only "
            f"measured on the same openings if both series were played on them"
        )


class MatchupSummary(NamedTuple):
    """What one rung-to-rung match came to, in the ladder's own terms.

    A projection of :class:`~komugi.eval.arena.ArenaSummary` with the two
    entrant ids attached, because that is what a report row is. Draws and mean
    length are kept for the reason the arena keeps them: an all-draws rung has
    fit an Elo gap out of nothing, and neither number is recoverable from the
    records afterwards.
    """

    ids: tuple[str, str]
    games: int
    decisive: int
    draws: int
    mean_plies: float

    def line(self) -> str:
        """Return the one-line summary a `progress` callback is handed."""
        return (
            f"{self.ids[0]} vs {self.ids[1]}: {self.games} games, "
            f"{self.decisive} decisive, {self.draws} draws, "
            f"{self.mean_plies:.1f} mean plies"
        )


class LadderResult(NamedTuple):
    """A finished ladder: who played, what they rate, and the games behind it.

    ``fit`` is the :class:`~komugi.eval.elo.EloResult` verbatim -- ``ratings``
    is a convenience alias for ``fit.ratings`` and the rest of it
    (``converged``, ``draw_param``, ``iterations``) is not summarised away,
    because ``converged=False`` means "this ladder is not identified from these
    games" and a caller must be able to see it.

    ``anchor`` is the id the fit was anchored on, **carried, not re-derived**.
    A rating of exactly ``0.0`` does not identify the anchor: on any *balanced*
    ladder every rating is exactly ``0.0`` (each player scores half its games,
    so :mod:`komugi.eval.elo`'s MM step divides a number by itself and every
    ``theta`` is ``+0.0``), and the report would then name the oldest rung as
    the gauge no matter what was configured -- a false statement about the run
    in the one artifact a human reads. One string is cheaper than that.
    """

    players: tuple[str, ...]
    anchor: str
    ratings: dict[str, float]
    fit: EloResult
    per_matchup: tuple[MatchupSummary, ...]
    records: tuple[GameRecord, ...]

    def report(self) -> str:
        """Return a plain-text report: the rating table, then the matchups.

        The table is in **ladder order**, oldest first, and deliberately not
        sorted by rating: the question a training ladder answers is "did this
        run get stronger", which is a question about the shape of a column read
        top to bottom. Sorting by rating would answer a different one and would
        hide exactly the case worth seeing -- a middle checkpoint that outrates
        its successors.

        The anchor's row is marked, because ``0.0`` in the rating column does
        not mark it: a rung at ``0.04`` prints as ``0.0`` too, and on a balanced
        ladder every row does. A fit that did not converge gets its own banner
        line above the tables rather than only the tail of the fit line --
        ``converged=False`` means the ladder is not identified from these games
        and every number under it is a clamp, which is not a thing to leave a
        reader to notice.
        """
        header = (
            f"ladder: {len(self.players)} players, {len(self.per_matchup)} matchups, "
            f"{len(self.records)} games"
        )
        fit_line = (
            f"fit: anchor {self.anchor} at 0.0 Elo, "
            f"draw_param {self.fit.draw_param:.4f}, "
            f"{'converged' if self.fit.converged else 'NOT CONVERGED'} "
            f"in {self.fit.iterations} iterations"
        )
        banner: list[str] = (
            []
            if self.fit.converged
            else [
                "!! NOT CONVERGED: this ladder is not identified from these "
                "games -- the ratings below are clamps, not measurements"
            ]
        )
        player_rows = _table(
            ("player", "rating", ""),
            [
                (
                    player,
                    f"{self.ratings[player]:.1f}",
                    "anchor" if player == self.anchor else "",
                )
                for player in self.players
            ],
        )
        matchup_rows = _table(
            ("matchup", "games", "decisive", "draws", "mean plies"),
            [
                (
                    f"{summary.ids[0]} vs {summary.ids[1]}",
                    str(summary.games),
                    str(summary.decisive),
                    str(summary.draws),
                    f"{summary.mean_plies:.1f}",
                )
                for summary in self.per_matchup
            ],
        )
        return "\n".join(
            [
                header,
                fit_line,
                *banner,
                "",
                "players",
                *player_rows,
                "",
                "matchups",
                *matchup_rows,
                "",
            ]
        )


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    """Return `rows` as fixed-width text lines under `header`.

    First column left-aligned (it is an identifier), the rest right-aligned
    (they are numbers). No separator rule: the report is read in a terminal and
    in a commit message, and the widths already do the work.
    """
    columns = list(zip(header, *rows, strict=True))
    widths = [max(len(cell) for cell in column) for column in columns]

    def render(cells: Sequence[str]) -> str:
        """Pad one row's cells to the column widths."""
        return "  ".join(
            cell.ljust(width) if index == 0 else cell.rjust(width)
            for index, (cell, width) in enumerate(zip(cells, widths, strict=True))
        ).rstrip()

    return [render(header), *(render(row) for row in rows)]


def matchups(count: int, adjacency: int) -> tuple[tuple[int, int], ...]:
    """Return the ``(i, j)`` rung pairs of a ladder over `count` players.

    In **step-major** play order: the whole chain (``i`` vs ``i+1``) first, then
    every ``i`` vs ``i+2``, and so on. That order is what makes the property the
    seeds rest on true -- ``matchups(count, k)`` is a *prefix* of
    ``matchups(count, k + 1)``, so a matchup's index, and with it its seed
    (:func:`matchup_seed`), is a function of the rung pair alone and does not
    shuffle when `adjacency` changes the *number* of matches. Rung-major order
    would interleave the wider links into the chain and re-seed every chain
    matchup after the first, so the same two checkpoints would play *different*
    games at a different `adjacency` -- which is exactly the comparison an
    operator who widens a ladder to look again at one rung pair is making.

    Pairs beyond the top of the ladder are dropped rather than wrapped: a ladder
    is a line, not a ring. `adjacency` is bounded by the ladder's own reach
    rather than looped over and filtered, so "``adjacency`` = the whole field"
    -- the round robin the module docstring names, spelled by an operator who
    does not want to count checkpoints as ``adjacency=10**9`` -- costs the pairs
    it yields rather than ``count * adjacency`` discarded iterations before the
    first game.
    """

    def pairs() -> Iterator[tuple[int, int]]:
        for step in range(1, min(adjacency, count - 1) + 1):
            for i in range(count):
                if i + step < count:
                    yield (i, i + step)

    return tuple(pairs())


def run_ladder(
    directory: str | os.PathLike[str],
    config: LadderConfig,
    *,
    device: str = "cpu",
    progress: Callable[[str], None] | None = None,
) -> LadderResult:
    """Play a checkpoint directory's ladder and fit one Elo table to it.

    Every matchup is one :func:`~komugi.eval.arena.play_match` call on its own
    seed, and every record from every match is pooled into a single
    :class:`~komugi.eval.records.MatchSet` before the fit -- one fit over the
    whole chain, not a chain of two-player fits, because the ratings of
    non-adjacent rungs are only comparable through the joint likelihood.

    The epoch book and the pair namespace reach the matches through
    :meth:`LadderConfig.arena_config` and **nothing here changes with them**:
    the schedule, the seeds and the pooling are what they were, which is why
    ``config.book is None`` at the default namespace reproduces a stored run
    exactly (§6 node 60). Every matchup is played on the *same* epoch book, so
    a rung's gain over its neighbour is measured on the same openings as the
    pool measures the milestone's (§D4.2).

    Args:
        directory: A campaign's checkpoint directory. A missing one reads as
            empty (and is then refused for having too few players).
        config: What to play and how to anchor it.
        device: Where each checkpoint's net is placed, passed through to
            :func:`~komugi.eval.arena.ladder_players`.
        progress: Called once per **completed** matchup with a one-line summary
            (:meth:`MatchupSummary.line`, prefixed with the matchup's position).
            A ladder at the production budget runs for hours; this is the only
            thing it says while it does. ``None`` runs silently.

    Returns:
        The :class:`LadderResult`: ladder-ordered ids, the anchor they were fit
        against, the fit, the per-matchup summaries in play order, and every
        record played. Checking ``result.fit.converged`` before reading
        ``ratings`` is the caller's job: an unidentified ladder (a rung that
        swept its only neighbour, say -- the *normal* early-training case)
        returns clamped numbers that look exactly like measurements, and
        :meth:`LadderResult.report` is the only thing that says so.

    Raises:
        ValueError: If fewer than :data:`MIN_PLAYERS` checkpoints are found, if
            two of them carry the same id, or if `config.anchor` names a player
            that is not on the ladder -- all refused here, before the first
            match, rather than by :func:`~komugi.eval.arena.play_match` or
            :func:`~komugi.eval.elo.fit_elo` after the games are paid for.
    """
    entrants: tuple[ArenaPlayer, ...] = ladder_players(
        directory, every_n=config.every_n, device=device
    )
    if len(entrants) < MIN_PLAYERS:
        raise ValueError(
            f"a ladder needs at least {MIN_PLAYERS} checkpoints, found "
            f"{len(entrants)} in {os.fspath(directory)!r} at every_n="
            f"{config.every_n}: there is nothing to rate against"
        )
    ids = tuple(entrant.player.id for entrant in entrants)
    if len(set(ids)) != len(ids):
        raise ValueError(
            f"two rungs of this ladder share an id; players are {list(ids)}: "
            f"a directory holding two files for one step (a crash between a "
            f"checkpoint's publish and the supersede that unlinks the old one) "
            f"schedules a rung against itself, which play_match refuses -- but "
            f"only after every matchup before it has been played"
        )
    anchor = ids[0] if config.anchor is None else config.anchor
    if anchor not in ids:
        raise ValueError(
            f"anchor {anchor!r} is not on this ladder; players are {list(ids)}"
        )

    schedule = matchups(len(entrants), config.adjacency)
    records: list[GameRecord] = []
    summaries: list[MatchupSummary] = []
    for index, (first, second) in enumerate(schedule):
        outcome = play_match(
            entrants[first],
            entrants[second],
            config.arena_config(matchup_seed(config.seed, index)),
        )
        records.extend(outcome.records)
        summary = MatchupSummary(
            ids=(ids[first], ids[second]),
            games=outcome.summary.games,
            decisive=outcome.summary.decisive,
            draws=outcome.summary.draws,
            mean_plies=outcome.summary.mean_plies,
        )
        summaries.append(summary)
        if progress is not None:
            progress(f"[{index + 1}/{len(schedule)}] {summary.line()}")

    fit = fit_elo(MatchSet.of(records), anchor=anchor)
    return LadderResult(
        players=ids,
        anchor=anchor,
        ratings=fit.ratings,
        fit=fit,
        per_matchup=tuple(summaries),
        records=tuple(records),
    )
