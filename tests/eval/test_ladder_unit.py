"""Ladder structure: the schedule, the seeds, the pooling, and the report.

No games are played here. :func:`~komugi.eval.arena.play_match` is the seam this
suite cuts at -- a recorder stands in for it and hands back fabricated records
-- because everything the ladder driver decides is decided *around* a match:
which rungs meet, on what seed, how many times, and what the pooled records are
fit to. The games themselves are the arena's business and are already tested
next door; paying for them again to observe a schedule would be paying 256-turn
games to count to three.

Two of the assertions below are the two constraints
:mod:`komugi.eval.arena`'s docstrings warn about, made observable:

* every matchup is played **once** (a second match of the same two players
  collides on ``pair_id``), which the recorder can see because it sees every
  call; and
* every matchup gets its **own** seed (one seed across matchups correlates
  their Gumbel streams), which is only observable at this seam -- the seed never
  reaches a record.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pytest

from komugi.eval import ladder
from komugi.eval.arena import (
    EMPTY_BOOK,
    ArenaConfig,
    ArenaPlayer,
    ArenaSummary,
    MatchOutcome,
)
from komugi.eval.records import (
    BLACK_WIN,
    DRAW,
    WHITE_WIN,
    GameRecord,
    MatchSet,
    Player,
)
from komugi.eval.pool import EpochManifest
from komugi.eval.seats import check_paired_books
from komugi.selfplay.inference import BatchEvaluator

# §D3's epoch-book fixtures, built from a real KOM-014 book file rather than
# hand-rolled here -- the pass-through this file pins is only worth pinning
# against the artefact the pool actually mints.
from test_pool_books import epoch_book

DRAW_SCORE: float = 0.20
"""``DESIGN.html`` section 3.3's contempt, and ``SelfPlayConfig``'s default."""

GAMES: int = 4
"""Two seat-swapped pairs per matchup: enough to carry a mixed result pattern."""

PLIES_PER_GAME: int = 10
"""Fabricated game length, so the summaries have something to be checked against."""

ResultPattern = tuple[float, ...]
"""Results of games ``k % len(pattern)`` of one match, from white's perspective."""

RESULTS: ResultPattern = (BLACK_WIN, WHITE_WIN, WHITE_WIN, DRAW)
"""The default pattern: game ``k % 4`` of every fabricated match.

Read against :meth:`MatchRecorder.__call__`'s seating (game ``2k`` is ``a`` as
white, ``2k+1`` is ``b``), this gives the *younger* rung two wins, the older one
one win, and one draw -- an identified, mildly drawish ladder in which ratings
rise along the chain, which is what makes the anchor assertions meaningful
rather than vacuous.

The three patterns below are the shapes that ladder is *not*, each built for a
claim about the report that the rising, converged, one-rung-at-zero default
cannot falsify.
"""

BALANCED: ResultPattern = (WHITE_WIN, WHITE_WIN, BLACK_WIN, BLACK_WIN)
"""A matchup both rungs split 2-2, with no draws at all.

Games 0 and 3 go to the older rung, games 1 and 2 to the younger (apply the
seating). Every player then scores exactly half its games, which is the case
:mod:`komugi.eval.elo` fits to a table of *exact* zeros: ``odds = 1``, so
``theta`` starts at 0, and the MM step divides ``score_i`` by ``played_i / 2``,
which is the same number. A ladder of near-identical checkpoints is this.
"""

YOUNGER_SWEEP: ResultPattern = (BLACK_WIN, WHITE_WIN)
"""A matchup the younger rung wins outright, every game.

The winning colour alternates because the seating does: game ``2k`` seats the
older rung as white, game ``2k+1`` the younger. A rung scoring 0% against the
rest is the unidentified case :mod:`komugi.eval.elo`'s identifiability note
describes -- and the *normal* shape of an early-training ladder, whose first
checkpoint is randomly initialised.
"""

OLDER_WINS_3_1: ResultPattern = (WHITE_WIN, BLACK_WIN, WHITE_WIN, WHITE_WIN)
"""A matchup the older rung takes 3-1: decisive, but not a sweep."""

YOUNGER_WINS_3_1: ResultPattern = (BLACK_WIN, WHITE_WIN, BLACK_WIN, BLACK_WIN)
"""A matchup the younger rung takes 3-1 -- :data:`OLDER_WINS_3_1` mirrored."""


def _never_built() -> BatchEvaluator:
    """Stand in for a checkpoint's evaluator factory; must never be called.

    :func:`~komugi.eval.arena.play_match` is what builds an entrant's network,
    and it is replaced in this suite -- so a call here means the driver built a
    net behind the arena's back, which would make a ladder run pay for torch
    per rung whether or not it played anything.
    """
    raise AssertionError("the ladder built an evaluator outside play_match")


def fake_players(
    count: int, ids: Sequence[str] | None = None
) -> tuple[ArenaPlayer, ...]:
    """Return `count` entrants named ``rung-0``.. in ladder (oldest-first) order.

    ``rung-`` rather than the arena's ``ckpt-`` prefix, and single digits: no id
    here is a substring of another, which is what lets the report assertions
    count occurrences without matching ``ckpt-1`` inside ``ckpt-10``.

    `ids` names the entrants instead, for the one case the driver has to refuse
    before it plays anything and no honest `count` can express: a directory
    holding two files for one step, which
    :func:`~komugi.eval.arena.ladder_players` turns into two entrants sharing an
    id.
    """
    names = (
        tuple(f"rung-{index}" for index in range(count)) if ids is None else tuple(ids)
    )
    return tuple(
        ArenaPlayer(
            player=Player(id=name, family="komugi"),
            evaluator_factory=_never_built,
        )
        for name in names
    )


class MatchRecorder:
    """A stand-in for :func:`~komugi.eval.arena.play_match` that notes its calls.

    Produces the pairing the real one produces -- game ``2k`` with `a` as white,
    game ``2k+1`` with the seats swapped, both under ``"<a>:<b>:<k:05d>"`` -- so
    the records it hands back are auditable by
    :func:`~komugi.eval.seats.check_paired_books` exactly as real ones are, and
    the pooled match set the driver builds is a real test of the pooling.

    `results` decides which pattern a matchup is played to, as a function of the
    two entrant ids; ``None`` plays every matchup to :data:`RESULTS`. A ladder's
    *shape* -- balanced, swept, non-monotone -- is a property of which matchup
    went which way, so it is not expressible by one constant.
    """

    def __init__(
        self, results: Callable[[str, str], ResultPattern] | None = None
    ) -> None:
        self.calls: list[tuple[str, str, int]] = []
        self.results = results

    def __call__(
        self, a: ArenaPlayer, b: ArenaPlayer, config: ArenaConfig
    ) -> MatchOutcome:
        """Fabricate `config.games` games between `a` and `b`, noting the seed."""
        self.calls.append((a.player.id, b.player.id, config.seed))
        pattern = (
            RESULTS if self.results is None else self.results(a.player.id, b.player.id)
        )
        records: list[GameRecord] = []
        for pair in range(config.pairs):
            pair_id = f"{a.player.id}:{b.player.id}:{pair:05d}"
            for seat in (0, 1):
                game = 2 * pair + seat
                white, black = (
                    (a.player.id, b.player.id)
                    if seat == 0
                    else (b.player.id, a.player.id)
                )
                records.append(
                    GameRecord(
                        white=white,
                        black=black,
                        result=pattern[game % len(pattern)],
                        book_id=config.book_id,
                        pair_id=pair_id,
                    )
                )
        draws = sum(1 for record in records if record.is_draw)
        summary = ArenaSummary(
            games=len(records),
            decisive=len(records) - draws,
            draws=draws,
            total_plies=PLIES_PER_GAME * len(records),
            shortest_plies=PLIES_PER_GAME,
            longest_plies=PLIES_PER_GAME,
        )
        return MatchOutcome(tuple(records), summary)


def install(
    monkeypatch: pytest.MonkeyPatch,
    count: int,
    *,
    ids: Sequence[str] | None = None,
    results: Callable[[str, str], ResultPattern] | None = None,
) -> tuple[MatchRecorder, list[int]]:
    """Wire the driver to `count` fake rungs and a recording ``play_match``.

    Returns the recorder and the ``every_n`` values the patched
    :func:`~komugi.eval.arena.ladder_players` was asked for, so a test can check
    the stride is passed through rather than dropped. `ids` renames the rungs
    (:func:`fake_players`) and `results` shapes the games
    (:class:`MatchRecorder`); both default to the suite's rising ladder.
    """
    strides: list[int] = []

    def players(
        directory: str | os.PathLike[str], every_n: int = 1, device: str = "cpu"
    ) -> tuple[ArenaPlayer, ...]:
        """Stand in for the checkpoint-directory scan."""
        del directory, device
        strides.append(every_n)
        return fake_players(count, ids)

    recorder = MatchRecorder(results)
    monkeypatch.setattr(ladder, "ladder_players", players)
    monkeypatch.setattr(ladder, "play_match", recorder)
    return recorder, strides


def config(
    *,
    games_per_matchup: int = GAMES,
    seed: int = 7,
    draw_score: float = DRAW_SCORE,
    every_n: int = 1,
    anchor: str | None = None,
    adjacency: int = 1,
) -> ladder.LadderConfig:
    """Return a :class:`~komugi.eval.ladder.LadderConfig` at the suite's defaults."""
    return ladder.LadderConfig(
        games_per_matchup=games_per_matchup,
        seed=seed,
        draw_score=draw_score,
        every_n=every_n,
        anchor=anchor,
        adjacency=adjacency,
    )


# ---------------------------------------------------------------------------
# configuration refusals
# ---------------------------------------------------------------------------


def test_draw_score_has_no_default() -> None:
    """Contempt is a decision the caller has to make (design review D3)."""
    with pytest.raises(TypeError):
        ladder.LadderConfig(games_per_matchup=2, seed=0)  # type: ignore[call-arg]


@pytest.mark.parametrize("adjacency", [0, -1])
def test_adjacency_below_one_is_refused(adjacency: int) -> None:
    """A ladder whose rungs meet nobody is refused at construction."""
    with pytest.raises(ValueError, match="adjacency must be >= 1"):
        config(adjacency=adjacency)


def test_the_arena_rules_are_inherited_not_restated() -> None:
    """Odd game counts and out-of-range contempt fail in the ladder's config.

    Checked here rather than only at the first ``play_match``: a ladder run
    loads a directory of checkpoints before it plays anything, and discovering
    that the game count was odd after twenty torch loads is a slow way to learn
    it. The rule itself is still the arena's -- ``LadderConfig`` validates by
    constructing the ``ArenaConfig`` its matches will carry.
    """
    with pytest.raises(ValueError, match="even"):
        config(games_per_matchup=3)
    with pytest.raises(ValueError, match="draw_score"):
        config(draw_score=1.5)
    with pytest.raises(ValueError, match="every_n"):
        config(every_n=0)
    with pytest.raises(ValueError, match="anchor"):
        config(anchor="  ")


@pytest.mark.parametrize("seed", [-1, -(2**64)])
def test_a_seed_numpy_will_not_take_is_refused_at_construction(seed: int) -> None:
    """The run seed is checked where it is decided, not where it is spent.

    ``LadderConfig`` validates by deriving the ``ArenaConfig`` its matches will
    carry -- but ``ArenaConfig`` does not validate its seed at all, so a run
    seed used to be checked by nothing, anywhere. The second half of this test
    is what that costs without the first: the same value raises out of numpy,
    naming neither ``seed`` nor ``LadderConfig``, from inside the game loop of a
    run an operator queued for hours.
    """
    with pytest.raises(ValueError, match="seed must be usable as numpy entropy"):
        config(seed=seed)
    with pytest.raises(ValueError):
        ladder.matchup_seed(seed, 0)


@pytest.mark.parametrize("games", [0, -2])
def test_a_matchup_of_fewer_than_two_games_is_refused(games: int) -> None:
    """The ladder's own rule, stricter than the arena's, with its own message.

    ``ArenaConfig`` allows ``games=0``: a zero-game match is degenerate but
    coherent for one match. Inherited into a ladder it is not -- every matchup
    plays nothing, the pool is empty, and the run dies inside ``fit_elo`` with
    "cannot fit Elo to an empty match set", which reads as "the ladder played
    games and they vanished" rather than "the game count was zero". This is the
    case validation-by-derivation cannot cover, so it is stated here.
    """
    with pytest.raises(ValueError, match="games_per_matchup must be >= 2"):
        config(games_per_matchup=games)
    # ...and the boundary the rule is drawn at is a playable ladder:
    assert config(games_per_matchup=2).games_per_matchup == 2


def test_fewer_than_two_players_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """One checkpoint is not a ladder: there is nothing to rate it against."""
    install(monkeypatch, 1)
    with pytest.raises(ValueError, match="at least 2"):
        ladder.run_ladder("nowhere", config())

    install(monkeypatch, 0)
    with pytest.raises(ValueError, match="at least 2"):
        ladder.run_ladder("nowhere", config())


def test_an_off_ladder_anchor_is_refused_before_any_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A misspelled anchor fails before the games, not after them."""
    recorder, _ = install(monkeypatch, 3)
    with pytest.raises(ValueError, match="not on this ladder"):
        ladder.run_ladder("nowhere", config(anchor="rung-9"))
    assert recorder.calls == []


def test_two_rungs_sharing_an_id_are_refused_before_any_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Duplicate ids are a directory fault, and they are caught in the pre-flight.

    ``ladder_players`` names entrants after their step, and a campaign that
    crashed between a checkpoint's publish and the supersede that unlinks the
    old file leaves two files for one step -- both listed, both named
    ``ckpt-200``. Without this check the anchor check passes, every earlier
    matchup is played (hours of games at the production budget), and then
    ``play_match`` refuses a rung against itself and the run's accumulated
    records are discarded with the stack frame. ``recorder.calls == []`` is the
    assertion that matters here: nothing was paid for.
    """
    recorder, _ = install(monkeypatch, 3, ids=("rung-0", "rung-1", "rung-1"))
    with pytest.raises(ValueError, match="share an id"):
        ladder.run_ladder("nowhere", config())
    assert recorder.calls == []


# ---------------------------------------------------------------------------
# the schedule
# ---------------------------------------------------------------------------


def test_the_chain_is_adjacent_rungs_in_order() -> None:
    """``adjacency=1`` is the pure chain; higher values *append* forward links."""
    assert ladder.matchups(4, 1) == ((0, 1), (1, 2), (2, 3))
    assert ladder.matchups(4, 2) == ((0, 1), (1, 2), (2, 3), (0, 2), (1, 3))
    assert ladder.matchups(2, 5) == ((0, 1),)
    assert ladder.matchups(1, 1) == ()
    assert ladder.matchups(0, 1) == ()


def test_a_wider_adjacency_appends_matchups_without_reindexing_them() -> None:
    """The schedule at ``k`` is a *prefix* of the schedule at ``k + 1``.

    The matchup index is the seed (:func:`~komugi.eval.ladder.matchup_seed`), so
    an index that moved when `adjacency` changed would mean the same two
    checkpoints played *different* games at the wider setting -- on a different
    Gumbel stream, with both reports looking exactly as they did before. That is
    precisely the comparison an operator makes when a chain result looks wrong
    and they widen the ladder to look again, so the pairs keep their indices and
    with them their entropy. A rung-major schedule interleaves the new links
    into the chain instead, and only index 0 survives it.
    """
    for count in (2, 4, 5, 8):
        for adjacency in range(1, count + 2):
            narrower = ladder.matchups(count, adjacency)
            wider = ladder.matchups(count, adjacency + 1)
            assert wider[: len(narrower)] == narrower

    def seeds_by_pair(adjacency: int) -> dict[tuple[int, int], int]:
        """Map each rung pair to the seed it would be played on."""
        return {
            pair: ladder.matchup_seed(7, index)
            for index, pair in enumerate(ladder.matchups(5, adjacency))
        }

    chain, doubled = seeds_by_pair(1), seeds_by_pair(3)
    assert len(chain) == 4, "the chain has matchups to compare, or this proves nothing"
    for pair, seed in chain.items():
        assert doubled[pair] == seed, f"{pair} was re-seeded by a wider adjacency"


def test_a_huge_adjacency_is_the_round_robin_and_costs_only_its_pairs() -> None:
    """``adjacency`` past the ladder's reach is bounded, not looped over.

    The module docstring invites "``adjacency`` = the whole field" as the way to
    ask for a round robin, and an operator who does not want to count
    checkpoints spells that as a very large number. Looping over `adjacency` and
    discarding the out-of-range pairs would spin ``count * adjacency`` times --
    5e10 iterations here, before the first game, with ``progress`` silent, so
    the run simply looks hung. This test is red by *not finishing* without the
    bound.
    """
    assert ladder.matchups(50, 10**9) == ladder.matchups(50, 49)
    assert len(ladder.matchups(50, 10**9)) == 50 * 49 // 2


def test_each_matchup_is_played_exactly_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """One matchup, one match -- the ``pair_id`` collision the arena warns about.

    Two matches of the same two players restart their pair indices at ``00000``
    and collide once pooled, which
    :func:`~komugi.eval.seats.check_paired_books` reports as ``GAME_COUNT``
    violations. The driver must therefore never split a matchup into several
    calls, and the pooled set must audit clean.
    """
    recorder, strides = install(monkeypatch, 4)
    result = ladder.run_ladder("nowhere", config(every_n=3, adjacency=2))

    pairings = [(first, second) for first, second, _ in recorder.calls]
    assert pairings == [
        ("rung-0", "rung-1"),
        ("rung-1", "rung-2"),
        ("rung-2", "rung-3"),
        ("rung-0", "rung-2"),
        ("rung-1", "rung-3"),
    ]
    assert len(set(pairings)) == len(pairings)
    assert strides == [3], "every_n must reach ladder_players unchanged"

    pooled = MatchSet.of(result.records)
    assert check_paired_books(pooled) == ()
    assert len(pooled) == len(pairings) * GAMES
    assert pooled.players() == ("rung-0", "rung-1", "rung-2", "rung-3")


def test_every_matchup_gets_its_own_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    """N matchups, N distinct seeds, none of them the run seed itself.

    One seed across two matchups replays the identical per-ply Gumbel streams
    (the arena's ``ArenaConfig.seed`` key does not name the entrants), so the
    two matches are correlated and a joint fit over them is overconfident. The
    seed never reaches a record, so this recorder is the only place the claim
    is checkable at all.
    """
    recorder, _ = install(monkeypatch, 5)
    ladder.run_ladder("nowhere", config(seed=11, adjacency=2))

    seeds = [seed for _, _, seed in recorder.calls]
    assert len(seeds) == len(ladder.matchups(5, 2)) == 7
    assert len(set(seeds)) == len(seeds)
    assert 11 not in seeds
    assert seeds == [ladder.matchup_seed(11, index) for index in range(len(seeds))]


def test_matchup_seeds_are_pure_and_distinct() -> None:
    """Asking twice answers the same; asking for another index does not.

    The spread below is *observed* distinctness at one run seed over 64 indices,
    not injectivity: ``generate_state`` reduces a 128-bit sequence to one
    ``uint64``, so distinctness is a birthday argument (~n**2 / 2**65) and no
    test can be a proof of it. The docstrings say the same.
    """
    assert ladder.matchup_seed(3, 5) == ladder.matchup_seed(3, 5)
    assert ladder.matchup_seed(3, 5) != ladder.matchup_seed(3, 6)
    assert ladder.matchup_seed(3, 5) != ladder.matchup_seed(4, 5)
    spread = {ladder.matchup_seed(99, index) for index in range(64)}
    assert len(spread) == 64


def test_the_matchup_seed_derivation_is_pinned_from_outside_itself() -> None:
    """The exact seed each ``(run seed, index)`` gets, spelled independently.

    Every other seed assertion in this suite recomputes its expectation *through*
    :func:`~komugi.eval.ladder.matchup_seed`, so the derivation is pinned to
    nothing: renaming the whole stream -- swapping the run seed for the index,
    or shifting the index by one -- keeps every one of them green while silently
    re-seeding every ladder run ever compared to another. Two runs of the same
    directory at the same seed would then disagree across a release, which is
    the one thing the seed exists to prevent.

    So the expectation is written out here rather than called:
    ``SeedSequence(entropy=run_seed, spawn_key=(index,))`` reduced to one
    ``uint64``, which is the derivation the docstrings claim. (Not a frozen
    integer literal: numpy owns that hash, and pinning its output would pin a
    numpy version rather than this module's composition of it.)
    """
    for run_seed in (0, 7, 99):
        for index in range(4):
            expected = int(
                np.random.SeedSequence(
                    entropy=run_seed, spawn_key=(index,)
                ).generate_state(1, dtype=np.uint64)[0]
            )
            assert ladder.matchup_seed(run_seed, index) == expected

    # The two renamings that survive every other assertion, refused explicitly:
    assert ladder.matchup_seed(7, 2) != ladder.matchup_seed(2, 7)
    assert ladder.matchup_seed(7, 2) != ladder.matchup_seed(7, 3)


# ---------------------------------------------------------------------------
# what comes back
# ---------------------------------------------------------------------------


def test_summaries_account_for_the_records(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each per-matchup summary counts the games that matchup actually produced."""
    install(monkeypatch, 4)
    result = ladder.run_ladder("nowhere", config())

    assert [summary.ids for summary in result.per_matchup] == [
        ("rung-0", "rung-1"),
        ("rung-1", "rung-2"),
        ("rung-2", "rung-3"),
    ]
    assert sum(summary.games for summary in result.per_matchup) == len(result.records)
    for summary in result.per_matchup:
        played = [
            record
            for record in result.records
            if {record.white, record.black} == set(summary.ids)
        ]
        draws = sum(1 for record in played if record.is_draw)
        assert summary.games == len(played) == GAMES
        assert summary.draws == draws
        assert summary.decisive == len(played) - draws
        assert summary.decisive + summary.draws == summary.games
        assert summary.mean_plies == PLIES_PER_GAME


def test_progress_fires_once_per_completed_matchup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The only thing a multi-hour run says while it runs."""
    install(monkeypatch, 4)
    lines: list[str] = []
    result = ladder.run_ladder("nowhere", config(), progress=lines.append)

    assert len(lines) == len(result.per_matchup) == 3
    for index, (line, summary) in enumerate(
        zip(lines, result.per_matchup, strict=True)
    ):
        assert line.startswith(f"[{index + 1}/3] ")
        assert summary.line() in line
        assert "\n" not in line
        assert summary.ids[0] in line and summary.ids[1] in line
        assert f"{summary.draws} draws" in line


def test_the_fit_is_anchored_at_the_oldest_rung(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ``None`` anchor means rung zero, and the fit is returned verbatim."""
    install(monkeypatch, 4)
    result = ladder.run_ladder("nowhere", config())

    assert result.players == ("rung-0", "rung-1", "rung-2", "rung-3")
    assert set(result.ratings) == set(result.players)
    assert result.ratings["rung-0"] == 0.0
    assert result.ratings is result.fit.ratings
    assert result.fit.converged
    assert result.fit.draw_param > 0.0
    # The fabricated pattern favours the younger rung of every matchup, so the
    # ladder rises: a driver that fit the chain backwards would fail here.
    ratings = [result.ratings[player] for player in result.players]
    assert ratings == sorted(ratings)


def test_an_explicit_anchor_moves_the_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """Anchoring elsewhere shifts the whole table without reshaping it."""
    install(monkeypatch, 3)
    default = ladder.run_ladder("nowhere", config())
    shifted = ladder.run_ladder("nowhere", config(anchor="rung-2"))

    assert shifted.ratings["rung-2"] == 0.0
    assert shifted.anchor == "rung-2" and default.anchor == "rung-0"
    offset = default.ratings["rung-2"]
    for player in default.players:
        # Not exact: both fits stop at the same log-scale tolerance (1e-6),
        # which is a fraction of an Elo point once scaled.
        assert shifted.ratings[player] == pytest.approx(
            default.ratings[player] - offset, abs=0.01
        )


# ---------------------------------------------------------------------------
# the report
# ---------------------------------------------------------------------------


def player_rows(report: str) -> list[str]:
    """Return the report's player-table rows, header excluded."""
    table = report.split("players\n", 1)[1].split("\n\nmatchups\n", 1)[0]
    return table.splitlines()[1:]


def test_the_report_lists_every_player_once_in_ladder_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One row per rung, oldest first -- **not** sorted by rating.

    The ladder answers "did this run get stronger", which is a question about
    reading a column top to bottom; a rating-sorted table answers a different
    one and hides the case worth seeing (a middle checkpoint that outrates its
    successors). The ids here have no substring relations, so the occurrence
    counts below mean what they say.
    """
    install(monkeypatch, 4)
    result = ladder.run_ladder("nowhere", config())
    report = result.report()

    rows = player_rows(report)
    assert [row.split()[0] for row in rows] == list(result.players)
    table = "\n".join(rows)
    for player in result.players:
        assert table.count(player) == 1
    for row, player in zip(rows, result.players, strict=True):
        assert row.split()[1] == f"{result.ratings[player]:.1f}"

    assert report.startswith("ladder: 4 players, 3 matchups, 12 games\n")
    assert "anchor rung-0 at 0.0 Elo" in report
    assert "converged" in report and "NOT CONVERGED" not in report


def test_the_report_carries_the_matchup_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The matchup section states decisive and draw counts per rung pair."""
    install(monkeypatch, 3)
    result = ladder.run_ladder("nowhere", config())
    section = result.report().split("\n\nmatchups\n", 1)[1]

    lines = section.strip().splitlines()
    assert lines[0].split() == ["matchup", "games", "decisive", "draws", "mean", "plies"]
    assert len(lines) == 1 + len(result.per_matchup)
    for line, summary in zip(lines[1:], result.per_matchup, strict=True):
        assert line.startswith(f"{summary.ids[0]} vs {summary.ids[1]}")
        assert line.split()[-4:] == [
            str(summary.games),
            str(summary.decisive),
            str(summary.draws),
            f"{summary.mean_plies:.1f}",
        ]


def test_the_report_is_plain_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """``report()`` returns a string; where it goes is the caller's decision."""
    install(monkeypatch, 3)
    report = ladder.run_ladder("nowhere", config()).report()
    assert isinstance(report, str)
    assert report.endswith("\n")
    assert all(not line.endswith(" ") for line in report.splitlines())


def test_results_are_stated_from_whites_perspective_in_the_fake() -> None:
    """The recorder's own pattern, pinned: this suite's ratings depend on it.

    Not a test of the driver -- a test of the fixture the driver's assertions
    are read against. ``RESULTS`` is by *game index*, and the seating alternates
    within a pair, so the younger rung's score is not readable off the constant
    without applying the seating; if either changes, the "ratings rise along the
    chain" assertion above would quietly stop meaning that.
    """
    younger = 0.0
    for game, result in enumerate(RESULTS):
        # Seat 0 is the older rung as white; seat 1 swaps.
        younger += result if game % 2 == 1 else WHITE_WIN - result
    assert younger == 2.5
    assert RESULTS.count(DRAW) == 1


def younger_score(pattern: ResultPattern, games: int = GAMES) -> float:
    """Return what the younger rung scores over `games` played to `pattern`.

    The same seating arithmetic as above, applied to the shaped patterns: their
    docstrings claim 2-2, 4-0 and 3-1, and a fixture whose claim is wrong would
    make every assertion read off it mean something else.
    """
    return sum(
        (
            pattern[game % len(pattern)]
            if game % 2 == 1
            else WHITE_WIN - pattern[game % len(pattern)]
        )
        for game in range(games)
    )


def test_the_shaped_patterns_are_the_shapes_they_claim() -> None:
    """:data:`BALANCED` splits 2-2, :data:`YOUNGER_SWEEP` is 4-0, the 3-1s are 3-1."""
    assert younger_score(BALANCED) == 2.0
    assert BALANCED.count(DRAW) == 0, "balanced *without* draws is the point"
    assert younger_score(YOUNGER_SWEEP) == 4.0
    assert younger_score(YOUNGER_WINS_3_1) == 3.0
    assert younger_score(OLDER_WINS_3_1) == 1.0


def test_the_report_names_the_configured_anchor_on_a_balanced_ladder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every rating is exactly 0.0 -- and the anchor label still says ``rung-1``.

    A *balanced* ladder, not merely an all-draws one, fits every rung to exactly
    zero: each player scores half its games, so ``elo.py``'s MM step divides a
    number by itself and every ``theta`` is ``+0.0``. A report that read its
    anchor back off the ratings would therefore name the **oldest** rung here
    whatever was configured -- a false statement about the run's configuration,
    in the one artifact a human reads, on a fit that *converged*, with nothing
    else in the report to contradict it. Two near-identical checkpoints splitting
    their games 2-2 is not a contrived ladder; it is a plateaued campaign.
    """
    install(monkeypatch, 2, results=lambda a, b: BALANCED)
    result = ladder.run_ladder("nowhere", config(anchor="rung-1"))

    assert result.anchor == "rung-1"
    assert result.ratings == {"rung-0": 0.0, "rung-1": 0.0}
    assert result.fit.converged

    report = result.report()
    assert "anchor rung-1 at 0.0 Elo" in report
    assert "anchor rung-0" not in report
    # ...and the table marks the gauge, because "0.0" cannot: a rung at 0.04
    # prints as 0.0 too, and here every row does.
    rows = player_rows(report)
    assert [row.split() for row in rows] == [
        ["rung-0", "0.0"],
        ["rung-1", "0.0", "anchor"],
    ]
    assert "NOT CONVERGED" not in report

    # The default anchor is labelled from the same stored string, not guessed:
    default = ladder.run_ladder("nowhere", config())
    assert default.anchor == "rung-0"
    assert "anchor rung-0 at 0.0 Elo" in default.report()
    assert player_rows(default.report())[0].split() == ["rung-0", "0.0", "anchor"]


def test_a_ladder_nobody_can_rate_says_so_on_the_fit_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rung that loses every game is clamped, and the report leads with it.

    ``rung-0`` scoring 0% is the unidentified case ``elo.py``'s identifiability
    note describes -- and the normal early-training one, where the first
    checkpoint is randomly initialised. The likelihood is unbounded, ``theta``
    runs to the clamp, and the returned table reads ``+6949`` for a ten-step
    improvement. It converges in *few* iterations while doing so, so an
    iteration count is not a convergence test; only ``fit.converged`` is, and
    the fit line has to be read off it.
    """
    install(monkeypatch, 2, results=lambda a, b: YOUNGER_SWEEP)
    result = ladder.run_ladder("nowhere", config())

    assert not result.fit.converged
    assert result.fit.iterations < 10000, "the clamp path converges *quickly*"
    assert result.ratings["rung-0"] == 0.0
    assert result.ratings["rung-1"] > 6000.0, "a clamp, not a measurement"

    lines = result.report().splitlines()
    fit_line = next(line for line in lines if line.startswith("fit: "))
    assert "NOT CONVERGED" in fit_line
    assert f"in {result.fit.iterations} iterations" in fit_line
    # And it is not left in the tail of a line: the banner is its own line,
    # above the ratings it disqualifies.
    assert lines[2].startswith("!! NOT CONVERGED")
    assert "not identified" in lines[2]


def test_the_report_keeps_ladder_order_when_the_ratings_do_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A middle rung outrating its successor stays in the middle of the table.

    This is the exact case ``report()``'s docstring says the ladder-ordered
    table exists to expose, and the suite's default fixture -- ratings rising
    monotonically along the chain -- cannot tell it apart from a table sorted by
    rating. Here ``rung-1`` beats both its neighbours and ``rung-2`` loses to
    both, so ladder order and rating order are different orders, and only one of
    them answers "did this run get stronger".
    """
    outcomes = {
        ("rung-0", "rung-1"): YOUNGER_WINS_3_1,
        ("rung-1", "rung-2"): OLDER_WINS_3_1,
        ("rung-0", "rung-2"): OLDER_WINS_3_1,
    }
    install(monkeypatch, 3, results=lambda a, b: outcomes[(a, b)])
    result = ladder.run_ladder("nowhere", config(adjacency=2))

    assert result.fit.converged
    ratings = [result.ratings[player] for player in result.players]
    assert ratings[1] > ratings[0] > ratings[2], "the fixture is non-monotone"
    assert ratings != sorted(ratings), "...or the assertion below proves nothing"

    rows = player_rows(result.report())
    assert [row.split()[0] for row in rows] == ["rung-0", "rung-1", "rung-2"]
    assert [row.split()[1] for row in rows] == [f"{value:.1f}" for value in ratings]


def test_the_table_right_justifies_every_column_but_the_first() -> None:
    """The numbers line up under each other; the identifier column does not.

    Every other report assertion in this suite goes through ``.split()``, which
    sees the same cells however they are padded, and ``_table``'s ``rstrip()``
    hides a left-justified last column from the "no trailing space" check too.
    So the alignment -- the whole reason ``_table`` exists rather than a
    ``"  ".join`` -- is only observable against the exact strings.
    """
    lines = ladder._table(("player", "rating"), [("a", "1.0"), ("bbbbbb", "-123.4")])

    assert lines == [
        "player  rating",
        "a" + " " * 10 + "1.0",
        "bbbbbb  -123.4",
    ]
    assert {len(line) for line in lines} == {len("player  rating")}
    # Which is what a reader actually scans: one decimal point, one column.
    assert {line.index(".") for line in lines[1:]} == {len("player  rating") - 2}


def test_a_rating_missing_from_the_fit_raises_rather_than_printing_nan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``report()`` indexes ``ratings``; it does not paper a gap over.

    Unreachable through ``run_ladder`` today -- every rung plays at least its
    chain neighbour, so ``fit_elo`` rates every one of them -- which is the
    argument for making the impossible case *loud* rather than defaulting it: a
    default turns a future invariant break (a schedule change, a partial-result
    path, a hand-built result) into the string ``nan``, or worse a plausible
    ``0.0``, in a column of Elo ratings, with no warning line, in an artifact
    that goes into a commit message. The rest of this package raises; so does
    this. Built by hand because nothing else can build it.
    """
    install(monkeypatch, 2)
    result = ladder.run_ladder("nowhere", config())

    broken = result._replace(ratings={"rung-0": 0.0})
    with pytest.raises(KeyError, match="rung-1"):
        broken.report()


# ---------------------------------------------------------------------------
# kom-011d §D4.2/§D7.3 -- the ladder and the pool, published together
# ---------------------------------------------------------------------------

_POOL_ANCHOR: str = "v1/iter-0020@7eef51d0"
_POOL_ROSTER: dict[str, str] = {
    _POOL_ANCHOR: "v1",
    "komugi/ckpt-3000@aaaa0001": "komugi",
}


def _manifest(epoch_book_id: str) -> EpochManifest:
    """Return a minimal epoch manifest carrying `epoch_book_id`."""
    return EpochManifest(
        epoch=7,
        anchor_id=_POOL_ANCHOR,
        roster_map=_POOL_ROSTER,
        schedule=(),
        epoch_book_id=epoch_book_id,
        pair_namespace="E007",
    )


@dataclass(frozen=True, slots=True)
class ContemptManifest(EpochManifest):
    """An epoch manifest that carries §D9's per-family ``contempt`` map.

    §6 node 51 pre-registers the field -- *"``contempt`` is a map; each family's
    entry names its scale and neutral value"* -- and T4's ``pool_run.py`` is
    what will write it. Planted here, one release early, because review F7's
    failure scenario is exactly the epoch that carries a **non-neutral** number:
    :func:`~komugi.eval.ladder.require_pool_agreement` compares the ladder to
    :data:`~komugi.eval.ladder.POOL_DRAW_SCORE`, which such an epoch passes
    while §D7.3's actual predicate -- the two series were played at the same
    contempt -- was never evaluated.
    """

    contempt: Mapping[str, Mapping[str, object]] = field(default_factory=dict)


def _manifest_at_contempt(epoch_book_id: str, value: float) -> ContemptManifest:
    """Return a manifest whose pool records having been played at `value`."""
    carried: Mapping[str, Mapping[str, object]] = {
        # §6 node 51's shape: a map, each family's entry naming its scale and
        # its neutral value, external's null.
        "komugi": {"value": value, "scale": "[-1, 1]", "neutral": 0.0},
        "v1": {"value": 0.5, "scale": "[0, 1]", "neutral": 0.5},
        "external": {"value": None, "scale": None, "neutral": None},
    }
    return ContemptManifest(
        epoch=7,
        anchor_id=_POOL_ANCHOR,
        roster_map=_POOL_ROSTER,
        schedule=(),
        epoch_book_id=epoch_book_id,
        pair_namespace="E007",
        contempt=carried,
    )


def test_the_ladder_and_the_pool_agree_on_contempt_and_book(tmp_path: Path) -> None:
    """kom-011d §6 node 15: the two series are comparable, or they are refused.

    §D4.2 publishes the in-family ladder as its own series in its own units, and
    §D7.3 says what makes it *comparable* with the pool's rather than merely
    adjacent to it: the same contempt and the same openings. So a
    :class:`~komugi.eval.ladder.LadderConfig` whose ``draw_score`` or whose
    epoch book differs from the pool manifest's is refused when the two would be
    published together, **naming both values** -- "the ladder is misconfigured"
    is not a sentence anybody can act on.

    *Non-vacuity, three ways.* (i) The agreeing configuration passes, so the
    node is red against a function that refuses everything. (ii) Both messages
    are asserted to carry *both* numbers, so a refusal that names only the
    expectation fails. (iii) The book half is checked twice -- against no book
    at all and against a **different** epoch's book -- because an implementation
    comparing only "is there a book" would pass the first and not the second.

    **The contempt half is checked against the manifest too** (review F7). The
    implementation could only ever compare the ladder to the module constant,
    because :class:`~komugi.eval.pool.EpochManifest` carried no contempt to
    compare with -- so a pool epoch itself played at a non-zero ``draw_score``
    (a T4 driver bug, or a deliberate experiment) passed beside a ``0.0`` ladder
    and §D7.3's real predicate went unevaluated. The last block plants exactly
    that epoch and asserts the refusal names the **manifest's** number, with a
    manifest at the neutral value passing on the same path so the node is red
    against a function that refuses every manifest that carries the field.

    **What this no longer guards** (rev 2): §D5.1 fits ``G_in`` over the pool's
    own records, so a mismatch here can no longer corrupt the gate statistic. It
    makes the two published series incomparable, which is still a refusal.
    """
    # k = 2 because ``GAMES`` is two seat-swapped pairs and pair k plays entry
    # k: a smaller book is refused by ``ArenaConfig`` before this node's
    # subject is reached.
    book = epoch_book(tmp_path, epoch=7, k=2, entries=8, seed=101)
    other = epoch_book(tmp_path / "other", epoch=8, k=2, entries=8, seed=102)
    manifest = _manifest(book.epoch_book_id)
    agreed = ladder.LadderConfig(
        games_per_matchup=GAMES,
        seed=7,
        draw_score=ladder.POOL_DRAW_SCORE,
        book=book,
    )

    ladder.require_pool_agreement(agreed, manifest)

    with pytest.raises(ValueError) as contempt:
        ladder.require_pool_agreement(
            replace(agreed, draw_score=DRAW_SCORE), manifest
        )
    message = str(contempt.value)
    assert repr(DRAW_SCORE) in message and repr(ladder.POOL_DRAW_SCORE) in message

    with pytest.raises(ValueError) as bookless:
        ladder.require_pool_agreement(replace(agreed, book=None), manifest)
    assert book.epoch_book_id in str(bookless.value)
    assert "None" in str(bookless.value)

    with pytest.raises(ValueError) as wrong_book:
        ladder.require_pool_agreement(replace(agreed, book=other), manifest)
    assert other.epoch_book_id in str(wrong_book.value)
    assert book.epoch_book_id in str(wrong_book.value)

    # F7: the epoch that was itself played at another contempt. The ladder is
    # at the pre-registered 0.0 here, so the constant check passes and only the
    # manifest's own number can refuse it.
    played_hot = _manifest_at_contempt(book.epoch_book_id, DRAW_SCORE)
    with pytest.raises(ValueError) as pool_contempt:
        ladder.require_pool_agreement(agreed, played_hot)
    hot_message = str(pool_contempt.value)
    assert repr(DRAW_SCORE) in hot_message and repr(ladder.POOL_DRAW_SCORE) in (
        hot_message
    )
    assert str(played_hot.epoch) in hot_message

    # And the same manifest at the neutral value agrees, so the refusal is
    # about the number and not about the field's presence.
    ladder.require_pool_agreement(
        agreed, _manifest_at_contempt(book.epoch_book_id, ladder.POOL_DRAW_SCORE)
    )
    assert ladder.manifest_contempt(manifest) is None, (
        "a manifest without the field carries no contempt to compare with, and "
        "the pre-registered constant is then the whole check"
    )


def test_the_ladder_threads_its_book_and_namespace_into_every_matchup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pass-through, at the seam it passes through: ``arena_config``.

    A ladder played beside a pool epoch has to be played *on the epoch's own
    openings and under its own namespace* (§D4.2), and the only route from the
    ladder's config to the games is
    :meth:`~komugi.eval.ladder.LadderConfig.arena_config`. So both halves are
    asserted there and again through :func:`~komugi.eval.ladder.run_ladder`, on
    the config every matchup was actually handed.

    *Non-vacuity:* the default configuration's ``arena_config`` is asserted to
    be **today's** -- no book, no namespace, :data:`~komugi.eval.arena.EMPTY_BOOK`
    -- so an implementation that always threaded something fails here as well as
    at node 60.
    """
    book = epoch_book(tmp_path, epoch=7, k=2, entries=8, seed=111)
    plain = config().arena_config(seed=3)
    assert (plain.book, plain.pair_namespace, plain.book_id) == (
        None,
        "",
        EMPTY_BOOK,
    )

    booked = ladder.LadderConfig(
        games_per_matchup=GAMES,
        seed=7,
        draw_score=ladder.POOL_DRAW_SCORE,
        book=book,
        pair_namespace="E007",
    )
    threaded = booked.arena_config(seed=3)
    assert threaded.book is book
    assert threaded.pair_namespace == "E007"
    assert threaded.book_id == book.epoch_book_id

    recorder, _strides = install(monkeypatch, 3)
    seen: list[ArenaConfig] = []
    real = recorder

    def watching(a: ArenaPlayer, b: ArenaPlayer, cfg: ArenaConfig) -> MatchOutcome:
        """Record the config each matchup was played under."""
        seen.append(cfg)
        return real(a, b, cfg)

    monkeypatch.setattr(ladder, "play_match", watching)
    ladder.run_ladder("nowhere", booked)
    assert len(seen) == 2
    for cfg in seen:
        assert cfg.book is book
        assert cfg.pair_namespace == "E007"
        assert cfg.book_id == book.epoch_book_id
