"""The ladder against reality: three saved checkpoints, real games, one fit.

The unit suite next door cuts at :func:`~komugi.eval.arena.play_match` and tests
the schedule. This one does not cut anywhere: a directory of real checkpoints is
written to disk, loaded back through
:func:`~komugi.eval.arena.ladder_players`'s own factories, played through the
real engine and the real Gumbel search, pooled, audited and fit. What it
certifies is the seam the unit suite cannot -- that the ids the driver reports
are the ids on disk, that the records real matches produce survive pooling into
one auditable :class:`~komugi.eval.records.MatchSet`, and that
:func:`~komugi.eval.elo.fit_elo` accepts them with no translation.

The one thing dialled down, and why
====================================
:func:`~komugi.eval.arena.ladder_players` hands out entrants at the
**production** budget (32 candidates, 256 simulations) and
:func:`~komugi.eval.ladder.run_ladder` has no knob to lower it -- deliberately:
a ladder is a measurement, and a driver that let the caller quietly rate a
cheaper search than self-play plays would make its numbers incomparable across
runs. So the budget is lowered the only way it can be, by wrapping the entrant
factory (``dataclasses.replace``, which the arena's own docstring names as the
way to rebudget a player) before the driver sees it. Everything else is real:
the checkpoints, the loading, the nets, the games, the pairing, the seeds, the
pooling and the fit. Four full-budget games at the engine's 256-turn cap would
be minutes of CI to certify nothing this does not.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

pytest.importorskip("torch")
pytest.importorskip("komugi.net")
pytest.importorskip("komugi.selfplay")

from komugi.campaign import factory
from komugi.eval import arena, ladder
from komugi.eval.arena import ArenaPlayer
from komugi.eval.records import VALID_RESULTS, GameRecord, MatchSet, Player
from komugi.eval.seats import check_paired_books
from komugi.net import KomugiNet
from komugi.selfplay.inference import NetEval
from komugi.train.checkpoint import save_checkpoint

DRAW_SCORE: float = 0.20
"""``DESIGN.html`` section 3.3's contempt, and ``SelfPlayConfig``'s default."""

STEPS: tuple[int, ...] = (0, 10, 20)
"""The saved checkpoint steps: three rungs, so the chain has two matchups."""

GAMES: int = 2
"""One seat-swapped pair per matchup -- the smallest match the arena will play."""

TINY_M: int = 2
TINY_N: int = 2
"""The budget the real-net arena test uses; see the module docstring."""


def write_checkpoints(directory: Path) -> None:
    """Save one small-net checkpoint per step in :data:`STEPS`.

    The same weights every time: this suite is not measuring a training curve,
    it is measuring that three checkpoints on disk become three rated rungs.
    Identical nets also make the games a fair coin at the search's noise, which
    is what keeps the assertions below about *structure* rather than strength.
    """
    state = KomugiNet(factory.SMALL_CONFIG).state_dict()
    for step in STEPS:
        save_checkpoint(directory, step, state, {}, {"step": step})


def install_tiny_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the driver's entrants cheap without faking where they come from."""
    real = arena.ladder_players

    def players(
        directory: str | os.PathLike[str], every_n: int = 1, device: str = "cpu"
    ) -> tuple[ArenaPlayer, ...]:
        """Call the real scan, then rebudget each entrant."""
        return tuple(
            replace(entrant, m=TINY_M, n=TINY_N)
            for entrant in real(directory, every_n, device)
        )

    monkeypatch.setattr(ladder, "ladder_players", players)


def test_a_checkpoint_directory_becomes_a_rated_ladder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Three checkpoints in, two real matches played, one Elo table out."""
    monkeypatch.setenv(factory.NET_ENV, factory.SMALL_NET)
    write_checkpoints(tmp_path)
    install_tiny_budget(monkeypatch)

    lines: list[str] = []
    result = ladder.run_ladder(
        tmp_path,
        ladder.LadderConfig(
            games_per_matchup=GAMES, seed=3, draw_score=DRAW_SCORE
        ),
        progress=lines.append,
    )

    # oldest first, and the ids are the steps on disk
    assert result.players == tuple(f"ckpt-{step}" for step in STEPS)
    assert [summary.ids for summary in result.per_matchup] == [
        ("ckpt-0", "ckpt-10"),
        ("ckpt-10", "ckpt-20"),
    ]

    # the games really were played, and pool into one auditable match set
    pooled = MatchSet.of(result.records)
    assert len(pooled) == len(STEPS[1:]) * GAMES == 4
    assert check_paired_books(pooled) == ()
    assert len(pooled.pair_ids()) == 2, "each matchup contributes its own pair id"
    assert pooled.players() == ("ckpt-0", "ckpt-10", "ckpt-20")
    for record in result.records:
        assert record.result in VALID_RESULTS
        assert record.book_id == arena.EMPTY_BOOK
    for summary in result.per_matchup:
        assert summary.games == GAMES
        assert summary.decisive + summary.draws == GAMES
        assert summary.mean_plies > 1.0, "a real game is more than one ply"

    # every rung is rated, anchored at the oldest -- and the anchor the run was
    # fit against is carried on the result, so the report names the rung that
    # was actually pinned rather than whichever one happens to sit at 0.0
    assert set(result.ratings) == set(result.players)
    assert result.ratings["ckpt-0"] == 0.0
    assert result.fit.ratings is result.ratings
    assert result.anchor == "ckpt-0"

    # and the run said something, once per matchup
    assert len(lines) == 2
    assert lines[0].startswith("[1/2] ckpt-0 vs ckpt-10: 2 games")
    assert lines[1].startswith("[2/2] ckpt-10 vs ckpt-20: 2 games")

    report = result.report()
    assert "ladder: 3 players, 2 matchups, 4 games" in report
    assert "anchor ckpt-0 at 0.0 Elo" in report
    for player in result.players:
        assert player in report


def test_a_strided_ladder_rates_the_checkpoints_it_sampled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``every_n`` reaches the scan: two rungs sampled, one match played.

    The cheap half of the end-to-end claim -- one match rather than two -- and
    the one that pins the stride against being dropped on the way through the
    driver, which no unit stub can see happen to a real directory.
    """
    monkeypatch.setenv(factory.NET_ENV, factory.SMALL_NET)
    write_checkpoints(tmp_path)
    install_tiny_budget(monkeypatch)

    result = ladder.run_ladder(
        tmp_path,
        ladder.LadderConfig(
            games_per_matchup=GAMES, seed=5, draw_score=DRAW_SCORE, every_n=2
        ),
    )

    assert result.players == ("ckpt-0", "ckpt-20")
    assert len(result.per_matchup) == 1
    assert len(result.records) == GAMES
    assert check_paired_books(MatchSet.of(result.records)) == ()
    assert result.ratings["ckpt-0"] == 0.0


def test_a_directory_with_one_checkpoint_is_not_a_ladder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Refused against a real directory, before any net is loaded."""
    monkeypatch.setenv(factory.NET_ENV, factory.SMALL_NET)
    save_checkpoint(
        tmp_path, 0, KomugiNet(factory.SMALL_CONFIG).state_dict(), {}, {"step": 0}
    )

    with pytest.raises(ValueError, match="at least 2"):
        ladder.run_ladder(
            tmp_path,
            ladder.LadderConfig(
                games_per_matchup=GAMES, seed=1, draw_score=DRAW_SCORE
            ),
        )
    with pytest.raises(ValueError, match="at least 2"):
        ladder.run_ladder(
            tmp_path / "nothing-here",
            ladder.LadderConfig(
                games_per_matchup=GAMES, seed=1, draw_score=DRAW_SCORE
            ),
        )


# ---------------------------------------------------------------------------
# kom-011d §6 node 60 -- the bookless ladder did not move
# ---------------------------------------------------------------------------

PIN_SEED: int = 3
PIN_STEPS: tuple[int, ...] = (0, 10, 20)

LADDER_NO_BOOK_DIGEST: str = "15b6a1d5e178f76d"
"""The bookless ladder run, hashed: :func:`ladder_digest` over the fixture below.

A stored literal, so that a later change to these games has to be re-pinned by a
human rather than absorbed. **What pins the run to its pre-T2b self** is
:func:`_reference_run` below -- ``run_ladder``'s body as it stood at T2b's base
commit, transcribed, building its ``ArenaConfig`` the way the driver built one
before it had a book or a namespace to thread.

The entrants are pure numpy stubs rather than freshly initialised nets on
purpose: a randomly initialised :class:`~komugi.net.KomugiNet` makes the games a
function of torch's global RNG, which no literal could survive.
"""


class PureStub:
    """A deterministic evaluator: one fixed prior, one indifferent value.

    Pure in ``obs``, row-wise, and seeded from an integer rather than from
    torch, so the games below are a function of the ladder's own seeds and
    nothing else.
    """

    def __init__(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        self._base = rng.standard_normal(factory.NUM_ACTIONS).astype(np.float32)

    def evaluate(
        self, obs: npt.NDArray[np.float32], legal: npt.NDArray[np.bool_]
    ) -> NetEval:
        """Answer one batch, identically row by row."""
        del legal
        rows = int(np.asarray(obs).shape[0])
        wdl = np.asarray(
            np.tile(np.asarray([0.5, 0.0, 0.5], dtype=np.float32), (rows, 1)),
            dtype=np.float32,
        )
        return NetEval(
            policy_logits=np.asarray(
                np.tile(self._base, (rows, 1)), dtype=np.float32
            ),
            wdl=wdl,
            moves_left=np.full((rows,), 8.0, dtype=np.float32),
        )


def rung_factory(step: int) -> Callable[[], PureStub]:
    """Return the evaluator factory for the rung saved at `step`."""

    def build() -> PureStub:
        """Build this rung's stub, freshly, as the arena expects."""
        return PureStub(step + 1)

    return build


def pinned_rungs() -> tuple[ArenaPlayer, ...]:
    """Return the three rungs of the pinned ladder, freshly built."""
    return tuple(
        ArenaPlayer(
            player=Player(id=f"ckpt-{step}", family=arena.LADDER_FAMILY),
            evaluator_factory=rung_factory(step),
            m=1,
            n=0,
        )
        for step in PIN_STEPS
    )


def _reference_run(
    rungs: Sequence[ArenaPlayer], config: ladder.LadderConfig
) -> tuple[GameRecord, ...]:
    """Play the ladder the way ``run_ladder`` played it **before** T2b.

    A transcription of the driver's body at T2b's base commit: the same
    schedule, one :func:`~komugi.eval.ladder.play_match` per matchup, and an
    :class:`~komugi.eval.arena.ArenaConfig` built from the games, the matchup's
    own seed and the contempt -- **and nothing else**, because there was nothing
    else to build it from. It therefore cannot inherit a book or a namespace
    from the config under test, which is what makes it a reference rather than a
    second call to the same code.

    **Which half this pins, stated rather than implied** (review F11). It
    transcribes ``run_ladder``'s *body* and then calls the **post-change**
    ``arena.play_match``, so a mutation inside ``play_match``'s bookless path
    moves both sides of the equality together and only
    :data:`LADDER_NO_BOOK_DIGEST` would catch it. The loop-level equivalence is
    node 53's -- ``test_arena_behavior.py::test_a_match_with_no_book_is_identical_to_todays``,
    whose ``_reference_game`` shares none of that code. The two nodes are
    jointly sufficient and separately insufficient, and this is where that is
    written down.
    """
    records: list[GameRecord] = []
    for index, (first, second) in enumerate(
        ladder.matchups(len(rungs), config.adjacency)
    ):
        outcome = arena.play_match(
            rungs[first],
            rungs[second],
            arena.ArenaConfig(
                games=config.games_per_matchup,
                seed=ladder.matchup_seed(config.seed, index),
                draw_score=config.draw_score,
            ),
        )
        records.extend(outcome.records)
    return tuple(records)


def ladder_digest(
    records: Sequence[GameRecord], moves: Sequence[Sequence[int]]
) -> str:
    """Return a short digest of a ladder's records **and** its move sequences."""
    canonical = "|".join(
        f"{record.white}>{record.black}={record.result}"
        f"@{record.book_id}#{record.pair_id}"
        for record in records
    )
    canonical += "||" + ";".join(
        ",".join(str(int(action)) for action in sequence) for sequence in moves
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def test_run_ladder_without_a_pool_book_is_identical_to_todays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """kom-011d §6 node 60: node 53's property, one level up.

    With no :class:`~komugi.eval.pool_books.PoolBook` and the default
    ``pair_namespace``, ``run_ladder`` plays the games it played before the book
    seam existed: the same schedule, the same per-matchup seeds
    (:func:`~komugi.eval.ladder.matchup_seed`), the same ``pair_id`` s, the same
    :data:`~komugi.eval.arena.EMPTY_BOOK` on every record, and the same moves --
    pinned against :data:`LADDER_NO_BOOK_DIGEST`.

    Two independent pins, because a digest alone would only say "different":
    :func:`_reference_run` is the pre-T2b driver transcribed, and its records
    must be the run's, record for record; the digest then fixes the moves too.
    A ``run_ladder`` that passed its own seed instead of the matchup's, that
    prefixed a namespace by default, or whose ``arena_config`` stopped carrying
    :data:`~komugi.eval.arena.EMPTY_BOOK` fails the first; a change in the games
    themselves fails the second.
    """

    def players(
        directory: str | os.PathLike[str], every_n: int = 1, device: str = "cpu"
    ) -> tuple[ArenaPlayer, ...]:
        """Stand in for the checkpoint scan with the pinned rungs."""
        del directory, every_n, device
        return pinned_rungs()

    monkeypatch.setattr(ladder, "ladder_players", players)

    config = ladder.LadderConfig(
        games_per_matchup=GAMES, seed=PIN_SEED, draw_score=DRAW_SCORE
    )
    assert (config.book, config.pair_namespace) == (None, "")

    result = ladder.run_ladder("nowhere", config)
    assert result.players == tuple(f"ckpt-{step}" for step in PIN_STEPS)
    assert [record.pair_id for record in result.records] == [
        "ckpt-0:ckpt-10:00000",
        "ckpt-0:ckpt-10:00000",
        "ckpt-10:ckpt-20:00000",
        "ckpt-10:ckpt-20:00000",
    ]
    assert {record.book_id for record in result.records} == {arena.EMPTY_BOOK}

    assert _reference_run(pinned_rungs(), config) == result.records, (
        "run_ladder no longer plays the ladder the pre-T2b driver played at "
        "this seed: the book seam has reached a run that supplied no book"
    )

    # The moves, for the digest: the first matchup's two games, replayed on the
    # config the driver handed it.
    rungs = pinned_rungs()
    first, second = ladder.matchups(len(rungs), config.adjacency)[0]
    matchup = config.arena_config(ladder.matchup_seed(config.seed, 0))
    assert matchup.book is None and matchup.pair_namespace == ""
    assert matchup.book_id == arena.EMPTY_BOOK
    moves: list[tuple[int, ...]] = []
    for seat in (0, 1):
        white, black = (
            (rungs[first], rungs[second])
            if seat == 0
            else (rungs[second], rungs[first])
        )
        moves.append(
            arena.play_arena_game(white, black, matchup, pair=0, seat=seat).moves
        )

    digest = ladder_digest(result.records, moves)
    assert digest == LADDER_NO_BOOK_DIGEST, (
        f"the bookless ladder is no longer the run this fixture was captured "
        f"from: digest {digest!r}, expected {LADDER_NO_BOOK_DIGEST!r}. Either "
        f"the book seam reached a run that supplied no book, or the engine, "
        f"the search or the stub changed underneath the pin"
    )
