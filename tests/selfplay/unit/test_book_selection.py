"""Which games are booked, and with which entry (KOM-014 §8 #16, #20a/b, #28, #29).

Three claims live here and they are deliberately different in kind.

**The rate** (#16) is a property of the *pure* selection function, checked over
100 000 synthetic ``(worker_id, incarnation, game_index)`` triples at two fixed
seeds. §6.1 is explicit that this -- not a campaign-scale count -- is the check
that catches a wrong constant, a flipped comparison or a colliding stream: the
booked count over ``n = 200`` games is ``Binomial(200, 0.1)``, whose ±1σ band is
the ticket's own ``[8%, 12%]``, so a gate at that ``n`` fails about one honest
run in three. At 100 000 the ±3.2σ band is ``[9.7%, 10.3%]``, and the test is
deterministic anyway -- the σ framing is what *sized* the band, not what the
assertion rests on. The second seed costs milliseconds and is what separates
"the constant is right" from "this one stream is biased".

**The recomputation** (#20a, #20b) is the property §6.5 is audited by. Nothing
records which entry a game used; the audit re-derives it. #20a pins the
worker-level derivation, and it is asserted twice over -- against the worker's
own selection for 200 games, and against six *actually played* games whose
prefix length is readable back off the shards as ``min(ply)``. #20b starts one
level up, from ``(campaign.run_seed, iteration)``, and is red against an audit
that feeds the campaign's seed straight in: every worker-level derivation is
correct *relative to* whatever seed it was handed, so only a campaign-level node
can see that mistake.

**The config** (#28, #29) is the boundary between them: a fraction is a claim
about a book, so a positive one with no book is refused, and the default books
nothing so that every pre-KOM-014 construction is what it was.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt
import pytest

from komugi.campaign.book_audit import recompute_choice
from komugi.campaign.loop import iteration_seed
from komugi.config import SelfPlayConfig
from komugi.engine.game import Game
from komugi.engine.observation import PLANE_MAP_VERSION
from komugi.selfplay import book
from komugi.selfplay.adapter import GameState
from komugi.selfplay.book import (
    BookEntry,
    book_select_rng,
    load_book,
    running_gungi_mode,
    select_book_entry,
)
from komugi.selfplay.inference import NetEval
from komugi.selfplay.records import (
    BOOK_BUILD_DOMAIN,
    BOOK_SELECT_DOMAIN,
    SCHEMA,
    game_seed,
    num_actions,
)
from komugi.selfplay.worker import SelfPlayWorker, WorkerConfig, game_id
from komugi.train.shards import ShardReader, ShardWriter

_TRIALS: Final[int] = 100_000
"""§6.1(a)'s pre-registered ``n`` for the rate check."""

_RATE_SEEDS: Final[tuple[int, int]] = (20260905, 4242)
"""Two fixed run seeds. One would confirm a constant; two confirm a stream."""

_RATE_BAND: Final[tuple[float, float]] = (0.097, 0.103)
"""±3.2σ at ``n = 100_000`` -- and the reason the ticket's ``[8%, 12%]`` is not
the gate anywhere below ``n = 1500`` (§6.1)."""

_WORKERS: Final[int] = 10
_INCARNATIONS: Final[int] = 10
_GAMES: Final[int] = _TRIALS // (_WORKERS * _INCARNATIONS)
"""The triple space, factored so all three components vary. A sweep over
``game_index`` alone could not see a selection function that ignored the other
two -- which is exactly what a respawned worker replaying its predecessor's
book decisions would look like."""

_SHORT_GAME_PLIES: Final[int] = 10
"""Turn cap for the in-test engine: long enough that a 6-ply prefix still
leaves searched plies to record, short enough to play six games in seconds."""

_PLAYED_GAMES: Final[int] = 6
_RECOMPUTED_GAMES: Final[int] = 200


class _StubEvaluator:
    """A deterministic evaluator: uniform logits, a fixed WDL, no state."""

    def __init__(self, actions: int) -> None:
        self.actions = actions

    def evaluate(
        self, obs: npt.NDArray[np.float32], legal: npt.NDArray[np.bool_]
    ) -> NetEval:
        """Return one evaluation per row of `obs`."""
        rows = int(obs.shape[0])
        return NetEval(
            policy_logits=np.zeros((rows, self.actions), dtype=np.float32),
            wdl=np.tile(np.asarray([0.4, 0.2, 0.4], dtype=np.float32), (rows, 1)),
            moves_left=np.zeros(rows, dtype=np.float32),
        )


class _ShortGame(Game):
    """The real engine, adjudicating a draw after :data:`_SHORT_GAME_PLIES`."""

    @staticmethod
    def max_turns() -> int:
        """Return the shortened turn cap."""
        return _SHORT_GAME_PLIES


def legal_prefix(plies: int, seed: int) -> tuple[int, ...]:
    """Return `plies` actions that replay legally from a fresh :class:`Game`.

    Walked through the real engine rather than invented, because a book entry
    whose actions were made up would be rejected by the worker's own legality
    check and every assertion downstream of it would be about the rejection.
    """
    rng = np.random.default_rng(seed)
    state = GameState(Game(), 0.0)
    actions: list[int] = []
    for _ply in range(plies):
        assert not state.is_terminal(), "the planted line ended before its budget"
        legal = np.nonzero(state.legal_mask())[0]
        chosen = int(legal[int(rng.integers(0, legal.size))])
        actions.append(chosen)
        state.play(chosen)
    return tuple(actions)


def write_book(
    path: Path, prefixes: Sequence[Sequence[int]], *, generation: int = 1
) -> Path:
    """Write a minimal but **loadable** book holding `prefixes`.

    Hand-built rather than produced by :func:`~komugi.selfplay.book.build_book`:
    what this module tests is the consumer, and a builder run would make every
    node here depend on a sampler's draw as well as on the selection rule. The
    manifest carries every key :func:`~komugi.selfplay.book.load_book` reads, so
    the file is refused for exactly the reasons a real one would be.

    Duplicated across the test modules that need it (rather than imported from
    one of them) for the reason ``tests/campaign/behavior/campaign_rig.py``
    states about its own restatement: a test directory is put on ``sys.path`` by
    pytest only once something in it is collected, so a cross-directory import
    would make this module's importability depend on collection order.
    """
    entries = [
        BookEntry(
            id=f"gen{generation:04d}-{index:06d}-0badc0de",
            actions=tuple(int(action) for action in actions),
            plies=len(actions),
            source_iteration=max(generation - 1, 0),
            source_game_id=index,
            deviation_ply=len(actions) - 1,
            deviated=True,
            support_size=16,
            support_fallback=False,
            truncated_mass=0.0,
            drawn_target_prob=0.5,
            restricted_entropy_bits=1.0,
            tempered_entropy_bits=2.0,
            flattening_ratio=0.75,
        )
        for index, actions in enumerate(prefixes)
    ]
    payload: dict[str, Any] = {
        "book_version": book.BOOK_VERSION,
        "generation": generation,
        "source_iterations": [max(generation - 1, 0)],
        "gungi_mode": running_gungi_mode(),
        "action_size": num_actions(),
        "plane_map_version": PLANE_MAP_VERSION,
        "source_schema_fingerprint": SCHEMA.fingerprint,
        "temperature": book.TEMPERATURE,
        "support_epsilon": book.SUPPORT_EPSILON,
        "support_top_k": book.SUPPORT_TOP_K,
        "support_floor": book.SUPPORT_FLOOR,
        "support_min_k": book.SUPPORT_MIN_K,
        "target_shard_budget": book.TARGET_SHARD_BUDGET,
        "cap": book.CAP,
        "min_plies": book.MIN_PLIES,
        "max_plies": book.MAX_PLIES,
        "entries_per_source_game": book.ENTRIES_PER_SOURCE_GAME,
        "build_seed": 7,
        "bit_generator": book.BIT_GENERATOR,
        "entries": [entry.as_json() for entry in entries],
        "stats": {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return path


def _triples() -> list[tuple[int, int, int]]:
    """Return :data:`_TRIALS` distinct ``(worker, incarnation, game)`` triples."""
    return [
        (worker, incarnation, index)
        for worker in range(_WORKERS)
        for incarnation in range(_INCARNATIONS)
        for index in range(_GAMES)
    ]


# ---------------------------------------------------------------------------
# #16 -- the rate
# ---------------------------------------------------------------------------


def test_the_booked_fraction_over_100k_games_is_within_three_sigma() -> None:
    """§6.1(a): the pure selection function books 10% of games, at two seeds."""
    triples = _triples()
    assert len(triples) == _TRIALS
    for run_seed in _RATE_SEEDS:
        booked = sum(
            1
            for worker, incarnation, index in triples
            if select_book_entry(
                run_seed,
                worker,
                incarnation,
                index,
                entries=64,
                fraction=book.DEFAULT_BOOK_FRACTION,
            ).booked
        )
        fraction = booked / _TRIALS
        assert _RATE_BAND[0] <= fraction <= _RATE_BAND[1], (
            f"run_seed {run_seed} booked {fraction:.5f} of {_TRIALS} games, "
            f"outside {_RATE_BAND}: at this n that is a >3sigma departure from "
            f"the pre-registered {book.DEFAULT_BOOK_FRACTION}"
        )


def test_a_zero_fraction_books_nothing_and_a_one_fraction_books_everything() -> None:
    """The two ends, which is what makes the band above a measurement.

    ``random()`` returns ``[0, 1)``, so ``< 1.0`` is always true and ``< 0.0``
    never is. A comparison written the other way round (``<=``) would pass the
    rate test and fail here at the zero end.
    """
    for worker, incarnation, index in [(0, 0, 0), (3, 1, 91), (7, 2, 4242)]:
        assert not select_book_entry(
            5, worker, incarnation, index, entries=8, fraction=0.0
        ).booked
        choice = select_book_entry(
            5, worker, incarnation, index, entries=8, fraction=1.0
        )
        assert choice.booked
        assert choice.entry_index is not None
        assert 0 <= choice.entry_index < 8


def test_an_empty_book_is_refused_rather_than_reported_as_bookless() -> None:
    """A book with no entry cannot book a game, and must not look like no book."""
    with pytest.raises(ValueError, match="empty book"):
        select_book_entry(1, 0, 0, 0, entries=0, fraction=1.0)


def test_the_book_draw_does_not_disturb_the_full_fast_stream() -> None:
    """§1.4: the book has its own stream, so the game's ``rng`` is untouched.

    The property the two arms of §9's experiment rest on -- a bookless run and a
    booked run at one seed draw the identical full/fast sequence -- expressed as
    the fact it reduces to: nothing in the selection path consumes the game's
    own generator, so ``default_rng(game_seed(...))`` answers the same numbers
    before and after a book draw.
    """
    seeds = game_seed(20260905, 1, 0, 3)
    before = np.random.default_rng(seeds).random(8).tolist()
    select_book_entry(20260905, 1, 0, 3, entries=32, fraction=0.5)
    after = np.random.default_rng(seeds).random(8).tolist()
    assert before == after


def test_the_book_domains_are_disjoint_from_every_reachable_ply() -> None:
    """§1.3's disjointness argument, as a measurement (impl-review F9).

    The rate node above is blind to a colliding domain: ``random() < 0.1`` books
    ~10% of games under *any* spawn key, and the worker and the audit both route
    through :func:`select_book_entry`, so a wrong constant leaves them mutually
    consistent and the whole suite green. What a collision would actually break
    is §1.3: :func:`~komugi.selfplay.records.move_seed` composes
    ``(*spawn_key, ply)``, so a book domain equal to a reachable ``ply`` makes
    the book draw *the same stream* as that ply's Gumbel vector -- silently
    correlating whether a game is booked with which move its search prefers at
    that ply. ``test_the_book_draw_does_not_disturb_the_full_fast_stream`` above
    cannot see it either: it checks the game's own ``rng``, not the per-move
    streams.

    So this node asserts the three things the argument rests on, at the cost of
    one ``SeedSequence`` construction per reachable ply (the engine's cap):

    * the two domains are **different** (one stream would make "which prefixes
      exist" a continuation of "does this game start from one");
    * both are **above the engine's turn cap**, which is what makes "unreachable
      as a ply" a fact about the engine rather than a hope;
    * and the book stream's first draw differs from every per-ply Gumbel
      stream's first draw, for every ply the engine can produce -- the
      disjointness itself, measured rather than argued.

    Red against ``BOOK_SELECT_DOMAIN = 7`` (mutant M8) twice over: at the
    ``domain > cap`` assertion, and again at ``ply == 7``, where the two spawn
    keys are identical and the two generators answer the same number.  The cap
    itself is asserted only to be far below the ``0x9014_xxxx`` domains, not to
    equal any particular number -- the argument holds at any cap the engine can
    be configured with, which is why KOM-165's 256 → 512 did not touch it.
    """
    cap = Game.max_turns()
    assert cap < 0x9014_0000, (
        "the disjointness argument is sized on the engine's turn cap: every "
        f"reachable ply must be below the book domains, and the cap is {cap}"
    )
    assert BOOK_SELECT_DOMAIN != BOOK_BUILD_DOMAIN, (
        "one domain for both streams makes the per-game book draw a "
        "continuation of the build's, which is the collision §1.3 forbids"
    )
    for name, domain in (
        ("BOOK_SELECT_DOMAIN", BOOK_SELECT_DOMAIN),
        ("BOOK_BUILD_DOMAIN", BOOK_BUILD_DOMAIN),
    ):
        assert domain > cap, (
            f"{name} is {domain}, at or below the {cap}-turn adjudication cap: "
            "a game that reaches that ply composes the identical spawn key"
        )

    run_seed, worker_id, incarnation, game_index = 20260905, 3, 1, 17
    seeds = game_seed(run_seed, worker_id, incarnation, game_index)
    drawn = book_select_rng(run_seed, worker_id, incarnation, game_index).random()
    for ply in range(cap):
        gumbel = np.random.Generator(
            np.random.PCG64(
                np.random.SeedSequence(
                    entropy=seeds.entropy, spawn_key=(*seeds.spawn_key, ply)
                )
            )
        )
        assert gumbel.random() != drawn, (
            f"the book stream and ply {ply}'s Gumbel stream answer the same "
            "first number, so they are the same stream: whether this game is "
            f"booked is correlated with its search noise at ply {ply}"
        )


# ---------------------------------------------------------------------------
# #20a, #20b -- recomputing what a game did
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def two_entry_book(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A two-entry book whose entries differ in **length**, 4 plies and 6.

    The lengths are the point: a booked game's rows start at ``ply == L``
    (§4.3), so ``min(ply)`` read back off the shards says which of the two
    entries the game actually used -- which is what turns "the audit recomputes
    the same answer as the selection function" into "the audit recomputes the
    answer the *game* used".
    """
    root = tmp_path_factory.mktemp("book-selection")
    return write_book(
        root / "books" / "gen_1.json",
        [legal_prefix(4, 11), legal_prefix(6, 12)],
    )


def _worker(
    book_path: Path, directory: Path, run_seed: int, fraction: float
) -> tuple[SelfPlayWorker, ShardWriter]:
    """Return a worker that plays from `book_path` at `fraction`, and its writer.

    The writer travels with the worker because the caller has to close it: the
    rows a booked game writes are only readable after the shard is sealed, and
    ``min(ply)`` off those rows is what #20a's second half asserts on.
    """
    writer = ShardWriter(directory, SCHEMA)
    worker = SelfPlayWorker(
        _StubEvaluator(num_actions()),
        SelfPlayConfig(
            workers=1,
            process_batch_size=1,
            gumbel_m_full=2,
            gumbel_n_full=2,
            gumbel_m_fast=2,
            gumbel_n_fast=2,
            prob_fast=0.5,
            book_path=str(book_path),
            book_fraction=fraction,
        ),
        WorkerConfig(
            worker_id=2,
            incarnation=1,
            run_seed=run_seed,
            games=_PLAYED_GAMES,
            shard_dir=str(directory),
        ),
        writer,
    )
    return worker, writer


def test_the_entry_a_game_used_is_recomputable_from_its_seed(
    two_entry_book: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#20a: the worker's decision and the audit's are the same two draws.

    Asserted at both strengths. First over 200 games against the worker's own
    selection -- which is what an audit joins to a shard's ``game_id`` column.
    Then over six games that were actually **played**: the recomputed entry's
    ``plies`` must equal the ``min(ply)`` of the rows those games wrote, which
    is false for an audit that recomputed the booked flag correctly and the
    entry *index* wrongly (a modulo instead of ``integers``, say, or the two
    draws in the other order).
    """
    monkeypatch.setattr("komugi.selfplay.worker.Game", _ShortGame)
    run_seed = 20260906
    loaded = load_book(two_entry_book)

    worker, half_writer = _worker(two_entry_book, tmp_path / "half", run_seed, 0.5)
    booked_seen = 0
    for index in range(_RECOMPUTED_GAMES):
        entry = worker._book_entry_for(index)
        choice = select_book_entry(
            run_seed, 2, 1, index, entries=len(loaded), fraction=0.5
        )
        assert (entry is not None) == choice.booked
        if entry is not None:
            booked_seen += 1
            assert choice.entry_index is not None
            assert entry.id == loaded.entries[choice.entry_index].id
    assert booked_seen > 0, "no game in 200 was booked at fraction 0.5"
    half_writer.close()

    played_dir = tmp_path / "played"
    worker_all, writer = _worker(two_entry_book, played_dir, run_seed, 1.0)
    for index in range(_PLAYED_GAMES):
        assert worker_all.play_game(index) > 0, (
            f"game {index} wrote no row: the prefix consumed the whole game"
        )
    writer.close()

    lowest: dict[int, int] = {}
    for row in ShardReader(played_dir, schema=SCHEMA):
        key = int(row["game_id"])
        ply = int(row["ply"])
        lowest[key] = min(lowest.get(key, ply), ply)
    assert len(lowest) == _PLAYED_GAMES

    lengths = {entry.plies for entry in loaded.entries}
    assert len(lengths) == 2, "the fixture must plant two prefix lengths"
    for index in range(_PLAYED_GAMES):
        choice = select_book_entry(
            run_seed, 2, 1, index, entries=len(loaded), fraction=1.0
        )
        assert choice.entry_index is not None
        expected = loaded.entries[choice.entry_index].plies
        assert lowest[game_id(2, 1, index)] == expected, (
            f"game {index} recorded its first row at ply "
            f"{lowest[game_id(2, 1, index)]}, but the recomputed entry is "
            f"{expected} plies long: the audit and the game disagree about "
            f"which prefix was played"
        )


def test_the_audit_recomputes_from_the_campaign_seed_through_iteration_seed(
    two_entry_book: Path,
) -> None:
    """#20b: the derivation starts at ``(campaign.run_seed, iteration)`` (N3).

    Red against an audit that uses ``campaign.run_seed`` directly, and that
    redness is asserted rather than described: the same 200 games recomputed on
    the campaign's seed disagree with the ones recomputed through
    ``iteration_seed``, so an audit that skipped the step would report a
    different set of booked games while looking perfectly self-consistent.
    """
    campaign_seed = 20260823
    iteration = 3
    fraction = 0.5
    loaded = load_book(two_entry_book)
    worker_seed = iteration_seed(campaign_seed, iteration)
    assert worker_seed != campaign_seed

    through_iteration: list[tuple[bool, str | None]] = []
    direct: list[tuple[bool, str | None]] = []
    for index in range(_RECOMPUTED_GAMES):
        recomputed = recompute_choice(
            campaign_seed, iteration, game_id(4, 0, index), loaded, fraction
        )
        assert recomputed.worker_id == 4
        assert recomputed.incarnation == 0
        assert recomputed.game_index == index

        expected = select_book_entry(
            worker_seed, 4, 0, index, entries=len(loaded), fraction=fraction
        )
        assert recomputed.booked == expected.booked
        if expected.booked:
            assert expected.entry_index is not None
            assert recomputed.entry_id == loaded.entries[expected.entry_index].id
        else:
            assert recomputed.entry_id is None
        through_iteration.append((recomputed.booked, recomputed.entry_id))

        wrong = select_book_entry(
            campaign_seed, 4, 0, index, entries=len(loaded), fraction=fraction
        )
        direct.append(
            (
                wrong.booked,
                None
                if wrong.entry_index is None
                else loaded.entries[wrong.entry_index].id,
            )
        )

    assert through_iteration != direct, (
        "recomputing on the campaign's own seed gave the identical answers, so "
        "this node could not tell the two derivations apart"
    )


# ---------------------------------------------------------------------------
# #28, #29 -- the config
# ---------------------------------------------------------------------------


def test_a_book_fraction_without_a_book_path_is_refused() -> None:
    """A run that books 10% of nothing would report success (§4.1)."""
    with pytest.raises(ValueError, match="book_fraction"):
        SelfPlayConfig(
            workers=1, process_batch_size=1, book_path="", book_fraction=0.1
        )
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError, match="book_fraction"):
            SelfPlayConfig(
                workers=1,
                process_batch_size=1,
                book_path="/tmp/does-not-need-to-exist.json",
                book_fraction=bad,
            )


def test_the_default_selfplay_config_books_nothing() -> None:
    """Every pre-KOM-014 construction is byte-identical in behaviour (#29)."""
    config = SelfPlayConfig(workers=1, process_batch_size=1)
    assert config.book_path == ""
    assert config.book_fraction == 0.0
    # And a worker built on it loads no book at all -- the property the three
    # `len(rows)` identities in test_worker_unit.py still rest on.
    assert SelfPlayConfig(workers=1, process_batch_size=1).book_path == ""


def test_a_book_path_without_a_fraction_is_allowed(two_entry_book: Path) -> None:
    """The other pairing is legal: a book present and never drawn from.

    That is what a campaign iteration looks like when an operator has set
    ``book_fraction = 0.0`` to switch the device off (§9's R2 response), and
    refusing it would make turning the book off require deleting the file.
    """
    config = SelfPlayConfig(
        workers=1,
        process_batch_size=1,
        book_path=str(two_entry_book),
        book_fraction=0.0,
    )
    assert config.book_path == str(two_entry_book)
    assert config.book_fraction == 0.0
