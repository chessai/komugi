"""§D3 nodes: epoch books, the refusals that keep them frozen, and their eras.

Design: ``docs/design/kom-011d-ladder-pool.md`` (rev 3), §6 nodes 9, 10, 11, 28,
47 and 49a.

Every book here is built by :func:`~komugi.eval.pool_books.build_epoch_book`
from a **real KOM-014 book file** written to disk and read back through
:func:`komugi.selfplay.book.load_book` -- not from a hand-built
:class:`~komugi.eval.pool_books.PoolBook`. That is the point of the module: an
epoch book's only source is a draft book file (§D3.4), and a fixture that
skipped the source would test the dataclass rather than the seam.

The two-epoch record sets are **planted** at exact win/draw/loss counts, in
``test_divergence.py``'s idiom, because a 128-game link is not a thing a test
plays; what is real about them is the ``book_id`` s, which are entry ids taken
off the epoch books built above.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt
import pytest

pytest.importorskip("torch")
pytest.importorskip("komugi.selfplay")

from komugi.campaign import factory
from komugi.engine.game import Game
from komugi.engine.observation import PLANE_MAP_VERSION
from komugi.eval import arena
from komugi.eval.divergence import (
    CONDITION_BOOK_ERA,
    DivergenceResult,
    Verdict,
    WindowKind,
    book_era,
    evaluate_window,
    summarise_matchups,
)
from komugi.eval.elo import fit_elo
from komugi.eval.pool_books import (
    DIGEST_CHARS,
    POOL_BOOK_K,
    POOL_BOOK_VERSION,
    PoolBook,
    PoolBookEntry,
    build_epoch_book,
    entry_id_for,
    epoch_book_id_for,
    load_pool_book,
)
from komugi.eval.records import BLACK_WIN, WHITE_WIN, GameRecord, MatchSet, Player
from komugi.eval.seats import GAME_COUNT, check_paired_books
from komugi.selfplay import book as book_module
from komugi.selfplay.adapter import GameState
from komugi.selfplay.book import BookEntry, load_book, running_gungi_mode
from komugi.selfplay.inference import NetEval
from komugi.selfplay.records import SCHEMA, num_actions

DRAW_SCORE: float = 0.20
"""``DESIGN.html`` section 3.3's contempt, and ``SelfPlayConfig``'s default."""

ENGINE_RULES_VERSION: int = 1
"""§D1.3's rules revision, declared by the caller rather than imported.

``rawmcts_entrant.ENGINE_RULES_VERSION`` is the campaign's copy, and it lives
behind ``v1bridge`` -> ``arena`` -> ``pool_books``; importing it *into*
``pool_books`` would close that cycle, so the builder takes the number as an
argument and this is the test's declaration of it.
"""

SOURCE_PLIES: tuple[int, int] = (book_module.MIN_PLIES, book_module.MAX_PLIES)
"""KOM-014's prefix length range: what a legal source book's entries carry."""


# ---------------------------------------------------------------------------
# fixtures: a real KOM-014 book on disk, and epoch books built from it
# ---------------------------------------------------------------------------


class StubEvaluator:
    """A deterministic batch evaluator: a fixed prior plus a position tilt.

    ``test_arena_unit.py``'s, in miniature. Deterministic in ``obs`` alone, and
    row-wise, so a wave of R positions answers exactly what R scalar calls
    would -- which is what keeps the games below a function of the seed and the
    book prefix and nothing else.
    """

    def __init__(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        self._base = rng.standard_normal(factory.NUM_ACTIONS).astype(np.float32)

    def evaluate(
        self, obs: npt.NDArray[np.float32], legal: npt.NDArray[np.bool_]
    ) -> NetEval:
        """Answer one ``(N, planes, 9, 9)`` batch."""
        del legal
        array = np.asarray(obs, dtype=np.float32)
        rows = int(array.shape[0])
        policy = np.asarray(np.tile(self._base, (rows, 1)), dtype=np.float32)
        wdl = np.asarray(
            np.tile(np.asarray([0.5, 0.0, 0.5], dtype=np.float32), (rows, 1)),
            dtype=np.float32,
        )
        return NetEval(
            policy_logits=policy,
            wdl=wdl,
            moves_left=np.full((rows,), 8.0, dtype=np.float32),
        )


def cheap_entrant(name: str, seed: int) -> arena.ArenaPlayer:
    """An entrant that spends **no** simulations: one root evaluation a move."""
    return arena.ArenaPlayer(
        player=Player(id=name, family="komugi"),
        evaluator_factory=lambda: StubEvaluator(seed),
        m=1,
        n=0,
    )


def random_prefix(rng: np.random.Generator, plies: int) -> tuple[int, ...]:
    """Return `plies` legal actions from a fresh game, drawn uniformly."""
    state = GameState(Game(), 0.0)
    actions: list[int] = []
    for _ply in range(plies):
        legal = np.flatnonzero(state.legal_mask())
        chosen = int(legal[int(rng.integers(0, legal.size))])
        actions.append(chosen)
        state.play(chosen)
    return tuple(actions)


def write_source_book(path: Path, *, entries: int, seed: int) -> Path:
    """Write a **valid** KOM-014 book of `entries` prefixes and return its path.

    Valid in the loader's own terms (``book.py:1111-1198``): the running action
    space and start mode, ``plies`` inside ``[MIN_PLIES, MAX_PLIES]``, and every
    action id in range. It is written through :mod:`json` rather than through
    ``build_book`` because ``build_book`` needs a campaign's shard tree, which
    is several minutes of self-play to produce a file this reads in a
    millisecond.
    """
    rng = np.random.default_rng(seed)
    low, high = SOURCE_PLIES
    items: list[BookEntry] = []
    for index in range(entries):
        plies = int(rng.integers(low, high + 1))
        actions = random_prefix(rng, plies)
        items.append(
            BookEntry(
                id=f"gen0001-{index:06d}-{index:08x}",
                actions=actions,
                plies=plies,
                source_iteration=0,
                source_game_id=index,
                deviation_ply=plies - 1,
                deviated=True,
                support_size=16,
                support_fallback=False,
                truncated_mass=0.0,
                drawn_target_prob=0.5,
                restricted_entropy_bits=1.0,
                tempered_entropy_bits=2.0,
                flattening_ratio=0.75,
            )
        )
    payload = {
        "book_version": book_module.BOOK_VERSION,
        "generation": 1,
        "source_iterations": [0],
        "gungi_mode": running_gungi_mode(),
        "action_size": num_actions(),
        "plane_map_version": PLANE_MAP_VERSION,
        "source_schema_fingerprint": SCHEMA.fingerprint,
        "temperature": book_module.TEMPERATURE,
        "support_epsilon": book_module.SUPPORT_EPSILON,
        "support_top_k": book_module.SUPPORT_TOP_K,
        "support_floor": book_module.SUPPORT_FLOOR,
        "support_min_k": book_module.SUPPORT_MIN_K,
        "target_shard_budget": book_module.TARGET_SHARD_BUDGET,
        "cap": book_module.CAP,
        "min_plies": book_module.MIN_PLIES,
        "max_plies": book_module.MAX_PLIES,
        "entries_per_source_game": book_module.ENTRIES_PER_SOURCE_GAME,
        "build_seed": seed,
        "bit_generator": book_module.BIT_GENERATOR,
        "entries": [entry.as_json() for entry in items],
        "stats": {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return path


def epoch_book(
    tmp_path: Path, *, epoch: int, k: int = 2, entries: int = 8, seed: int = 5
) -> PoolBook:
    """Build one epoch book from a freshly written source book."""
    source = write_source_book(
        tmp_path / f"books/gen_{epoch}.json", entries=entries, seed=seed
    )
    return build_epoch_book(
        source,
        tmp_path / f"pool/books/epoch_{epoch:03d}.json",
        epoch=epoch,
        selection_seed=1000 + epoch,
        engine_rules_version=ENGINE_RULES_VERSION,
        k=k,
    )


# ---------------------------------------------------------------------------
# planted links, test_divergence.py's idiom
# ---------------------------------------------------------------------------

_A: str = "komugi/ckpt-3000@aaaa0001"
_B: str = "komugi/ckpt-6000@aaaa0002"
_ANCHOR: str = "v1/iter-0020@7eef51d0"
_REF_LOW: str = "rawmcts/n100@0f31ab70"
_REF_HIGH: str = "rawmcts/n6400@0f31ab78"

_ROSTER_MAP: dict[str, str] = {
    _A: "komugi",
    _B: "komugi",
    _ANCHOR: "v1",
    _REF_LOW: "rawmcts",
    _REF_HIGH: "rawmcts",
}

_WINDOW: tuple[str, str] = (_A, _B)
_CROSS_PAIRS: int = 64
"""64 seat-swapped pairs is §D2.1's 128 games per matchup."""

_IN_FAMILY_PAIRS: int = 128
"""§D2.2(b)1's 256 games: the link that carries ``G_in`` alone."""

_REPLICATES: int = 20
"""``B`` for the one node here that reaches the bootstrap at all.

Far below §D5.2's pre-registered 2,000: this suite asserts *which* verdict comes
back, never the width of an interval, and the conditions that are its subject
are all evaluated before the resampling starts.
"""


def even_link(
    first: str, second: str, *, book: str, pairs: int = _CROSS_PAIRS
) -> list[GameRecord]:
    """Return an evenly split link: score rate 0.5, every game decisive.

    Even on purpose -- the fit is then identified, every endpoint link clears
    §D5.3's decisive floor, and every score rate sits in the middle of
    condition 6's ``[0.15, 0.85]`` -- so a window built from these can only fail
    on the condition a node plants.
    """
    records: list[GameRecord] = []
    for index in range(pairs):
        pair_id = f"{first}:{second}:{index:05d}"
        first_seat = WHITE_WIN if index % 2 == 0 else BLACK_WIN
        records.append(
            GameRecord(
                white=first,
                black=second,
                result=first_seat,
                book_id=book,
                pair_id=pair_id,
            )
        )
        records.append(
            GameRecord(
                white=second,
                black=first,
                result=WHITE_WIN - first_seat,
                book_id=book,
                pair_id=pair_id,
            )
        )
    return records


def pool_records(*, milestone_book: str, reference_book: str) -> list[GameRecord]:
    """Return a §D2.2-shaped pool: an in-family link, an anchor and a bracket.

    `milestone_book` is the era ``B``'s own links were played in and
    `reference_book` the era ``A``'s were. Passing one string twice is the
    single-era control; passing two is the straddle.
    """
    return [
        *even_link(_A, _B, pairs=_IN_FAMILY_PAIRS, book=milestone_book),
        *even_link(_A, _ANCHOR, book=reference_book),
        *even_link(_A, _REF_LOW, book=reference_book),
        *even_link(_A, _REF_HIGH, book=reference_book),
        *even_link(_B, _ANCHOR, book=milestone_book),
        *even_link(_B, _REF_LOW, book=milestone_book),
        *even_link(_B, _REF_HIGH, book=milestone_book),
    ]


def evaluate(records: Sequence[GameRecord]) -> DivergenceResult:
    """Evaluate §D5's single-epoch gate over `records`."""
    return evaluate_window(
        MatchSet.of(records),
        roster_map=_ROSTER_MAP,
        window=_WINDOW,
        kind=WindowKind.SINGLE,
        bootstrap_seed=20260908,
        replicates=_REPLICATES,
    )


# ---------------------------------------------------------------------------
# §6 node 10
# ---------------------------------------------------------------------------


def test_building_over_an_existing_epoch_book_is_refused(tmp_path: Path) -> None:
    """§D3.3: an epoch book is written once and is thereafter read-only.

    The second build refuses **and leaves the bytes alone**, which is the half
    that matters: stored records already carry the ids inside that file, so an
    overwrite would silently re-define what an ``E007-`` id names while every
    ``MatchSet`` holding one still claimed to know its openings.

    *Non-vacuity:* the first build is asserted to have produced a loadable book
    at that path, and the bytes are compared before and after -- so the node
    fails both against a builder that overwrites and against one that "refuses"
    by having written nothing in the first place.
    """
    source = write_source_book(tmp_path / "books/gen_1.json", entries=8, seed=3)
    target = tmp_path / "pool/books/epoch_007.json"
    first = build_epoch_book(
        source,
        target,
        epoch=7,
        selection_seed=1007,
        engine_rules_version=ENGINE_RULES_VERSION,
        k=2,
    )
    published = target.read_bytes()
    assert load_pool_book(target).epoch_book_id == first.epoch_book_id

    with pytest.raises(FileExistsError, match="already holds an epoch book"):
        build_epoch_book(
            source,
            target,
            epoch=7,
            # A *different* selection, so an overwrite would be visible in the
            # bytes rather than accidentally reproducing them.
            selection_seed=999,
            engine_rules_version=ENGINE_RULES_VERSION,
            k=3,
        )

    assert target.read_bytes() == published, (
        "the refused build changed the published bytes: a book epoch that can "
        "be rewritten is not frozen, and every stored record naming it is now "
        "a claim about openings the file no longer holds"
    )


# ---------------------------------------------------------------------------
# §6 node 47
# ---------------------------------------------------------------------------


def test_a_source_book_with_fewer_than_k_entries_is_refused(tmp_path: Path) -> None:
    """§D3.2 (F16): a source that cannot supply ``K`` is refused, not stretched.

    The source here carries exactly :data:`komugi.selfplay.book.MIN_ENTRIES`
    entries -- a **legal** KOM-014 book, which is the whole point: the refusal
    is not about a broken file, it is about a file that is fine for self-play
    and too thin to be an epoch book. The message names the path, the count and
    ``K``, because "the book is too small" does not say which of the three to
    change.

    *Non-vacuity:* no epoch book is written (a partial file would be worse than
    none -- a reader would load it and play 32 openings where the manifest
    claims 128), and the same source **does** build at a ``k`` it can supply, so
    the node is red against a builder that refuses every source.
    """
    source = write_source_book(
        tmp_path / "books/gen_1.json", entries=book_module.MIN_ENTRIES, seed=11
    )
    target = tmp_path / "pool/books/epoch_007.json"

    with pytest.raises(ValueError) as raised:
        build_epoch_book(
            source,
            target,
            epoch=7,
            selection_seed=1007,
            engine_rules_version=ENGINE_RULES_VERSION,
        )

    message = str(raised.value)
    assert str(source) in message
    assert str(book_module.MIN_ENTRIES) in message
    assert str(POOL_BOOK_K) in message
    assert not target.exists(), (
        "a refused build left an epoch book behind; a reader would load it and "
        "play fewer openings than the epoch claims"
    )

    smaller = build_epoch_book(
        source,
        target,
        epoch=7,
        selection_seed=1007,
        engine_rules_version=ENGINE_RULES_VERSION,
        k=book_module.MIN_ENTRIES,
    )
    assert len(smaller) == book_module.MIN_ENTRIES


# ---------------------------------------------------------------------------
# §6 node 9
# ---------------------------------------------------------------------------


def test_two_epochs_of_records_concatenate_without_a_pair_violation(
    tmp_path: Path,
) -> None:
    """§D2.3: the pair namespace is what lets two epochs pool into one audit.

    The same two entrants play at ``E007`` and again at ``E008`` -- a pool's
    re-link, or simply a matchup that recurs across epochs -- and the
    concatenated :class:`~komugi.eval.records.MatchSet` audits clean.

    *Non-vacuity:* the identical concatenation with ``pair_namespace=""`` on
    **both** matches produces ``GAME_COUNT`` violations, which is the failure
    the namespace exists to prevent (``arena.py:645-654``: pair indices restart
    at ``00000`` every match and the id carries no epoch). Without that half the
    node would pass on an arena that ignored the namespace entirely.
    """
    seventh = epoch_book(tmp_path / "e7", epoch=7, k=1, entries=8, seed=21)
    eighth = epoch_book(tmp_path / "e8", epoch=8, k=1, entries=8, seed=22)

    def play(book: PoolBook, namespace: str) -> arena.MatchOutcome:
        """Play one pair between the same two entrants on `book`."""
        return arena.play_match(
            cheap_entrant("komugi/ckpt-3000@aaaa0001", 3),
            cheap_entrant("v1/iter-0020@7eef51d0", 4),
            arena.ArenaConfig(
                games=2,
                seed=17,
                draw_score=DRAW_SCORE,
                book_id=book.epoch_book_id,
                pair_namespace=namespace,
                book=book,
            ),
        )

    namespaced = MatchSet.of(
        play(seventh, "E007").records + play(eighth, "E008").records
    )
    assert check_paired_books(namespaced) == ()
    assert len(namespaced.pair_ids()) == 2
    assert namespaced.book_ids() == tuple(
        sorted({seventh.entries[0].id, eighth.entries[0].id})
    )

    collided = MatchSet.of(play(seventh, "").records + play(eighth, "").records)
    violations = check_paired_books(collided)
    assert [violation.reason for violation in violations] == [GAME_COUNT], (
        "two epochs of one matchup pooled under one pair id must fail the "
        "audit; if they do not, the namespace is not what is keeping the "
        "clean case clean"
    )
    assert len(collided.pair_ids()) == 1


# ---------------------------------------------------------------------------
# §6 node 11
# ---------------------------------------------------------------------------


def test_a_records_set_mixing_two_book_epochs_is_detected(tmp_path: Path) -> None:
    """§D3.3: a mixed-era record set is visible, and the window is ineligible.

    Two halves, and both are the design's: the ``E<NNN>-`` prefix on every entry
    id makes the mixture detectable by inspection of
    :meth:`~komugi.eval.records.MatchSet.book_ids` alone, and the pool's own
    validator (:func:`komugi.eval.divergence.evaluate_window`) refuses the
    window that straddles rather than reporting a gain measured across an
    opening-set change.

    *Non-vacuity:* the ids are real -- they come off two epoch books built here
    -- and a single-era set built from the same helper is asserted to *not*
    trip the era condition in :func:`test_a_book_epoch_change_makes_the_straddling_window_ineligible`.
    """
    first = epoch_book(tmp_path / "e1", epoch=1, k=2, entries=8, seed=31)
    seventh = epoch_book(tmp_path / "e7", epoch=7, k=2, entries=8, seed=32)

    mixed = MatchSet.of(
        pool_records(
            milestone_book=seventh.entries[0].id,
            reference_book=first.entries[0].id,
        )
    )
    eras = {book_era(book_id) for book_id in mixed.book_ids()}
    assert eras == {"E001", "E007"}, (
        f"the record set's book ids are {mixed.book_ids()}, whose eras are "
        f"{sorted(eras)}: an epoch book's entry ids must carry the epoch, or a "
        f"stored MatchSet can mix two book epochs undetectably"
    )
    assert first.entries[0].id != seventh.entries[0].id

    result = evaluate(mixed.records)
    assert result.verdict is Verdict.NOT_EVALUABLE
    assert result.condition == CONDITION_BOOK_ERA


# ---------------------------------------------------------------------------
# §6 node 28
# ---------------------------------------------------------------------------


def test_a_book_epoch_change_makes_the_straddling_window_ineligible(
    tmp_path: Path,
) -> None:
    """Nodes 11 and 26's third case composed, over a two-epoch record set.

    A book era boundary does not invalidate the earlier games -- they stay in
    the joint fit, and the fit below converges over both eras -- it invalidates
    *pooling them into one comparison as though the openings were the same*. So
    the window that straddles is excluded from the gate, and the same window
    played entirely inside one era is not.

    *Non-vacuity:* the control is the same records with one era's id
    everywhere. Without it the node would pass against an implementation that
    reported ``not_evaluable`` for every window it was ever handed.
    """
    seventh = epoch_book(tmp_path / "e7", epoch=7, k=2, entries=8, seed=41)
    eighth = epoch_book(tmp_path / "e8", epoch=8, k=2, entries=8, seed=42)

    straddling = pool_records(
        milestone_book=eighth.entries[0].id, reference_book=seventh.entries[0].id
    )
    inside = pool_records(
        milestone_book=eighth.entries[0].id, reference_book=eighth.entries[1].id
    )

    straddled = evaluate(straddling)
    assert straddled.verdict is Verdict.NOT_EVALUABLE
    assert straddled.condition == CONDITION_BOOK_ERA

    control = evaluate(inside)
    assert control.verdict is not Verdict.NOT_EVALUABLE, (
        f"the single-era control is not evaluable either, on "
        f"{control.condition!r}: the era condition cannot be what the "
        f"straddling case failed on if a window inside one era fails too"
    )
    # Two entries of one epoch book are one era -- which is exactly what makes
    # "every pair has its own opening" compatible with "one window, one era".
    assert book_era(eighth.entries[0].id) == book_era(eighth.entries[1].id) == "E008"


# ---------------------------------------------------------------------------
# §6 node 49a
# ---------------------------------------------------------------------------


def test_reused_frozen_records_carry_their_original_book_epoch(
    tmp_path: Path,
) -> None:
    """§D2.2(a): re-used reference records keep the era they were played in.

    The frozen sub-pool is played **once, ever**, so from the second era on
    every joint fit mixes era-1 openings (all reference x reference links) with
    the current era's (all milestone links). That is permitted and declared, not
    a straddle: the fit is over games between the same entrants and the Davidson
    likelihood carries no book term. What the design requires instead is that
    every matchup carry *its own* era, which is a property of the records and
    needs no side table to stay true.

    *Non-vacuity:* the two eras are asserted to be different ids from different
    books, and the reference link's era is asserted to be the **older** one --
    an implementation that re-stamped re-used records with the current epoch
    would pass a bare "every matchup has one era" check and fail here.
    """
    first = epoch_book(tmp_path / "e1", epoch=1, k=2, entries=8, seed=51)
    seventh = epoch_book(tmp_path / "e7", epoch=7, k=2, entries=8, seed=52)

    records = [
        *even_link(_A, _B, pairs=_IN_FAMILY_PAIRS, book=seventh.entries[0].id),
        *even_link(_A, _ANCHOR, book=seventh.entries[1].id),
        *even_link(_B, _ANCHOR, book=seventh.entries[1].id),
        # Played once, in era 1, and re-used verbatim ever since.
        *even_link(_ANCHOR, _REF_LOW, book=first.entries[0].id),
        *even_link(_REF_LOW, _REF_HIGH, book=first.entries[1].id),
    ]
    match_set = MatchSet.of(records)

    fit = fit_elo(match_set, anchor=_ANCHOR)
    assert fit.converged, (
        "a mixed-era record set must still fit: a book era change does not "
        "invalidate the earlier games, only their pooling into one gain"
    )

    eras = {
        facts.players: facts.book_eras
        for facts in summarise_matchups(match_set, _ROSTER_MAP)
    }
    for players, matchup_eras in eras.items():
        assert len(matchup_eras) == 1, (
            f"matchup {players} carries eras {matchup_eras}: a matchup is "
            f"played once, on one epoch book"
        )

    def key(first: str, second: str) -> tuple[str, str]:
        """Return the matchup key ``summarise_matchups`` files a link under."""
        low, high = sorted((first, second))
        return (low, high)

    assert eras[key(_ANCHOR, _REF_LOW)] == ("E001",)
    assert eras[key(_REF_LOW, _REF_HIGH)] == ("E001",)
    assert eras[key(_A, _B)] == ("E007",)
    assert {era for eras_of in eras.values() for era in eras_of} == {"E001", "E007"}


# ---------------------------------------------------------------------------
# the T2b review: F1's module boundary, F2's contents hash, F4's load refusals,
# F5's rebuild, F9's path, and the two constants nothing pinned
# ---------------------------------------------------------------------------

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
"""``<repo>`` -- this file is ``<repo>/tests/eval/test_pool_books.py``."""

_BOOK_PROBE: Final[str] = '''
"""Import the epoch-book module and report what came in with it."""
import sys

import komugi.eval.pool_books  # noqa: F401

print("torch" in sys.modules, "komugi.eval.arena" in sys.modules)
'''

_RULER_PROBE: Final[str] = '''
"""Import §D6.3's net-free ruler: the control the probe must answer False for."""
import sys

import komugi.eval.rawmcts  # noqa: F401

print("torch" in sys.modules, "komugi.eval.arena" in sys.modules)
'''


def _imports_after(program: str) -> tuple[bool, bool]:
    """Return ``(torch, arena)`` residency after `program` in a fresh interpreter.

    Node 29's spawned-child idiom (``kom-036-fast-mode-ab.md:1218-1222``): this
    file's own imports pull torch in at collection, so an in-process
    ``sys.modules`` assertion would be about pytest's ordering rather than about
    the module's import graph.
    """
    environment = dict(os.environ)
    source = str(REPO_ROOT / "src")
    existing = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = (
        f"{source}{os.pathsep}{existing}" if existing else source
    )
    completed = subprocess.run(
        [sys.executable, "-c", program],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        cwd=str(REPO_ROOT),
    )
    assert completed.returncode == 0, completed.stderr
    answers = completed.stdout.strip().splitlines()[-1].split()
    assert len(answers) == 2, completed.stdout
    assert all(answer in ("True", "False") for answer in answers), completed.stdout
    return answers[0] == "True", answers[1] == "True"


def test_the_book_module_pulls_torch_only_through_the_engine() -> None:
    """Review F1: the module's boundary, pinned where the docstring claims it.

    ``pool_books`` used to say *"No torch here"* and it was false: ``num_actions``
    comes from :mod:`komugi.selfplay.records` and :class:`GameState` from
    :mod:`komugi.selfplay.adapter`, both of which import torch, and this file's
    own ``pytest.importorskip("torch")`` is the evidence. The sentence mattered
    because a later ticket would site a torch-free node on this module the way
    node 42 does on ``pool``/``interval``/``divergence``, or an operator would be
    told an epoch book can be minted on a box with no torch -- the first fails in
    CI, the second in the field at import.

    So the true boundary is pinned instead, in both directions: torch **does**
    arrive with the module, and :mod:`komugi.eval.arena` **does not** -- the
    arena imports the books, never the reverse, which is what keeps the book seam
    a leaf rather than a cycle through the match driver.

    *Non-vacuity:* the same probe is run over :mod:`komugi.eval.rawmcts`, §D6.3's
    deliberately net-free ruler, and must answer ``False`` for torch. Without
    that control a probe that always said ``True`` -- a harness leaking torch
    into every child -- would satisfy the first assertion.
    """
    torch_here, arena_here = _imports_after(_BOOK_PROBE)
    assert torch_here, (
        "importing komugi.eval.pool_books no longer pulls torch in: the module "
        "docstring's boundary paragraph is now wrong in the other direction, and "
        "a torch-free node can be sited on this module -- update the docstring "
        "and this assertion together"
    )
    assert not arena_here, (
        "importing komugi.eval.pool_books pulled komugi.eval.arena in: the "
        "dependency runs the other way (§D3), and a cycle through the match "
        "driver is what this edge exists to prevent"
    )

    ruler_torch, ruler_arena = _imports_after(_RULER_PROBE)
    assert not ruler_torch and not ruler_arena, (
        "the control module komugi.eval.rawmcts came back torch-tainted: the "
        "probe cannot distinguish a module that pulls torch from one that does "
        "not, so the assertions above say nothing"
    )


def hand_built_book(
    prefixes: Sequence[tuple[int, ...]], *, epoch: int = 7
) -> PoolBook:
    """Return an epoch book over `prefixes`, with every id derived.

    The construction ``build_epoch_book`` performs, without a source book:
    entry ``k``'s id is :func:`~komugi.eval.pool_books.entry_id_for` over its
    index and its actions, and the book's is
    :func:`~komugi.eval.pool_books.epoch_book_id_for` over the versioned fields
    -- which the constructor then re-derives and checks (review F2), so this
    helper cannot mint the unbound artefact F2 is about.
    """
    entries = tuple(
        PoolBookEntry(
            id=entry_id_for(epoch, index, actions),
            actions=actions,
            plies=len(actions),
            source_entry_id=f"gen0001-{index:06d}-0badc0de",
        )
        for index, actions in enumerate(prefixes)
    )
    fields: dict[str, Any] = {
        "pool_book_version": POOL_BOOK_VERSION,
        "epoch": epoch,
        "source_book": "tests/eval/test_pool_books.py",
        "source_book_sha256": "0" * 64,
        "gungi_mode": running_gungi_mode(),
        "action_size": num_actions(),
        "engine_rules_version": ENGINE_RULES_VERSION,
        "selection_seed": 1,
        "k": len(entries),
        "entries": [entry.as_json() for entry in entries],
    }
    return PoolBook(
        epoch=epoch,
        source_book="tests/eval/test_pool_books.py",
        source_book_sha256="0" * 64,
        gungi_mode=running_gungi_mode(),
        action_size=num_actions(),
        engine_rules_version=ENGINE_RULES_VERSION,
        selection_seed=1,
        k=len(entries),
        entries=entries,
        epoch_book_id=epoch_book_id_for(fields),
    )


def test_a_book_whose_id_is_not_its_own_contents_hash_is_refused_at_construction(
    tmp_path: Path,
) -> None:
    """Review F2: the id **is** the version, enforced by the type.

    ``__post_init__`` used to check only that the ids carried the ``E<NNN>-``
    prefix and were unique; the re-derivation lived in ``load_pool_book`` alone.
    A :class:`PoolBook` assembled in code -- a fixture, a driver building one
    from a manifest it already parsed, a merge of two loaders -- could therefore
    carry one book's ``epoch_book_id`` over a different entry list, pass the
    rated declaration check on the cross-family path, and store records whose
    entry ids were themselves unbound to their actions, with no refusal
    anywhere. §D3.3's whole claim would be false for that artefact.

    Three tampers, each a way an id can stop naming its contents: the book id
    itself, the entry **list** under an unchanged book id, and an entry id
    carried over another index's actions -- the last is the one that keeps the
    index inside the entry hash load-bearing.

    *Non-vacuity:* the untampered book reconstructs from its own fields, so the
    node is red against a constructor that refused every book rather than
    against one that checks.
    """
    book = epoch_book(tmp_path, epoch=7, k=2, entries=8, seed=121)

    intact = replace(book)
    assert intact.epoch_book_id == book.epoch_book_id
    assert [entry.id for entry in intact.entries] == [
        entry.id for entry in book.entries
    ]

    with pytest.raises(ValueError, match="not the hash of its own contents"):
        replace(book, epoch_book_id="E007-deadbeef")

    # The book id kept, the contents changed underneath it: the artefact F2
    # describes, and the one a hand-assembled book would produce.
    with pytest.raises(ValueError, match="not the hash of its own contents"):
        replace(book, entries=book.entries[:1], k=1)

    # An entry id carried over another index's actions -- swapped, so both ids
    # are real ids of this very book and only their positions are wrong.
    with pytest.raises(ValueError, match="hash to"):
        replace(book, entries=(book.entries[1], book.entries[0]))


def test_two_entries_with_identical_actions_get_distinct_ids() -> None:
    """The ``index`` inside an entry id is load-bearing (reviewer's mutant M5).

    :func:`~komugi.eval.pool_books.entry_id_for` hashes the index **and** the
    actions. Dropping the index passes every fixture in this suite, because a
    KOM-014 book's entries are distinct with probability ~1 -- so the duplicate
    guard never fires and the docstring's argument goes untested. Here the same
    prefix is deliberately selected twice: the two entries must still get
    distinct ids, and a book carrying both must still construct. Under an id
    that hashed the actions alone, both halves fail -- the ids collide and
    ``PoolBook`` refuses the pair as one opening claimed twice.
    """
    actions = random_prefix(np.random.default_rng(7), 4)
    first = entry_id_for(7, 0, actions)
    second = entry_id_for(7, 1, actions)
    assert first != second, (
        "two entries carrying the identical opening got the identical id: "
        "book_id would then be ambiguous about which pair was played, and "
        "check_paired_books compares the string for equality and would never "
        "notice"
    )

    book = hand_built_book((actions, actions))
    assert [entry.id for entry in book.entries] == [first, second]
    assert book.entry_for_pair(0).actions == book.entry_for_pair(1).actions


def test_the_pool_book_constants_are_the_pre_registered_literals() -> None:
    """``K``, the format version and ``sha8`` are numbers, not derivations.

    ``POOL_BOOK_K = 128`` is derived in §D3.2 from the largest matchup an epoch
    book must serve -- 128 pairs for §D2.2(b)1's in-family link -- and it is
    **pre-registered**: an epoch built at another ``K`` is a different
    experiment. Nothing pinned the literal (the reviewer's mutant M3 took it to
    64 and every test stayed green, because node 47 asserts ``str(POOL_BOOK_K)
    in message``, which reads the constant under test and moves with it). The
    same argument covers the format version and §D3.2's ``sha8`` width, which
    every stored id's length depends on.
    """
    assert POOL_BOOK_K == 128
    assert POOL_BOOK_VERSION == 1
    assert DIGEST_CHARS == 8
    assert len(entry_id_for(7, 0, (0,)).split("-")[1]) == DIGEST_CHARS


def _published_book(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    """Build one epoch book and return its path and its parsed manifest."""
    source = write_source_book(tmp_path / "books/gen_1.json", entries=8, seed=151)
    target = tmp_path / "pool/books/epoch_007.json"
    build_epoch_book(
        source,
        target,
        epoch=7,
        selection_seed=1007,
        engine_rules_version=ENGINE_RULES_VERSION,
        k=2,
    )
    payload: dict[str, Any] = json.loads(target.read_text(encoding="utf-8"))
    return target, payload


def _bump_version(payload: dict[str, Any]) -> None:
    """A file written by a reader this process does not have."""
    payload["pool_book_version"] = POOL_BOOK_VERSION + 1


def _wrong_action_size(payload: dict[str, Any]) -> None:
    """A book built under another action encoding."""
    payload["action_size"] = int(payload["action_size"]) + 1


def _wrong_mode(payload: dict[str, Any]) -> None:
    """A book built under another start mode."""
    payload["gungi_mode"] = "not-the-running-mode"


def _edit_an_action_list(payload: dict[str, Any]) -> None:
    """One entry shortened by a ply, under the id the file still declares.

    A truncated legal prefix is itself legal, so this edit survives the replay
    and is caught only by the id derivation -- which is the point.
    """
    entry = payload["entries"][0]
    entry["actions"] = list(entry["actions"])[:-1]
    entry["plies"] = len(entry["actions"])


def _delete_a_key(payload: dict[str, Any]) -> None:
    """A half-written or hand-truncated manifest."""
    del payload["source_book"]


@pytest.mark.parametrize(
    ("tamper", "names"),
    [
        pytest.param(_bump_version, "pool_book_version", id="version"),
        pytest.param(_wrong_action_size, "action space", id="action_size"),
        pytest.param(_wrong_mode, "GUNGI_MODE", id="gungi_mode"),
        pytest.param(_edit_an_action_list, "hash to", id="edited_actions"),
        pytest.param(_delete_a_key, "missing manifest key", id="deleted_key"),
    ],
)
def test_a_tampered_epoch_book_is_refused(
    tmp_path: Path, tamper: Callable[[dict[str, Any]], None], names: str
) -> None:
    """Review F4: every ``load_pool_book`` refusal, and all of them ``ValueError``.

    The four guards -- format version, action space, start mode and the
    re-derived id -- were each written behind a careful message that nothing
    exercised; a regression dropping the ``action_size`` guard would have
    shipped green, and that is the guard that stops a stale book replaying a
    plausible **wrong** line rather than failing. Separately, five of the nine
    manifest keys were read outside any guard, so a truncated file raised
    ``KeyError`` while the docstring promised ``ValueError`` on *"a missing
    manifest key"* -- a caller that wrote ``except ValueError`` exactly as
    documented would have watched it escape.

    Every case therefore asserts both: a ``ValueError``, and a message that
    names the file **and** the clause that refused it.

    *Non-vacuity:* the untampered file is asserted to load first, on the same
    path, so the node is red against a loader that refused everything.
    """
    target, payload = _published_book(tmp_path)
    assert load_pool_book(target).k == 2, "the untampered book must load"

    tamper(payload)
    target.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as raised:
        load_pool_book(target)
    message = str(raised.value)
    assert str(target) in message, (
        f"the refusal does not name the file it is about: {message!r}. A driver "
        f"loading a directory of epoch books cannot act on it"
    )
    assert names in message, (
        f"the refusal does not name the clause that fired: expected {names!r} "
        f"in {message!r}"
    )


SELECTION_SEED: int = 8027
"""§D3.2's own example ``selection_seed``, used for the rebuild node."""


def test_two_builds_of_one_epoch_from_one_source_are_byte_identical(
    tmp_path: Path,
) -> None:
    """§D3.2's rebuild property, and the generator that makes it true.

    ``epoch_book_id`` *means* "two builds of one epoch from one source are
    byte-identical"; nothing built one epoch twice and compared, and nothing
    pinned the bit generator by name. A refactor to ``np.random.default_rng`` --
    or to any other bit generator (the reviewer's mutant M4, PCG64 to MT19937) --
    changes **which** prefixes an epoch carries without changing a single test,
    and two sites rebuilding one epoch then disagree about what ``E007-...``
    names.

    Byte-equality alone cannot see that: a swapped generator moves both builds
    together. So the draw itself is re-derived here from an explicitly named
    ``Generator(PCG64(SeedSequence(seed)))`` -- ``book.py:196-201``'s rule, which
    exists because this artefact is compared byte-wise across machines -- and the
    selected entries are compared against it in **draw order**.

    *Non-vacuity:* a different ``selection_seed`` over the same source is
    asserted to mint a different id, so the equality above is evidence about
    determinism rather than about a builder that ignores its seed.
    """
    source = write_source_book(tmp_path / "books/gen_1.json", entries=16, seed=131)
    first = build_epoch_book(
        source,
        tmp_path / "one/epoch_007.json",
        epoch=7,
        selection_seed=SELECTION_SEED,
        engine_rules_version=ENGINE_RULES_VERSION,
        k=4,
    )
    second = build_epoch_book(
        source,
        tmp_path / "two/epoch_007.json",
        epoch=7,
        selection_seed=SELECTION_SEED,
        engine_rules_version=ENGINE_RULES_VERSION,
        k=4,
    )
    assert (tmp_path / "one/epoch_007.json").read_bytes() == (
        tmp_path / "two/epoch_007.json"
    ).read_bytes(), (
        "two builds of one epoch from one source produced different bytes: the "
        "epoch_book_id is a hash of those bytes, so it would name two artefacts"
    )
    assert first.epoch_book_id == second.epoch_book_id

    reseeded = build_epoch_book(
        source,
        tmp_path / "three/epoch_007.json",
        epoch=7,
        selection_seed=SELECTION_SEED + 1,
        engine_rules_version=ENGINE_RULES_VERSION,
        k=4,
    )
    assert reseeded.epoch_book_id != first.epoch_book_id

    # The bit generator, by name -- kills the PCG64 -> MT19937 mutant, which
    # byte-equality cannot see because it moves both builds together.
    candidates = load_book(source).entries
    drawn = np.random.Generator(
        np.random.PCG64(np.random.SeedSequence(SELECTION_SEED))
    ).choice(len(candidates), size=4, replace=False)
    expected = [candidates[int(index)].id for index in drawn]
    assert [entry.source_entry_id for entry in first.entries] == expected, (
        "the epoch's entries are not the ones PCG64 draws at this seed, in draw "
        "order: the selection's bit generator is pinned by name (§D3.2), and a "
        "different one mints a different epoch under the same id derivation"
    )


def test_the_epoch_id_does_not_depend_on_the_callers_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review F9: the source path is versioned, so it is stored resolved.

    §D3.3 does say the source file's *path* mints a new epoch, and that is kept.
    What is not kept is the **spelling**: the same file named
    ``books/gen_0031.json`` from a campaign's scripts directory and
    ``<root>/books/gen_0031.json`` from the campaign root would otherwise mint
    two different ids for a byte-identical artefact, which silently defeats
    §D3.2's rebuild property for every build a script runs relatively.

    *Non-vacuity:* a **second file** with the same bytes under a different name
    is asserted to mint a *different* id, so the fix normalises the path rather
    than dropping it from the version.
    """
    source = write_source_book(tmp_path / "books/gen_1.json", entries=8, seed=141)
    absolute = build_epoch_book(
        source,
        tmp_path / "pool/absolute.json",
        epoch=7,
        selection_seed=1007,
        engine_rules_version=ENGINE_RULES_VERSION,
        k=2,
    )

    monkeypatch.chdir(tmp_path)
    relative = build_epoch_book(
        Path("books/gen_1.json"),
        tmp_path / "pool/relative.json",
        epoch=7,
        selection_seed=1007,
        engine_rules_version=ENGINE_RULES_VERSION,
        k=2,
    )
    assert relative.epoch_book_id == absolute.epoch_book_id, (
        "the same source file named relatively minted a different epoch id: two "
        "builds of one epoch from one source are supposed to be byte-identical "
        "(§D3.2), and a scripted build names its source relatively"
    )
    assert relative.source_book == absolute.source_book == str(source.resolve())

    renamed = tmp_path / "books/gen_1_copy.json"
    renamed.write_bytes(source.read_bytes())
    copied = build_epoch_book(
        renamed,
        tmp_path / "pool/copied.json",
        epoch=7,
        selection_seed=1007,
        engine_rules_version=ENGINE_RULES_VERSION,
        k=2,
    )
    assert copied.epoch_book_id != absolute.epoch_book_id, (
        "two source files under different names minted one id: §D3.3 makes the "
        "source path part of the version, and resolving it must not drop it"
    )
