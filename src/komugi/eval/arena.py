"""The match arena: two checkpoints, one engine, the games a ladder is fit to.

Everything else in :mod:`komugi.eval` consumes
:class:`~komugi.eval.records.GameRecord` s and never asks where they came from
-- that is what lets the Elo core be tested against planted ratings
(:mod:`komugi.eval.synth`). This module is the other end: it *produces* those
records by actually playing games, so it is the one file in the package that
imports the engine, the search and torch. It is deliberately **not** re-exported
from :mod:`komugi.eval`'s ``__init__``; importing the rating stack must not drag
in a Cython engine and a deep-learning framework.

The loop is :meth:`komugi.selfplay.worker.SelfPlayWorker.play_game` with the
self-play parts removed and one part added:

* removed -- the full/fast draw, the recorded rows, the shard writer, and the
  training targets. An arena game produces exactly one number, the result from
  white's perspective, plus the moves that produced it.
* added -- **two** evaluators. Which one is asked depends on which seat is to
  move (:func:`play_match`), and that is the entire difference between self-play
  and a match.

What is *not* different, on purpose:

* **A fresh :class:`~komugi.search.gumbel.GumbelConfig` per move, seeded per
  move** (design F10d). The search draws its Gumbel vector once per call from
  the config's seed, so one seed per game would replay the identical noise
  vector at every ply and give one lucky action id the same top-``m``
  tie-break for the whole game. Seeds here are
  :func:`~komugi.selfplay.records.move_seed` over
  :func:`~komugi.selfplay.records.game_seed`, i.e. the *same* derivation
  self-play uses (see :func:`game_seeds` for the spawn key an arena game gets).
* **The engine owns every termination**, the engine's turn cap included. There is no
  ply cap in this module and no ``max_plies`` knob to add one: the loop runs
  while ``not state.is_terminal()``, exactly as the worker's does. A cap here
  would silently convert lost positions into draws and the ladder would rate
  the cap rather than the players.
* **No resign, no temperature, no root noise.** The move played is
  :attr:`~komugi.search.gumbel.SearchResult.action` and nothing else.

Pairing, honestly (design review D9)
------------------------------------
Games ``2k`` and ``2k+1`` are one ``pair_id`` with the seats swapped, which is
what :func:`komugi.eval.seats.check_paired_books` audits and what
:mod:`komugi.eval.elo` 's seat-free likelihood assumes. What a pair balances
depends on whether :attr:`ArenaConfig.book` carries an epoch book:

* **Bookless** (the default, and every record stored before kom-011d T2b):
  every game starts from the empty board, ``book_id`` is the constant
  :data:`EMPTY_BOOK`, and a pair balances **seats and nothing else** -- the two
  games are not the same opening played twice, they are two games from the same
  start whose only structural difference is who holds white. That is worth
  having on its own (it is what removes the seat term from the fit) but it is
  not the variance reduction a real book set buys, and a caller reading such a
  pair as "the same position from both sides" would be over-reading it.
* **With a** :class:`~komugi.eval.pool_books.PoolBook`: pair ``k`` replays entry
  ``k``'s prefix in **both** its games, so the two entrants see the identical
  opening from opposite seats and ``book_id`` is that entry's id (kom-011d
  §D3.4). The audit's "same ``book_id``" clause stops being vacuous, and the
  pair balances the opening as well as the seat.

Determinism, and exactly how far it goes (design review D5)
------------------------------------------------------------
On the **CPU**, with evaluators that are pure functions of their input (a stub,
or a torch module in ``eval()`` mode), a match is **bit-deterministic**: the same
players and the same :class:`ArenaConfig` produce the identical records and the
identical move sequences, because every source of randomness in the stack is the
per-move seed derived above and nothing reads a clock, a hash or a global RNG.

On a **GPU** the claim weakens to "deterministic under pinned settings":
:func:`pin_determinism` asks torch for deterministic kernels
(``use_deterministic_algorithms(True, warn_only=True)``) and turns off cuDNN
autotuning, which is what the platform offers -- but the answer still depends on
the device, the driver and the batch composition of each wave, so two *machines*
are not promised the same game. ``warn_only=True`` rather than hard failure: an
arena run that dies because one kernel has no deterministic implementation buys
nothing, while a warning names the kernel and the run still measures strength.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, NamedTuple, cast

import numpy as np
import numpy.typing as npt
import torch

from komugi.campaign import factory
from komugi.config import SelfPlayConfig
from komugi.engine.game import Game
from komugi.engine.repetition import RepetitionRule
from komugi.eval.records import (
    BLACK_WIN,
    DRAW,
    WHITE_WIN,
    GameRecord,
    MatchSet,
    Player,
)
from komugi.search import modes
from komugi.search.gumbel import GumbelConfig, GumbelSearch
from komugi.search.state import Evaluator, SearchState, WaveEvaluator
from komugi.search.tactical import TacticalProvider
from komugi.eval.pool_books import PoolBook, PoolBookEntry
from komugi.selfplay.adapter import EvaluableState, GameState, make_evaluator
from komugi.selfplay.tactical import GameMarshalCaptureProvider
from komugi.selfplay.inference import (
    WDL_SIZE,
    BatchEvaluator,
    InferenceConfig,
    InProcessEvaluator,
    wdl_utility,
)
from komugi.selfplay.records import game_seed, move_seed, num_actions
from komugi.train.checkpoint import (
    CheckpointRef,
    list_checkpoints,
    load_checkpoint_file,
    require_matching_plane_count,
)

_PRODUCTION: Final[SelfPlayConfig] = SelfPlayConfig(workers=1, process_batch_size=1)
"""A default :class:`~komugi.config.SelfPlayConfig`, read for its budgets only.

Constructed rather than restated so :data:`DEFAULT_M` and :data:`DEFAULT_N` are
*the* production full-search budget by derivation. A match played at a budget
that had quietly drifted from the one self-play uses would rate a search nobody
runs.
"""

DEFAULT_M: Final[int] = _PRODUCTION.gumbel_m_full
"""Root actions an arena player samples: the production full search (32)."""

DEFAULT_N: Final[int] = _PRODUCTION.gumbel_n_full
"""Simulation budget of one arena move: the production full search (256)."""

DEFAULT_GUMBEL_SCALE: Final[float] = GumbelConfig(
    num_actions=1, m=1, n=0, seed=0
).gumbel_scale
"""The search's own default noise scale, read off a throwaway config.

Same reason as :data:`DEFAULT_M`: the number lives in
:class:`~komugi.search.gumbel.GumbelConfig` and is read from there, so a change
to the search's default cannot leave a stale copy behind in the arena.
"""

EMPTY_BOOK: Final[str] = "empty"
"""``book_id`` for a game started from the engine's empty board.

Not a placeholder for "unknown": it names a real, single opening position that
every game of a **bookless** run shares, and it is what
:attr:`ArenaConfig.book_id` defaults to.

**Books are wired here now** (kom-011d §D3, ``docs/design/kom-011d-ladder-pool.md``),
which is what makes the sentence this docstring used to carry -- that the draft
book generator had landed and was deliberately not wired to the arena -- false.
What changed is not KOM-014's book but its *lifecycle*: a draft book is a
self-play initial-state device **refreshed every generation**
(``docs/design/kom-014-draft-books.md`` §5.2) and a match book must be
*"versioned and frozen per ladder epoch"* (``DESIGN.html:260``), so the arena
consumes a :class:`~komugi.eval.pool_books.PoolBook` -- a frozen, hash-versioned
selection of prefixes *from* one named KOM-014 book file -- rather than a
campaign root's newest generation. Pass one as :attr:`ArenaConfig.book` and each
pair's records carry that pair's **entry** id instead of this constant; pass
none, which is the default, and nothing about a match moves.

A draft book sitting in a campaign root still changes nothing here: the only way
one reaches the arena is through
:func:`komugi.eval.pool_books.build_epoch_book`, which freezes it first.
"""

LADDER_FAMILY: Final[str] = "komugi"
"""Engine family stamped on every checkpoint entrant.

One of :data:`komugi.eval.records.KNOWN_FAMILIES`, so a ladder report can be
grouped by lineage without a translation table.
"""

_ARENA_GAME_INDEX: Final[int] = 0
"""The ``game_index`` component of an arena game's seed key -- see :func:`game_seeds`."""

_WHITE: Final[int] = 0
"""The engine's index for the white seat, in ``to_play`` and ``win_state``."""

_DRAW_INDEX: Final[int] = 2
"""Draw slot in the engine's absolute ``[white, black, draw]`` outcome."""

_NUM_ACTIONS: Final[int] = num_actions()
"""The engine's action-space size, read from the encoding rather than restated."""

_PRODUCTION_PROVIDER_ID: Final[str] = "game-marshal-capture-v1"
"""The one tactical provider a discovery-mode arena match is allowed to run.

Restated as a literal rather than read off the class: this is the id a record
claims the discovery layer consulted, so :func:`_wire` can assert against it and
a provider swapped underneath the arena fails loudly instead of quietly changing
what a published identity means.
"""


def pin_determinism() -> None:
    """Ask torch for reproducible kernels; safe to call repeatedly.

    Global process state, deliberately: the settings below are torch-wide, and
    an arena that set them per match would be lying about the games another
    thread is playing. :func:`play_match` calls this at setup so that a caller
    who never heard of the flags still gets the claim in the module docstring.

    ``warn_only=True`` means an operation with no deterministic implementation
    warns and runs anyway rather than raising: on the CPU path this module
    actually promises bit-determinism for, nothing in the stack needs the
    escape hatch, and on a GPU a hard failure would cost the measurement to
    protect a claim that was already weaker there.

    Every attribute is probed before it is set. ``torch.backends.cudnn`` is
    absent from CPU-only builds, and a match that refused to start because a
    machine has no cuDNN would be an arena that only runs on GPUs.
    """
    torch.use_deterministic_algorithms(True, warn_only=True)
    cudnn = getattr(torch.backends, "cudnn", None)
    if cudnn is None:
        return
    if hasattr(cudnn, "deterministic"):
        cudnn.deterministic = True
    if hasattr(cudnn, "benchmark"):
        # Autotuning picks a kernel by timing it, so the choice -- and with it
        # the arithmetic -- depends on what else the machine was doing.
        cudnn.benchmark = False


def make_batch_wave_evaluator(
    evaluator: BatchEvaluator, draw_score: float
) -> WaveEvaluator:
    """Adapt a :class:`~komugi.selfplay.inference.BatchEvaluator` to the wave seam.

    The in-process twin of
    :func:`komugi.selfplay.adapter.make_wave_evaluator`, which cannot be reused
    here because it is typed against a
    :class:`~komugi.selfplay.inference.SlotClient` -- the shared-memory end of
    the seam, which an arena has no server for. The batching an arena wants is
    the *evaluator's own* ``evaluate``: one halving round's leaves become one
    forward instead of one forward each, which is the whole cost of a match at
    256 simulations a move.

    This changes when the network is asked and never what is asked: the search's
    wave schedule is bit-identical to its scalar one (pinned by
    ``tests/search/unit/test_gumbel_waves.py``), and the perspective rules here
    are :func:`komugi.selfplay.adapter.make_evaluator`'s, applied per row.

    Args:
        evaluator: The batch evaluator both seams share, so a wave of one and a
            scalar call ask the same object the same question.
        draw_score: What a draw is worth to the side to move at the evaluated
            node -- design D1's one network-side entry point for contempt.

    Returns:
        A :data:`~komugi.search.state.WaveEvaluator` closure over `evaluator`.
    """

    def evaluate_wave(
        states: Sequence[SearchState],
    ) -> Sequence[tuple[npt.NDArray[np.float32], float]]:
        """Evaluate every state in `states` in one forward."""
        if not states:
            raise ValueError(
                "a wave must carry at least one state: the search never submits "
                "an empty wave, so an empty one here is a caller bug"
            )
        evaluables = [cast(EvaluableState, state) for state in states]
        obs = np.stack(
            [
                np.ascontiguousarray(evaluable.observation(), dtype=np.float32)
                for evaluable in evaluables
            ]
        )
        legal = np.stack(
            [
                np.asarray(evaluable.legal_mask(), dtype=np.bool_)
                for evaluable in evaluables
            ]
        )
        answer = evaluator.evaluate(obs, legal)
        rows = len(evaluables)
        policy = np.asarray(answer.policy_logits, dtype=np.float32).reshape(rows, -1)
        wdl = np.asarray(answer.wdl, dtype=np.float32).reshape(rows, WDL_SIZE)
        utility = wdl_utility(wdl, draw_score)
        return [(policy[row], float(utility[row])) for row in range(rows)]

    return evaluate_wave


@dataclass(frozen=True, slots=True)
class ArenaPlayer:
    """One entrant: who it is, how to build its network, and what it may spend.

    The evaluator is a **factory**, not an instance, for the same reason the
    self-play orchestrator takes one (design D2): the arena decides when a
    network is built and how long it lives -- once per match, here -- and a
    caller that handed in a live object would be deciding that instead.

    Args:
        player: The rateable identity that reaches every
            :class:`~komugi.eval.records.GameRecord`. Two entrants in one match
            must have different ids; a record refuses ``white == black``, and
            :func:`play_match` refuses it earlier, before any game is played.
        evaluator_factory: Builds this player's network wrapper. Called exactly
            once per :func:`play_match`, and the resulting evaluator serves both
            games of every pair.
        m: Root actions sampled per move. Defaults to :data:`DEFAULT_M`.
        n: Simulation budget per move. Defaults to :data:`DEFAULT_N`.
        gumbel_scale: Multiplier on the root Gumbel noise, defaulting to the
            search's own :data:`DEFAULT_GUMBEL_SCALE`.
        proof_mode: Which proof layer this entrant searches with
            (:mod:`komugi.search.modes`). ``off`` is the legacy search, byte for
            byte, and is the default so that every existing call site keeps the
            player it already had.
        discovery_mode: Which discovery layer runs before halving. ``off`` never
            constructs a tactical provider at all; ``root_marshal_v1`` makes
            :func:`_wire` build one per match, and requires
            ``proof_mode != "off"`` (validated here, at construction).

            **Discovery is UNRATED and uncounted here.** The rows the layer
            spends do not pass through §D3.3's counted evaluator, and the
            theoretical per-move allowance it may reach is 41 rows at ``m=8``/
            ``n=32`` and 289 at ``m=32``/``n=256`` -- both above the rated
            ceiling of 257. So a discovery match is an in-process experiment
            (what the layer buys), never a rung of the rated ladder: the rated
            seam, :class:`~komugi.eval.gumbel_mover.GumbelMover` and
            :func:`~komugi.eval.gumbel_mover.komugi_entrant`, refuses any
            non-``off`` ``discovery_mode`` outright, before it loads a single
            weight.

    The budget is **per player**, not per match: rating a checkpoint at 32
    simulations against the same checkpoint at 256 is a legitimate experiment
    (it measures what search buys), and a per-match budget could not express it.
    The modes are per player for the same reason: "what does the proof layer buy"
    is a match between two entrants that differ in exactly that, and
    :attr:`search_identity` is the provenance string each side is recorded under.
    """

    player: Player
    evaluator_factory: Callable[[], BatchEvaluator]
    m: int = DEFAULT_M
    n: int = DEFAULT_N
    gumbel_scale: float = DEFAULT_GUMBEL_SCALE
    proof_mode: str = "off"
    discovery_mode: str = "off"

    def __post_init__(self) -> None:
        """Refuse an unrunnable mode pair before a single game is played."""
        modes.validate_search_modes(self.proof_mode, self.discovery_mode)

    @property
    def search_identity(self) -> modes.SearchAlgorithmIdentity:
        """Return the algorithm this entrant's two modes name.

        Derived, never stored: there is no way to record a decision rule this
        player does not actually search with.
        """
        return modes.search_algorithm_identity(self.proof_mode, self.discovery_mode)


@dataclass(frozen=True, slots=True)
class ArenaConfig:
    """How many games to play, from what entropy, and under what contempt.

    Args:
        games: Total games in the match. **Must be even**: a match is a whole
            number of seat-swapped pairs, and an odd game count would leave one
            player a seat ahead -- exactly the bias the pairing exists to
            remove, and one :func:`komugi.eval.seats.check_paired_books` would
            report as a dangling ``pair_id`` long after the games were paid for.
        seed: The match's entropy. With the pair and seat indices it names every
            search in the match (:func:`game_seeds`). The key does **not**
            include the players' identities, so two matches played at one seed
            -- adjacent rungs of a ladder, say -- draw *identical* per-ply
            Gumbel streams: their games are correlated, which mildly biases a
            joint Elo fit. A ladder run should give every matchup its own seed
            (disjoint ranges when parallelised).
        draw_score: What a draw is worth to the side to move. **Required, with
            no default** (design review D3): it is the contempt the games are
            played under, it reaches both the terminal values
            (:class:`~komugi.selfplay.adapter.GameState`) and the network values
            (:func:`komugi.selfplay.inference.wdl_utility`), and a match played
            at a different contempt from the self-play that trained the
            checkpoints is a different experiment. A silent default would make
            that difference invisible in the call site.
        book_id: The opening the games start from when `book` is ``None``:
            :data:`EMPTY_BOOK`, the engine's empty board. With a `book`
            supplied it is **not** what the records carry -- each pair's games
            carry their entry's id -- and it names the epoch book as a whole
            instead, which is what
            :func:`komugi.eval.v1bridge.play_bridged_match` checks a rated
            cross-family match's declaration against (kom-011d §D3.2, KOM-023
            §D4 amendment 4).
        book: The frozen epoch book this match's openings come from
            (:class:`~komugi.eval.pool_books.PoolBook`), or ``None`` for the
            bookless empty-board match every record stored before kom-011d T2b
            was played as. Pair ``k`` replays entry ``k``'s prefix in both of
            its games (§D3.4), so the match needs at least :attr:`pairs`
            entries and is refused here if the book is smaller -- before any
            game is paid for, rather than at the pair the book runs out on.
        pair_namespace: An optional prefix on every ``pair_id`` this match
            writes (:func:`pair_id_for`), so that two matches of the *same* two
            entrants -- a pool epoch's re-link, say -- can be pooled into one
            :func:`komugi.eval.seats.check_paired_books` audit instead of
            colliding on ``pair_id`` and failing with GAME_COUNT violations
            (kom-011d §D2.3). It must not contain ``":"`` or ``"/"``: the id is
            ``:``-joined and the namespace is ``/``-joined onto it, and a
            namespace carrying either separator would make a ``pair_id`` that
            cannot be split back into its parts.

            **The default ``""`` is today's behaviour byte-for-byte** -- no
            prefix, no separator, the exact ``f"{a}:{b}:{k:05d}"`` every stored
            record and every existing test already carries. A namespace is
            something a caller opts into; nothing about a bookless,
            single-epoch match changes when this field exists.
        repetition_rule: The fourfold-repetition key this match's **referee**
            adjudicates on (KOM-141). The default,
            :attr:`~komugi.engine.repetition.RepetitionRule.FULL`, is Komugi's
            repaired 274-byte key -- the key self-play trains under, and today's
            behaviour byte for byte.
            :attr:`~komugi.engine.repetition.RepetitionRule.PLACEMENT` is
            gungi.js@3.4.11's cells-only key, and is for one purpose only:
            refereeing a match under the SITE's rule so the Elo cost of the
            mismatch can be measured. It re-introduces the rules-audit F1-F4
            aliasing deliberately, so a caller must ask for it by name; no
            other :class:`ArenaConfig` construction site in this repo passes
            it, and ``tests/engine/test_repetition_rule_default.py`` is what
            keeps that true.

    Raises:
        ValueError: On a negative or odd `games`, a `draw_score` outside
            ``[-1, 1]``, a blank `book_id`, a `pair_namespace` containing
            ``":"`` or ``"/"``, a `book` this config does not declare
            (``book_id != book.epoch_book_id``), or a `book` with fewer entries
            than the match has pairs.
    """

    games: int
    seed: int
    draw_score: float
    book_id: str = EMPTY_BOOK
    pair_namespace: str = ""
    book: PoolBook | None = None
    repetition_rule: RepetitionRule = RepetitionRule.FULL

    def __post_init__(self) -> None:
        """Reject a match that could not be a whole number of paired books."""
        if self.games < 0:
            raise ValueError(f"games must be >= 0, got {self.games!r}")
        if self.games % 2 != 0:
            raise ValueError(
                f"games must be even, got {self.games!r}: a match is played as "
                f"seat-swapped pairs, so an odd count would give one player an "
                f"extra game with white"
            )
        if not -1.0 <= self.draw_score <= 1.0:
            raise ValueError(
                f"draw_score must be in [-1, 1], got {self.draw_score!r}"
            )
        if not self.book_id.strip():
            raise ValueError("book_id must be a non-empty identifier")
        if ":" in self.pair_namespace or "/" in self.pair_namespace:
            found = ":" if ":" in self.pair_namespace else "/"
            raise ValueError(
                f"pair_namespace must not contain {found!r}, got "
                f"{self.pair_namespace!r}: a pair_id is ':'-joined and the "
                f"namespace is '/'-joined onto it (pair_id_for), so a "
                f"namespace carrying either produces an id that cannot be "
                f"parsed back to the entrants that played it"
            )
        if self.book is not None and self.book_id != self.book.epoch_book_id:
            # The in-process twin of ``v1bridge._assert_declared_book`` (review
            # F10). It costs nothing -- every caller already satisfies it -- and
            # without it a hand-built in-family pool match stores records whose
            # per-game ids are entry ids while the config declares the empty
            # board, so the manifest's epoch_book_id and the records disagree
            # and nothing refuses.
            raise ValueError(
                f"a config carrying epoch book {self.book.epoch_book_id!r} must "
                f"declare it: book_id is {self.book_id!r}. The per-game book_ids "
                f"are this book's entry ids (kom-011d §D3.3), so the config is "
                f"the only place the artefact they came from is named"
            )
        if self.book is not None and len(self.book) < self.pairs:
            raise ValueError(
                f"epoch book {self.book.epoch_book_id!r} carries "
                f"{len(self.book)} entries and this match is {self.pairs} "
                f"pairs: pair k plays entry k (kom-011d §D3.2), so a matchup "
                f"wider than K would either repeat an opening inside itself or "
                f"run out of book mid-match. Mint an epoch book at a larger K"
            )

    @property
    def pairs(self) -> int:
        """Return how many seat-swapped pairs this match is."""
        return self.games // 2

    def entry_for_pair(self, pair: int) -> PoolBookEntry | None:
        """Return the epoch-book entry pair `pair` plays, or ``None`` if bookless.

        The one place the pair-to-entry map is read, so :func:`play_match` and
        :func:`komugi.eval.v1bridge.play_bridged_match` cannot disagree about
        which opening a pair was played from -- the same argument
        :func:`pair_id_for` makes about the id itself.
        """
        if self.book is None:
            return None
        return self.book.entry_for_pair(pair)

    def book_id_for_pair(self, pair: int) -> str:
        """Return the ``book_id`` pair `pair`'s two games are recorded under.

        The **entry** id when a book is supplied (§D3.3: *"every ``GameRecord``
        written under epoch ``E`` carries ``book_id`` = the entry id, which is
        prefixed with the epoch"*), and :attr:`book_id` otherwise. Both games of
        the pair get the same string, so ``BOOK_MISMATCH`` cannot fire
        spuriously and a real mismatch still can.
        """
        entry = self.entry_for_pair(pair)
        return self.book_id if entry is None else entry.id

    def prefix_for_pair(self, pair: int) -> tuple[int, ...]:
        """Return the actions both games of pair `pair` open with; ``()`` if bookless."""
        entry = self.entry_for_pair(pair)
        return () if entry is None else entry.actions


class ArenaSummary(NamedTuple):
    """What a match came to, beyond the records themselves.

    Draws and ply lengths are the two things a ladder operator reads *before*
    the ratings: a match that is 100% draws has fit an Elo gap out of nothing,
    and games that all run to the engine's turn cap say the same thing
    louder. Both are cheap to accumulate while the games are being played and
    impossible to recover from the records afterwards (a
    :class:`~komugi.eval.records.GameRecord` has no length and, by design, no
    termination field).

    **The plies counted are the *searched* ones** -- a booked game's replayed
    prefix is excluded (review F8). ``GameOutcome.moves`` legitimately carries
    the prefix, because a caller replaying the list has to play it, but §D2.4
    prices a game as ``P x (257/w) x 2.7 ms`` over the plies a search was
    actually run for, and T2a's probe divides its measured forward calls by this
    counter to report calls per move. Counting an ``L``-ply opening as played
    would read both low by ``L`` on every booked game -- 5-20% at the campaign's
    mean 59-90 plies -- and set the epoch cadence §D2.4 derives from them too
    long. A bookless match is unchanged: its prefix is empty.
    """

    games: int
    decisive: int
    draws: int
    total_plies: int
    shortest_plies: int
    longest_plies: int

    @property
    def draw_rate(self) -> float:
        """Return the fraction of games drawn; ``0.0`` for an empty match."""
        if self.games == 0:
            return 0.0
        return self.draws / self.games

    @property
    def mean_plies(self) -> float:
        """Return the mean game length in plies; ``0.0`` for an empty match."""
        if self.games == 0:
            return 0.0
        return self.total_plies / self.games


class GameOutcome(NamedTuple):
    """One finished arena game: the result, and the moves that produced it.

    ``result`` is in :data:`~komugi.eval.records.VALID_RESULTS` and is stated
    from **white's** perspective, which is the convention every consumer of
    :class:`~komugi.eval.records.GameRecord` already assumes.

    ``moves`` is the played action ids in order. Nothing in the rating stack
    wants them -- they are here because a game is otherwise unreproducible
    except by replaying the whole search, which makes both debugging a
    surprising result and the worker-parity test in ``tests/eval`` possible.
    """

    result: float
    moves: tuple[int, ...]


class MatchOutcome(NamedTuple):
    """A finished match: the records to rate, and the summary to sanity-check."""

    records: tuple[GameRecord, ...]
    summary: ArenaSummary

    def match_set(self) -> MatchSet:
        """Return the records as the :class:`~komugi.eval.records.MatchSet`."""
        return MatchSet.of(self.records)


def game_seeds(config: ArenaConfig, pair: int, seat: int) -> np.random.SeedSequence:
    """Return the sequence owning one arena game's randomness.

    ``game_seed(config.seed, pair, seat, 0)``, i.e. the *self-play* derivation
    (:func:`komugi.selfplay.records.game_seed`) with an arena game's coordinates
    in the spawn key: the match's entropy, the pair index, and which of the
    pair's two seatings this is. Composing the existing helper rather than
    building a ``SeedSequence`` by hand buys two things:

    * the per-move seeds are then :func:`~komugi.selfplay.records.move_seed`'s,
      so the arena and the self-play worker cannot drift apart in how a ply's
      Gumbel seed is derived; and
    * an arena game is *seed-identical* to a self-play game at
      ``(run_seed=config.seed, worker_id=pair, incarnation=seat,
      game_index=0)``, which is what makes the parity test in
      ``tests/eval/test_arena_behavior.py`` a comparison rather than a
      coincidence -- the same seeds, the same budgets, the same evaluator, so
      any difference in the move sequence is a difference in *this* loop.

    The trailing :data:`_ARENA_GAME_INDEX` is the one component with nothing to
    say: an arena game is named by its pair and its seat, and the fourth slot
    exists only because the helper's key is the worker's.
    """
    return game_seed(config.seed, pair, seat, _ARENA_GAME_INDEX)


def move_seed_for(config: ArenaConfig, pair: int, seat: int, ply: int) -> int:
    """Return the Gumbel seed for one move: ply `ply` of game `(pair, seat)`.

    A pure function of its arguments (design F10d): asking twice for the seed of
    ply 3 answers the same both times, which is what the replay claim in the
    module docstring rests on. Public because "was this really seeded per move?"
    is a question a test should be able to ask without a recording stub.
    """
    return move_seed(game_seeds(config, pair, seat), ply)


def pair_id_for(config: ArenaConfig, a_id: str, b_id: str, pair: int) -> str:
    """Return the ``pair_id`` pair `pair` of an `a_id` vs `b_id` match is stored under.

    One rule with **two** construction sites -- :func:`play_match` and
    :func:`komugi.eval.v1bridge.play_bridged_match` -- so it is written once and
    called twice rather than typed twice (kom-011d §D2.3). The bridged path
    already borrows this module's loop for the same reason; a second copy of the
    format is a second thing to keep in step with
    :func:`komugi.eval.seats.check_paired_books`, which parses nothing but does
    group by exactly this string.

    The namespace is a **prefix**, joined with ``"/"`` and applied only when it
    is non-empty: ``ArenaConfig.pair_namespace == ""`` -- the default -- returns
    the byte-identical ``f"{a}:{b}:{k:05d}"`` this function replaced, so no
    stored record and no existing caller moves. ``"/"`` rather than ``":"``
    because the id is ``:``-joined and a third colon-separated component would
    be indistinguishable from an entrant id (the namespace is refused a ``":"``
    at :class:`ArenaConfig` construction for the same reason).

    Args:
        config: The match config, read for its ``pair_namespace`` alone.
        a_id: The first entrant's ``player.id``, in the order it was passed --
            not in seat order, so both games of a pair carry the same id.
        b_id: The second entrant's ``player.id``.
        pair: The pair index, zero-padded to five digits.

    Returns:
        The ``pair_id`` both games of the pair are recorded under.
    """
    prefix = f"{config.pair_namespace}/" if config.pair_namespace else ""
    return f"{prefix}{a_id}:{b_id}:{pair:05d}"


class _Side(NamedTuple):
    """One entrant wired to a live evaluator, for the length of a match.

    Both seams over one evaluator: the scalar one the search uses for the root
    and for forced moves, and the wave one it uses for a halving round's leaves.
    They share the object, so a player cannot answer two different ways
    depending on which schedule reached it.

    `provider` is the rules-level seam the KOM-220 discovery layer consults, and
    it is ``None`` for an entrant whose ``discovery_mode`` is ``off`` -- the
    legacy player never has one built for it. It is per match rather than per
    move for the same reason the evaluator is: the search is rebuilt every move
    (:func:`move_search`) and carries no state between them, so the provider is
    the only thing that would pay for a rebuild without buying anything.
    """

    entrant: ArenaPlayer
    scalar: Evaluator
    wave: WaveEvaluator
    provider: TacticalProvider | None = None


def _wire(entrant: ArenaPlayer, draw_score: float) -> _Side:
    """Build `entrant`'s evaluator once and adapt it to both search seams.

    The provider is built and checked **before** the evaluator factory runs: the
    refusal below is a wiring error, and paying for a checkpoint's weights --
    seconds of load, a device allocation -- to discover it is a cost a match that
    cannot start should not incur.
    """
    provider: TacticalProvider | None = None
    if entrant.discovery_mode != "off":
        provider = GameMarshalCaptureProvider()
        # The identity is what a record says the discovery layer consulted, so
        # a provider whose id is not the published one is a provenance bug, not
        # a substitution the arena may make quietly.
        if provider.provider_id != _PRODUCTION_PROVIDER_ID:
            raise ValueError(
                f"the arena's tactical provider reports "
                f"{provider.provider_id!r}, expected "
                f"{_PRODUCTION_PROVIDER_ID!r}"
            )
    evaluator = entrant.evaluator_factory()
    return _Side(
        entrant=entrant,
        scalar=make_evaluator(evaluator, draw_score),
        wave=make_batch_wave_evaluator(evaluator, draw_score),
        provider=provider,
    )


def _result_of(state: GameState) -> float:
    """Return the finished game's result from white's perspective."""
    outcome = state.absolute_outcome()
    if bool(outcome[_DRAW_INDEX]):
        return DRAW
    if bool(outcome[_WHITE]):
        return WHITE_WIN
    if bool(outcome[1 - _WHITE]):
        return BLACK_WIN
    raise ValueError(
        "the game loop exited on a position the engine has not adjudicated: "
        "win_state() is all zero"
    )


def move_search(
    scalar: Evaluator,
    wave: WaveEvaluator,
    *,
    m: int,
    n: int,
    gumbel_scale: float,
    seed: int,
    proof_mode: str = "off",
    discovery_mode: str = "off",
    tactical_provider: TacticalProvider | None = None,
) -> GumbelSearch:
    """Return the search **one arena move** is chosen by.

    Factored out of :func:`_play_wired_game` rather than written twice: a
    cross-engine entrant (:mod:`komugi.eval.gumbel_mover`) has to run *the same
    search this loop runs* for its numbers to be comparable with a ladder rung's,
    and "the same search" is a claim that only holds by construction if there is
    one construction. Every argument that could make two searches differ is a
    parameter here, and there are no defaults: a caller cannot silently inherit a
    budget it did not state.

    Fresh per move because the seed is what changes per move (design F10d): one
    :class:`~komugi.search.gumbel.GumbelConfig` reused down a game replays one
    Gumbel vector at every ply.

    Args:
        scalar: The single-position seam, used for the root and forced moves.
        wave: The batched seam, used for a halving round's leaves.
        m: Root actions sampled.
        n: Simulation budget.
        gumbel_scale: Multiplier on the root Gumbel noise.
        seed: This move's seed, i.e. ``move_seed(seeds, ply)``.
        proof_mode: The searching entrant's proof layer, ``off`` by default.
        discovery_mode: Its discovery layer, ``off`` by default.
        tactical_provider: The provider the discovery layer consults; ``None``
            unless `discovery_mode` asks for one, which
            :class:`~komugi.search.gumbel.GumbelSearch` itself enforces both ways.

    The three mode arguments **do** default, unlike the budget: ``off``/``off``/
    ``None`` is the legacy search this function has always built, so every call
    site that predates the layers keeps constructing byte-identically the search
    it already constructed.

    Returns:
        The :class:`~komugi.search.gumbel.GumbelSearch` for this one move.
    """
    return GumbelSearch(
        scalar,
        GumbelConfig(
            num_actions=_NUM_ACTIONS,
            m=m,
            n=n,
            seed=seed,
            gumbel_scale=gumbel_scale,
            proof_mode=proof_mode,
            discovery_mode=discovery_mode,
        ),
        wave_evaluator=wave,
        # Passed only when a provider exists: the legacy call shape is preserved
        # for every off-mode caller and for test doubles that predate the seam.
        **({"tactical_provider": tactical_provider} if tactical_provider is not None else {}),
    )


def replay_prefix(
    state: GameState,
    prefix: Sequence[int],
    *,
    announce: Callable[[GameState, int], None] | None = None,
) -> tuple[int, ...]:
    """Play an epoch book's prefix onto `state` before a game begins (§D3.4).

    One helper, two loops: this module's :func:`_play_wired_game` and
    :func:`komugi.eval.v1bridge.play_bridged_game` both open a booked game with
    it, so a bridged game and an in-process one cannot disagree about what "the
    pair's opening" means. The prefix is replayed on the **referee's** position,
    which is the same object the loop then plays on -- there is no second board
    and no FEN handed to an adapter, which is what would need a new protocol
    request, a new capability flag and a new failure mode (§D3.4).

    `announce` is how a *stateful* mover learns the opening. It is called once
    per prefix action with the position the action is played **from**, before
    it is played, because decoding an action id to a move record needs that
    board (``game.py:358-367``) and the loop has already advanced past it by the
    time both movers would be notified. The in-process arena passes ``None``:
    :class:`~komugi.eval.gumbel_mover.GumbelMover` searches the referee's
    position and carries no board of its own, so there is nothing to update.

    Args:
        state: The referee's position, mutated in place.
        prefix: The actions to replay, in order.
        announce: Called as ``announce(state, action)`` before each action is
            played, or ``None``.

    Returns:
        The replayed actions, so a caller can seed its move list with them.

    Raises:
        ValueError: If the prefix carries an action outside the running
            encoding, or one that is illegal or terminal here. Construction
            errors all: :class:`~komugi.eval.pool_books.PoolBook` verifies
            every entry by replay at construction, so reaching one of these is
            an engine that disagrees with the book, not a bad match config.
    """
    played: list[int] = []
    for action in prefix:
        if state.is_terminal():
            raise ValueError(
                f"the epoch book prefix passes through a terminal position at "
                f"ply {len(played)}: the engine had already adjudicated"
            )
        if not 0 <= action < _NUM_ACTIONS:
            # The range check ``pool_books.verify_prefix`` has, copied here
            # rather than assumed (review F6). Without it a negative id is
            # wrapped by numpy's negative indexing and can read ``True`` off the
            # far end of the mask -- at which point ``state.play(-1)`` is asked
            # to play an action nobody chose -- while an out-of-range positive
            # raises ``IndexError`` instead of the ``ValueError`` documented
            # below. This helper is public and a caller may bring its own
            # prefix, so it cannot rely on ``PoolBook`` having checked.
            raise ValueError(
                f"the epoch book prefix carries action {action!r} at ply "
                f"{len(played)}, outside [0, {_NUM_ACTIONS}) -- this process's "
                f"action encoding"
            )
        if not bool(state.legal_mask()[action]):
            raise ValueError(
                f"the epoch book prefix plays illegal action {action} at ply "
                f"{len(played)}: this engine cannot replay the line the book "
                f"claims, so the games would not be the openings it names"
            )
        if announce is not None:
            announce(state, action)
        state.play(action)
        played.append(int(action))
    if state.is_terminal():
        raise ValueError(
            f"the epoch book prefix leaves a terminal position after "
            f"{len(played)} plies: there is no game to play from it"
        )
    return tuple(played)


def new_arena_game(config: ArenaConfig) -> Game:
    """Return the referee one arena game is played on, wired to `config`.

    The **one** place a match's ``Game`` is constructed, module-level so that
    :func:`komugi.eval.v1bridge.play_bridged_game` calls it too. The in-process
    loop and the bridged loop must referee under the same rule for the same
    config -- a cross-family match refereed under a different key from the
    in-family one it is chained to would put the mismatch into the Elo instead
    of measuring it -- and one function is how that is guaranteed rather than
    remembered.

    Everything about the returned game except :attr:`ArenaConfig.repetition_rule`
    is the engine's default: the start position, the engine's ply cap, and every
    adjudication predicate. In particular this adds no cap, no contempt and no
    book; the contempt lives on the :class:`~komugi.selfplay.adapter.GameState`
    the caller wraps this in, and the book is replayed onto it afterwards.
    """
    return Game(repetition_rule=config.repetition_rule)


def _play_wired_game(
    white: _Side,
    black: _Side,
    config: ArenaConfig,
    seeds: np.random.SeedSequence,
    *,
    prefix: Sequence[int] = (),
) -> GameOutcome:
    """Play one game between two already-wired sides.

    The reference loop of :meth:`komugi.selfplay.worker.SelfPlayWorker.play_game`
    with the seat dispatch added. Three properties are load-bearing and each is
    one line below:

    * The mover is ``state.to_play()``, never ply parity. During the draft the
      same player can move twice in a row, so parity is not the seat.
    * The searching player's *own* budget and noise scale are used, because a
      match may deliberately pit two budgets against each other.
    * The loop condition is the engine's ``is_terminal()`` and nothing else.

    `prefix` is the pair's epoch-book opening (§D3.4), replayed onto the referee
    before the loop and counted into :attr:`GameOutcome.moves` -- they are moves
    the game was played through, and a caller replaying the list has to play
    them too. ``ply`` therefore starts at ``len(prefix)``, so the searched plies
    keep their **true** ply index and with it their per-move Gumbel seed. The
    default ``()`` is the bookless game, byte for byte.
    """
    state = GameState(new_arena_game(config), config.draw_score)
    moves: list[int] = list(replay_prefix(state, prefix))
    ply = len(moves)
    while not state.is_terminal():
        side = white if state.to_play() == _WHITE else black
        entrant = side.entrant
        search = move_search(
            side.scalar,
            side.wave,
            m=entrant.m,
            n=entrant.n,
            gumbel_scale=entrant.gumbel_scale,
            seed=move_seed(seeds, ply),
            proof_mode=entrant.proof_mode,
            discovery_mode=entrant.discovery_mode,
            tactical_provider=side.provider,
        )
        result = search.search(state)
        state.play(result.action)
        moves.append(int(result.action))
        ply += 1
    return GameOutcome(result=_result_of(state), moves=tuple(moves))


def play_arena_game(
    white: ArenaPlayer,
    black: ArenaPlayer,
    config: ArenaConfig,
    *,
    pair: int = 0,
    seat: int = 0,
) -> GameOutcome:
    """Play exactly one game, building both evaluators for it.

    :func:`play_match` is what a ladder run calls; this is the single-game door
    for a smoke test, a debugging replay, or a comparison against another
    driver. It builds one evaluator per player and throws them away with the
    game, which is why a match does not use it.

    Args:
        white: The entrant holding white.
        black: The entrant holding black.
        config: The match config. ``seed``, ``draw_score`` and ``book`` matter
            here; ``games`` is not consulted.
        pair: Pair index, for the seed key (:func:`game_seeds`) **and** for the
            epoch book's pair-to-entry map, so this door opens the same opening
            :func:`play_match` would have opened for the same pair.
        seat: Seating index within the pair, for the same key. ``0`` and ``1``
            in a match; any integer is a valid, distinct stream.

    Returns:
        The :class:`GameOutcome`: result from white's perspective, and moves.
    """
    pin_determinism()
    return _play_wired_game(
        _wire(white, config.draw_score),
        _wire(black, config.draw_score),
        config,
        game_seeds(config, pair, seat),
        prefix=config.prefix_for_pair(pair),
    )


def _summarise(
    records: Sequence[GameRecord], plies: Sequence[int]
) -> ArenaSummary:
    """Return the :class:`ArenaSummary` for a finished match."""
    draws = sum(1 for record in records if record.is_draw)
    return ArenaSummary(
        games=len(records),
        decisive=len(records) - draws,
        draws=draws,
        total_plies=sum(plies),
        shortest_plies=min(plies, default=0),
        longest_plies=max(plies, default=0),
    )


def play_match(a: ArenaPlayer, b: ArenaPlayer, config: ArenaConfig) -> MatchOutcome:
    """Play `config.games` games between `a` and `b` as seat-swapped pairs.

    Pair ``k`` is games ``2k`` and ``2k+1``: the first with `a` as white, the
    second with `b` as white, both under ``pair_id`` ``"<a>:<b>:<k:05d>"`` (the
    format :mod:`komugi.eval.synth` already generates and
    :func:`komugi.eval.seats.check_paired_books` already audits). The pair id
    names the entrants in the order they were passed, not in seat order, so both
    games of a pair carry the same one -- and so does the pair's ``book_id``,
    which is entry ``k``'s id when :attr:`ArenaConfig.book` carries an epoch
    book and :attr:`ArenaConfig.book_id` when it does not.

    Pair indices restart at ``00000`` every match, and the id carries no seed
    component -- so two matches of the *same* two players cannot be pooled into
    one :func:`~komugi.eval.seats.check_paired_books` audit ("play 100 more
    games and merge" collides on ``pair_id`` and fails loudly with GAME_COUNT
    violations). To extend a match, play a fresh, larger one, fit the two match
    sets separately, or -- when the two matches are deliberately distinct
    events, as a pool epoch's re-link is -- give them different
    :attr:`ArenaConfig.pair_namespace` s, which is the one config component the
    id does carry (:func:`pair_id_for`).

    Each player's evaluator is built **once**, here, and serves the whole match.
    A per-game factory call would reload a checkpoint per game -- seconds of
    torch work to produce the identical weights -- and a per-match one keeps a
    match to two live networks however long it runs.

    Args:
        a: One entrant. Holds white in the even-numbered game of every pair.
        b: The other. Must have a different ``player.id``.
        config: Game count, seed, contempt, and the epoch book (if any) whose
            entry ``k`` both games of pair ``k`` are opened with.

    Returns:
        A :class:`MatchOutcome` -- the records in play order, plus the summary.

    Raises:
        ValueError: If both entrants carry the same player id. Refused here,
            before any game is played, rather than by
            :class:`~komugi.eval.records.GameRecord` after the first one:
            self-matching a checkpoint (to measure draw rate, or to check the
            seat balance of the harness itself) is a legitimate thing to want,
            and the way to ask for it is two ``ArenaPlayer`` s carrying two
            distinct ids over the same weights (design review D14).
    """
    if a.player.id == b.player.id:
        raise ValueError(
            f"both entrants are {a.player.id!r}: a game record cannot name one "
            f"player in both seats. To play a checkpoint against itself, give "
            f"the two entrants distinct ids over the same evaluator factory"
        )
    pin_determinism()
    side_a = _wire(a, config.draw_score)
    side_b = _wire(b, config.draw_score)

    records: list[GameRecord] = []
    plies: list[int] = []
    for pair in range(config.pairs):
        pair_id = pair_id_for(config, a.player.id, b.player.id, pair)
        # One entry, both seats: the two games of a pair are the same opening
        # from opposite sides (§D3.4), which is what makes the audit's
        # "same book_id" clause a measurement rather than a tautology.
        book_id = config.book_id_for_pair(pair)
        prefix = config.prefix_for_pair(pair)
        for seat in (0, 1):
            white, black = (side_a, side_b) if seat == 0 else (side_b, side_a)
            outcome = _play_wired_game(
                white,
                black,
                config,
                game_seeds(config, pair, seat),
                prefix=prefix,
            )
            records.append(
                GameRecord(
                    white=white.entrant.player.id,
                    black=black.entrant.player.id,
                    result=outcome.result,
                    book_id=book_id,
                    pair_id=pair_id,
                )
            )
            # Searched plies, not played ones: the pair's replayed prefix is
            # not a game length (review F8, :class:`ArenaSummary`).
            plies.append(len(outcome.moves) - len(prefix))
    return MatchOutcome(tuple(records), _summarise(records, plies))


def _checkpoint_evaluator(
    path: Path, device: str
) -> Callable[[], BatchEvaluator]:
    """Return a factory that loads `path` and serves it on `device`.

    A closure over a *path*, not over a loaded model: a ladder of twenty
    checkpoints is twenty of these, and building every net up front would hold
    twenty networks in memory to play one match.

    The net is built by :func:`komugi.campaign.factory.new_net` -- the same
    constructor the campaign's trainer and servers use, so the arena cannot
    disagree with them about the shape -- and only the **weights** are loaded
    from the checkpoint; a checkpoint's optimiser state has nothing to say to
    inference. ``eval()`` is not optional: a net left in training mode updates
    its BatchNorm running statistics from whatever wave the search happened to
    submit, which would make a player's strength depend on its opponent.
    """

    def build() -> BatchEvaluator:
        """Load the checkpoint and wrap it as a batch evaluator."""
        checkpoint = load_checkpoint_file(path)
        if checkpoint is None:
            raise FileNotFoundError(
                f"{path} names no checkpoint that verifies; refusing to rate "
                f"freshly initialised weights under a checkpoint's name"
            )
        net = factory.new_net()
        # On the returned object, before load_state_dict: a checkpoint from an
        # older plane map is refused by name here rather than dying as a torch
        # size mismatch inside the load (KOM-002).
        require_matching_plane_count(checkpoint, net.config.trunk.in_planes)
        net.load_state_dict(checkpoint.model_state)
        net.to(device)
        net.eval()
        return InProcessEvaluator(
            net,
            # The seam's shapes, from the factory that owns them -- and the
            # factory reads its plane count off the encoder. Literals here would
            # be a second place for the plane count and 37341 to be written
            # down, and the plane count has already drifted once.
            InferenceConfig(
                in_planes=factory.IN_PLANES, num_actions=factory.NUM_ACTIONS
            ),
        )

    return build


def checkpoint_evaluator(
    path: str | os.PathLike[str], device: str = "cpu"
) -> Callable[[], BatchEvaluator]:
    """Return the ladder's own checkpoint factory, by its public name.

    Exactly :func:`_checkpoint_evaluator` -- the *same* loader
    :func:`ladder_players` builds every rung with, so a checkpoint seated at
    §D1's move seam (:class:`komugi.eval.gumbel_mover.GumbelMover`) and the same
    checkpoint seated as a ladder rung are the same weights on the same device by
    construction. A second loader written next door would be a second place for
    "only the weights, ``eval()``, and the plane-count guard" to be true.
    """
    return _checkpoint_evaluator(Path(path), device)


def ladder_players(
    ckpt_dir: str | os.PathLike[str],
    every_n: int = 1,
    device: str = "cpu",
) -> tuple[ArenaPlayer, ...]:
    """Return one :class:`ArenaPlayer` per sampled checkpoint, **oldest first**.

    The ladder driver: point it at a campaign's checkpoint directory and it
    produces the entrants a round-robin (or a chain of adjacent matches) is
    played between.

    :func:`komugi.train.checkpoint.list_checkpoints` returns **newest first**,
    and this function reverses it *before* sampling. That order matters twice
    over: a ladder reads as a training curve when it is in training order, and
    -- the part that would be a silent bug -- sampling every ``N`` th of a
    newest-first list anchors the sample to the *newest* checkpoint, so the set
    of checkpoints rated would change every time a new one was written and two
    runs of the same command would rate different players.

    Args:
        ckpt_dir: A directory of checkpoints. A missing directory reads as
            empty, exactly as ``list_checkpoints`` defines it.
        every_n: Sample stride over the ordered checkpoints. ``1`` rates all of
            them; ``5`` rates the 1st, 6th, 11th, ... oldest.
        device: Where each net is placed, e.g. ``"cpu"`` or ``"cuda"``. Passed
            to ``Module.to`` unchanged.

    Returns:
        The entrants, oldest first, at the default full-search budget. Ids are
        ``ckpt-<step>`` and the family is :data:`LADDER_FAMILY`; a caller who
        wants a different budget rebuilds them with
        ``dataclasses.replace(player, n=...)``.

    Raises:
        ValueError: If `every_n` is not positive.
    """
    if every_n < 1:
        raise ValueError(f"every_n must be >= 1, got {every_n!r}")
    ordered: tuple[CheckpointRef, ...] = tuple(reversed(list_checkpoints(ckpt_dir)))
    return tuple(
        ArenaPlayer(
            player=Player(id=f"ckpt-{ref.step}", family=LADDER_FAMILY),
            evaluator_factory=_checkpoint_evaluator(ref.path, device),
        )
        for ref in ordered[::every_n]
    )
