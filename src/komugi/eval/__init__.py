"""The evaluation ladder: how komugi's strength claims get measured.

The **measurement core is engine-independent by construction**, and everything
this module re-exports is part of it: nothing reachable from here imports a game
engine, a search or torch. It consumes :class:`~komugi.eval.records.GameRecord`
tuples and produces ratings, seat tables and audits, so the core can be tested
against synthetic data whose answer is known before any real engine plays a
move.

Layout:

* :mod:`komugi.eval.records` -- self-validating Player/GameRecord/MatchSet.
* :mod:`komugi.eval.elo` -- maximum-likelihood Elo under the Davidson draw
  model.
* :mod:`komugi.eval.seats` -- per-seat tallies and the paired-book audit that
  justifies the fit's seat-free assumption.
* :mod:`komugi.eval.synth` -- seeded synthetic games from planted ratings.
* :mod:`komugi.eval.arena` -- the games themselves: checkpoint-vs-checkpoint
  matches through the real engine and the real Gumbel search. Deliberately
  **not** imported below, and deliberately not in ``__all__``: it is the one
  module in the package that needs the engine, the search and torch, and
  re-exporting it would make ``import komugi.eval`` -- which the rating stack's
  own tests do -- pay for all three. Import it by name
  (``from komugi.eval.arena import play_match``) when you want to play games.
* :mod:`komugi.eval.ladder` -- the driver over the arena: a checkpoint
  directory becomes adjacent-rung matches, one pooled match set and one Elo
  fit, plus the report that comes with it. Not imported below either, and for
  the same reason: it reaches the engine, the search and torch through the
  arena. Import it by name (``from komugi.eval.ladder import run_ladder``).
* :mod:`komugi.eval.ab_endpoint` -- KOM-036's pre-registered A/B endpoint: the
  Wilson interval on a head-to-head score rate, the three-way decision rule it
  feeds, and the structural guards a **cross-arm** pool needs (namespaced ids,
  a connected match graph) before its Elo offset means anything. Part of the
  engine-independent core -- it consumes counts and ``GameRecord`` tuples, and
  touches :class:`~komugi.eval.arena.ArenaPlayer` for typing only -- but not
  re-exported below, because it is a driver for one experiment rather than
  part of the rating vocabulary. Import it by name
  (``from komugi.eval.ab_endpoint import ab_endpoint``).

The kom-011d cross-family pool adds eight modules
(``docs/design/kom-011d-ladder-pool.md``). **None is re-exported**, and the
rule is the one the list above already applies: this package's namespace is the
*rating vocabulary* -- records, ratings, seats, synthetic data -- and a driver
for one experiment is imported by name. **Five** of the eight are additionally
torch-free by design -- :mod:`~komugi.eval.pool`,
:mod:`~komugi.eval.interval`, :mod:`~komugi.eval.divergence`,
:mod:`~komugi.eval.rawmcts` and :mod:`~komugi.eval.pool_report` -- and all five
are pinned as such in a spawned interpreter: ``rawmcts`` by §6 node 29 and the
other four by node 42, which ``pool_report`` joined in KOM-140 (the T4 review's
F11 found the module claiming *"No torch"* with no node behind the claim, and
this sentence miscounting both halves). The split below is therefore
load-bearing rather than descriptive:

* :mod:`komugi.eval.pool` -- entrant identity (``family/member@sha8``), the
  roster, §D2.2's schedule and the epoch manifest. **No torch, at import time
  or through a function body**, so a schedule and a roster can be built and
  reviewed on a box with no engine installed. Not re-exported: an entrant id is
  the pool's vocabulary, not the ladder's.
* :mod:`komugi.eval.interval` -- §D5.2's paired bootstrap over ``pair_id``
  units, the percentile interval every published rating carries, and
  ``fit_statistic``, the seam through which a point estimate and its own
  interval are computed by one code path. No torch. Not re-exported: it is an
  estimator's companion, and a caller that wants an interval already knows
  which fit it is around.
* :mod:`komugi.eval.divergence` -- §D5's in/cross stop-gate as a predicate: the
  three gains, the two windows, the six validity conditions, and thresholds
  that take no argument to override. No torch. Not re-exported, because
  ``THRESHOLD_SINGLE_ELO`` in this namespace would read as a general constant
  rather than as one experiment's pre-registration.
* :mod:`komugi.eval.pool_books` -- §D3's frozen, hash-versioned epoch book: the
  ``K = 128`` selection from a KOM-014 book, its refusals, and the pair-to-entry
  map. Imported by :mod:`komugi.eval.arena`, so it stays out of the re-export
  list for the same reason the arena does.
* :mod:`komugi.eval.rawmcts` -- §D6.3's net-free ruler: uniform prior, random
  rollouts, PUCT selection, seeded from the ``Mover`` seam. **No torch**, which
  is what its local six-operation ``Protocol`` over the game state buys. Not
  re-exported: it is an opponent, not a rating.
* :mod:`komugi.eval.rawmcts_entrant` -- the seam that seats that ruler in a
  match, returning a ``BridgeEntrant``. **Torch-tolerant by necessity**
  (``v1bridge.py:67`` -> ``arena.py:81``) and claims nothing otherwise; the
  split from :mod:`komugi.eval.rawmcts` is the whole point, and re-exporting it
  here would put torch behind ``import komugi.eval``.
* :mod:`komugi.eval.pool_run` -- §D9's epoch driver: the R5 gate before any
  game, §D2.2's schedule, the paired-book audit before the fit, four fits, the
  bootstrap, both windows' gates, the report and the JSON record. Reaches the
  engine through the arena and the bridge, so it is deliberately **not**
  imported below. Import it by name (``from komugi.eval.pool_run import
  run_epoch``).
* :mod:`komugi.eval.pool_report` -- §D9.1's renderer: banners above the tables,
  a rating table whose anchor flag is read from the record, the per-seat block
  that says ``clean, 0 violations`` out loud, the two divergence rows and the
  Elo-vs-visits block. No torch (it copies ``ladder._table``'s eight-line idiom
  rather than importing a private name from a module that reaches one), and
  **pinned** as such by §6 node 42's child, which renders a planted report
  before it looks at ``sys.modules``. Not re-exported: it renders one
  experiment's page.

Three more modules predate that design and are named here because *"every
module, with its rationale"* is the property this docstring is supposed to have:

* :mod:`komugi.eval.gumbel_mover` -- the arena's Gumbel search behind the
  cross-engine ``Mover`` seam, plus ``CountedEvaluator``, which is what makes
  KOM-023's 257-evaluation ceiling a counted quantity rather than a claim.
  Reaches torch through the arena; imported by name.
* :mod:`komugi.eval.v1bridge` -- KOM-023's cross-engine harness: the wire, the
  handshake, the ten void modes, pair-atomic scoring and the 2% admissibility
  gate. Torch-tolerant, large, and imported by name.
* :mod:`komugi.eval.r5` -- KOM-023 §D3's pre-flight and its three-way branch.
  ``assert_rated_run_permitted`` is what every rated driver calls before its
  first game; it is imported by name, at the one call site that needs it, so
  that "no rated run without a committed Δmean" is a statement in a driver
  rather than a package-level import nobody reads.
  (:mod:`komugi.eval.v1runner` is v1's own side of that wire, consumed and
  never edited.)
"""

from __future__ import annotations

from komugi.eval.elo import EloResult, fit_elo, log_likelihood
from komugi.eval.records import (
    BLACK_WIN,
    DRAW,
    KNOWN_FAMILIES,
    VALID_RESULTS,
    WHITE_WIN,
    GameRecord,
    MatchSet,
    Player,
    is_known_family,
    roster_families,
)
from komugi.eval.seats import (
    BOOK_MISMATCH,
    GAME_COUNT,
    SEATS_NOT_SWAPPED,
    PairViolation,
    SeatTally,
    check_paired_books,
    seat_matrix,
)
from komugi.eval.synth import davidson_probabilities, simulate_match_set

__all__ = [
    "BLACK_WIN",
    "BOOK_MISMATCH",
    "DRAW",
    "GAME_COUNT",
    "KNOWN_FAMILIES",
    "SEATS_NOT_SWAPPED",
    "VALID_RESULTS",
    "WHITE_WIN",
    "EloResult",
    "GameRecord",
    "MatchSet",
    "PairViolation",
    "Player",
    "SeatTally",
    "check_paired_books",
    "davidson_probabilities",
    "fit_elo",
    "is_known_family",
    "log_likelihood",
    "roster_families",
    "seat_matrix",
    "simulate_match_set",
]
