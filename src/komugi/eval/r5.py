"""Run R5 -- the ``Δmean`` pre-flight -- and the rated-run gate it decides.

§D3.4 showed that P4's equality ("the two engines spend the same mean evaluations
per move") is entailed by nothing: v1 is forced at plies where Komugi is not, the
two in-tree repetition predicates fire at different rates, and the two sides move
at *disjoint* plies, so neither the sign nor the magnitude of ``Δmean`` was known.
Rev 2 gated the anchor on that unmeasured quantity and would have discovered the
breach after 500 rated games, at which point the only available repair -- widening
P4 -- is the act §D3's opening paragraph forbids.

Rev 3 replaces the premise with a measurement and pre-registers the response.
This module is both halves:

* :func:`run_r5` plays the pre-flight -- **10 pairs = 20 games, Komugi vs v1, at
  the rated per-move ceiling of 257** -- and reports ``mean_K``, ``mean_v1``,
  ``Δmean = mean_K − mean_v1``, its per-pair spread, and the per-game counts of
  the two mechanisms that produce it (diagnostic D-1 and ``no_move``/M5 plies).
  What shrinks relative to the rated run is the **pair count and nothing else**:
  ``Δmean`` is a per-move quantity, so a reduced per-move ceiling would measure a
  different number and would not transfer (§D3.5).
* :func:`p4_branch` is §D3.6's three-way branch, decided **now**, as a pure
  function of the committed number: ``|Δmean| ≤ 5.14`` proceeds, ``Δmean < −5.14``
  proceeds one-sided, ``Δmean > +5.14`` refuses the ceiling outright. In none of
  the three arms is P4's own threshold moved -- that is the entire point of
  deciding them before the number exists.
* :func:`rated_run_branch` is A24's refusal: the rated-run entry point **will not
  start** without a committed R5 artefact, and it dispatches on the branch above
  rather than on an operator's judgement. An absent ``delta_mean`` is not a
  default to proceed on; it is the one state in which no rated game may be played.

**The run itself is not run here.** This module is the driver and the gate; R5 is
Tier 2, it needs the v1 checkout, and its output is committed to
``tickets/KOM-023.md`` before any rated game.

Why the driver instruments rather than re-implements the loop
------------------------------------------------------------
Everything R5 measures is per-*ply* and per-*game*, and
:func:`~komugi.eval.v1bridge.play_bridged_match` reports neither: it returns
scored :class:`~komugi.eval.records.GameRecord` s and §D6's void tally. The
temptation is to write a second game loop that also counts. That would put a
second, silently divergent pairing and seeding schedule in the repository --
exactly the failure §D5 pre-registers ``move_seed_for`` against -- and R5's whole
value depends on its games being *the same games* the rated run will play.

So the loop is the bridge's own, and the instrumentation rides in on the movers:
:func:`run_r5` wraps each entrant's ``mover_factory`` in :class:`_R5Probe`, a
:class:`~komugi.eval.v1bridge.Mover` that records what passed through it and
delegates everything else. Pair atomicity, §D6's voids, the crash respawn, the
seed derivation and the start mode are then not reproduced here at all.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Final, cast

import numpy as np

from komugi.engine.action_encoding import A1_START, ACTION_SIZE, B_START, C_START
from komugi.engine.utils import SQUARES
from komugi.eval.arena import ArenaConfig
from komugi.eval.v1bridge import (
    COMPLETED,
    EVAL_CEILING,
    M5,
    M8,
    BridgeEntrant,
    BridgedMatch,
    BridgeVoided,
    BudgetBreach,
    MoveRecord,
    Mover,
    bridged_book_id,
    play_bridged_match,
    resolved_gungi_mode,
)
from komugi.selfplay.adapter import GameState

# ---------------------------------------------------------------------------
# what R5 is, as literals rather than as prose in a ticket
# ---------------------------------------------------------------------------

R5_PAIRS: Final[int] = 10
"""§D3.5: ten seat-swapped pairs. The pair count is what shrinks."""

R5_GAMES: Final[int] = R5_PAIRS * 2
"""Twenty games. P4's alternative discovery path costs 500."""

R5_RUN_SEED: Final[int] = 0xC0FFEE
"""The pre-registered ``run_seed``, so that two people running "the
pre-registered pre-flight" get the same games (review F8.2). The same literal R3
is pinned to."""

R5_DRAW_SCORE: Final[float] = 0.0
"""§D3.4 pre-registers ``draw_score = 0.0`` for every cross-family match."""

P4_TOLERANCE: Final[float] = 5.14
"""P4's budget: 2% of the ceiling of 257. **Never widened after seeing data.**"""

PROCEED: Final[str] = "proceed"
"""§D3.6 branch 1: ``|Δmean| ≤ 5.14``. The rated run proceeds."""

PROCEED_ONE_SIDED: Final[str] = "proceed_one_sided"
"""§D3.6 branch 2: ``Δmean < −5.14``. The rated run proceeds, but the anchor may
be cited **only** as a lower bound at a v1-favouring compute imbalance."""

REFUSE: Final[str] = "refuse"
"""§D3.6 branch 3: ``Δmean > +5.14``. No rated anchor run at this ceiling."""

DELTA_MEAN_FIELD: Final[str] = "delta_mean"
"""The artefact key the gate reads. Named once so :meth:`R5Result.to_json` and
:func:`rated_run_branch` cannot drift into writing and reading different keys --
which would make the gate silently unsatisfiable, or silently vacuous."""

BRANCH_FIELD: Final[str] = "branch"
"""Where :meth:`R5Result.to_json` records the branch it derived. The gate
**re-derives** it and refuses a committed artefact whose stored branch disagrees;
see :func:`rated_run_branch`."""

KOMUGI_SIDE: Final[str] = "K"
V1_SIDE: Final[str] = "v1"
"""The two sides, as the keys the per-game tallies are held under."""

_SQUARES: Final[int] = len(SQUARES)
"""81. Every action-id block is plane-major over the destination square."""


# ---------------------------------------------------------------------------
# diagnostic D-1 (A18): the ids Komugi can play and v1 cannot name
# ---------------------------------------------------------------------------


def ids_without_v1_preimage(action_ids: Iterable[int]) -> tuple[int, ...]:
    """Return the ids in `action_ids` that **no v1 action id decodes to**.

    Diagnostic D-1 (A18), and the measurement bias B3 is booked against. §D2.3:
    ``v1 → Komugi is injective but not surjective``, and the direction rev 1 did
    not draw is the one that matters -- when it is v1's turn, v1 cannot *choose*
    a move outside the image, and **no failure mode detects it**, because the
    move v1 does play is legal under the referee.

    The image is read straight off the two layouts, and the arithmetic checks:
    v1's policy head is 15390 wide
    (``docs/reference-v1/alphazero-env/action_encoding.py:36``) ``= 81 × 190``,
    i.e. 190 planes against Komugi's 461 -- 81 route/capture, 81 tsuke/betray
    with **no** conversion selector, 14 arata and 14 arata-with-draft-finish with
    **no** betray bit. So an id is in the image iff:

    * **block A0** (route / capture) -- always. v1 carries no selector here and
      neither does Komugi.
    * **block A1** (tsuke / betray) -- iff its selector is the *maximal*
      conversion available for that ``(from_sq, to_sq)``. v1 says so in its own
      words (``docs/reference-v1/alphazero-env/action_encoding.py:18-24``): "we
      canonicalize to the MAXIMAL conversion (the betray with the most
      captured/converted pieces). Partial betrayals and plain tsuke when a betray
      exists at that square pair are deliberately UNREACHABLE action ids."
    * **blocks B and C** (arata) -- iff ``bsel == 0``. "Arata conversions were
      entirely unreachable under the v1 layout (no id decoded to one); they are
      ordinary actions here" (``action_encoding.py:36-38``).

    **Maximal is read off the legal set, not recomputed.** The legal selectors at
    a square are exactly the submasks of the qualifying mask ``Q``
    (:func:`komugi.engine.action_encoding.qualifying_mask`), and ``Q`` is its own
    largest submask -- so the maximal selector present in a ``(from_sq, to_sq)``
    group *is* that square's maximal conversion. Taking it from the ids rather
    than from the board keeps this function pure over the legal set, which is
    what lets a test plant one. Maximality is ordered by **converted-piece
    count** first, as v1 words it, and by selector value only to break a tie that
    the submask structure makes unreachable anyway.

    The reading is deliberately **conservative**: if move generation ever
    withheld the maximal combination while offering a partial one, the group's
    maximum would be that partial one and the ply would go uncounted. D-1 is a
    lower bound on the asymmetry, and a lower bound is the safe direction for a
    diagnostic whose large values are the alarming ones.

    Args:
        action_ids: The ids Komugi's referee calls legal in one position.

    Returns:
        The subset with no v1 pre-image, ascending. Empty is the good answer.

    Raises:
        ValueError: If an id is outside ``[0, ACTION_SIZE)`` -- an id from
            another action space, which this predicate would answer nonsense for.
    """
    stack_sel: dict[tuple[int, int], tuple[int, int]] = {}
    stack_ids: list[tuple[int, tuple[int, int], int]] = []
    orphans: list[int] = []

    for raw in action_ids:
        action = int(raw)
        if not 0 <= action < ACTION_SIZE:
            raise ValueError(
                f"action id {action} is outside Komugi's action space "
                f"[0, {ACTION_SIZE}): D-1 is a statement about *this* layout, "
                f"and an id from another one has no meaningful pre-image here"
            )
        if action < A1_START:
            continue  # block A0: v1 names every route and every capture.
        if action < B_START:
            plane, to_sq = divmod(action - A1_START, _SQUARES)
            from_sq, sel = divmod(plane, 4)
            key = (from_sq, to_sq)
            stack_ids.append((action, key, sel))
            rank = (sel.bit_count(), sel)
            if rank > stack_sel.get(key, (-1, -1)):
                stack_sel[key] = rank
            continue
        block = B_START if action < C_START else C_START
        plane, _to_sq = divmod(action - block, _SQUARES)
        _piece_index, bsel = divmod(plane, 2)
        if bsel:
            orphans.append(action)

    orphans.extend(
        action
        for action, key, sel in stack_ids
        if (sel.bit_count(), sel) != stack_sel[key]
    )
    return tuple(sorted(orphans))


def d1_orphans(state: GameState) -> tuple[int, ...]:
    """Return :func:`ids_without_v1_preimage` over `state`'s legal set.

    The referee's own mask (``state.legal_mask()``), because §D6 makes Komugi's
    ``Game`` the sole referee: D-1 asks what *this* position offered, not what
    either search happened to look at.
    """
    return ids_without_v1_preimage(np.flatnonzero(state.legal_mask()).tolist())


# ---------------------------------------------------------------------------
# what one R5 game and one R5 pair came to
# ---------------------------------------------------------------------------


def _mean(values: Sequence[int]) -> float | None:
    """Return the arithmetic mean of `values`, or ``None`` if there are none.

    ``None`` rather than ``0.0`` or ``nan``: a mean over no plies does not exist,
    and every consumer downstream -- P4's branch above all -- must be made to say
    what it does about that rather than quietly average it in as a zero.
    """
    if not values:
        return None
    return sum(values) / len(values)


@dataclass(frozen=True, slots=True)
class R5Game:
    """One game of the pre-flight: what each side spent, and the two diagnostics.

    Args:
        pair: Pair index.
        seat: Seat index within the pair (0 or 1); the Komugi entrant holds white
            in seat 0 and black in seat 1, which is §D4's swap.
        evals_k: Komugi's ``evals_used``, one entry per ply **Komugi moved**, in
            ply order. Forced plies (``evals == 1``) and the turn-capped final
            ply are ordinary entries here: P4 is stated over every ply at which
            that side moved, and an exclusion would be a second, unregistered
            definition of the mean (§D3.6, re-review F8).
        evals_v1: v1's, likewise.
        d1_plies: Plies of this game at which the referee's legal set held an id
            with no v1 pre-image (D-1, A18). Counted at **every** ply, whoever
            was to move.
        d1_plies_v1_to_move: The subset at which **v1** was to move -- the plies
            at which B3 actually costs v1 a choice.
        d1_ids: Total such ids summed over the game's plies, so a count of "one
            ply, one orphan" is distinguishable from "one ply, forty".
        no_move_plies: ``no_move``/M5 plies (mechanism 2's observable shadow). At
            most one per game: an M5 voids the pair and ends the game.
        seeds: The Gumbel seed handed to each ply, in ply order. Not part of the
            artefact -- it is here so a test can assert, without a recording
            stub, that R5's plies were seeded by
            :func:`~komugi.eval.arena.move_seed_for` and not by a second schedule
            that happens to agree today (§D5, design F10d).
        void_mode: The §D6 mode this game ended on, or ``None`` if it finished.
        scored: Whether this game's pair was scored. **Only scored pairs reach a
            mean** (§D3.6: "every mean in P4 and P5 is computed over the plies of
            scored pairs only"); the diagnostics are reported for every game
            played, because a ``no_move`` count restricted to scored pairs would
            be identically zero by construction.
    """

    pair: int
    seat: int
    evals_k: tuple[int, ...] = ()
    evals_v1: tuple[int, ...] = ()
    d1_plies: int = 0
    d1_plies_v1_to_move: int = 0
    d1_ids: int = 0
    no_move_plies: int = 0
    seeds: tuple[int, ...] = ()
    void_mode: str | None = None
    scored: bool = False

    @property
    def plies(self) -> int:
        """Return how many plies this game ran, both sides together."""
        return len(self.evals_k) + len(self.evals_v1)

    def to_json(self) -> dict[str, Any]:
        """Return this game's row of the artefact."""
        return {
            "pair": self.pair,
            "seat": self.seat,
            "komugi_seat": "white" if self.seat == 0 else "black",
            "scored": self.scored,
            "void_mode": self.void_mode,
            "plies": self.plies,
            "plies_K": len(self.evals_k),
            "plies_v1": len(self.evals_v1),
            "mean_K": _mean(self.evals_k),
            "mean_v1": _mean(self.evals_v1),
            "d1_plies": self.d1_plies,
            "d1_plies_v1_to_move": self.d1_plies_v1_to_move,
            "d1_ids": self.d1_ids,
            "no_move_plies": self.no_move_plies,
            # Kept verbatim so the committed number can be recomputed from the
            # committed artefact by someone who does not trust this module.
            "evals_K": list(self.evals_k),
            "evals_v1": list(self.evals_v1),
        }


@dataclass(frozen=True, slots=True)
class R5Pair:
    """One **scored** pair's means -- §D3.5's "per-pair spread"."""

    pair: int
    mean_k: float | None
    mean_v1: float | None
    delta_mean: float | None
    plies_k: int
    plies_v1: int

    def to_json(self) -> dict[str, Any]:
        """Return this pair's row of the artefact."""
        return {
            "pair": self.pair,
            "mean_K": self.mean_k,
            "mean_v1": self.mean_v1,
            DELTA_MEAN_FIELD: self.delta_mean,
            "plies_K": self.plies_k,
            "plies_v1": self.plies_v1,
        }


@dataclass(frozen=True, slots=True)
class P4Branch:
    """Which of §D3.6's three pre-registered arms a ``Δmean`` falls in.

    A value object rather than a bare string, because two of the three arms
    permit the rated run and only one of *those* two permits the anchor to be
    cited symmetrically -- a distinction a caller that received ``"proceed"`` or
    ``"proceed_one_sided"`` as text would have to re-derive, and could re-derive
    wrongly.
    """

    branch: str
    delta_mean: float
    tolerance: float
    rationale: str

    @property
    def rated_run_permitted(self) -> bool:
        """Return whether a rated anchor run may be played at this ceiling."""
        return self.branch != REFUSE

    @property
    def one_sided_only(self) -> bool:
        """Return whether the anchor may be cited **only** as a lower bound."""
        return self.branch == PROCEED_ONE_SIDED

    def to_json(self) -> dict[str, Any]:
        """Return the branch as the artefact records it."""
        return {
            "branch": self.branch,
            DELTA_MEAN_FIELD: self.delta_mean,
            "tolerance": self.tolerance,
            "rated_run_permitted": self.rated_run_permitted,
            "one_sided_only": self.one_sided_only,
            "rationale": self.rationale,
        }


def p4_branch(delta_mean: float | None) -> P4Branch:
    """Return §D3.6's pre-registered branch for `delta_mean`.

    A pure function, decided before the number existed, of the number alone::

        |Δmean| ≤ 5.14   → proceed
        Δmean  < −5.14   → proceed, anchor citable one-sided only
        Δmean  > +5.14   → refuse: no rated anchor run at this ceiling

    The tolerance is **not** a parameter. Widening P4 after seeing data is
    forbidden by §D3's opening paragraph, and a ``tolerance=`` argument is
    precisely how a pre-registration gets quietly retuned after its first
    failure.

    Args:
        delta_mean: ``mean_K − mean_v1`` from the committed R5 result.

    Returns:
        The arm, with the two permissions it carries.

    Raises:
        R5Missing: If `delta_mean` is ``None``. A run whose ``Δmean`` does not
            exist is A24's blocking state, not a proceed.
    """
    if delta_mean is None:
        raise R5Missing(
            "R5 reported no Δmean -- the pre-flight has no scored pair to take a "
            "mean over. The rated run does not start (§D3.5, A24): an absent "
            "measurement is the one state in which no rated game may be played, "
            "and it is not a default to proceed on"
        )
    delta = float(delta_mean)
    if abs(delta) <= P4_TOLERANCE:
        return P4Branch(
            branch=PROCEED,
            delta_mean=delta,
            tolerance=P4_TOLERANCE,
            rationale=(
                f"|Δmean| = {abs(delta):.4f} ≤ {P4_TOLERANCE} (§D3.6 branch 1): "
                f"the rated run proceeds and P4 is re-asserted on the rated match "
                f"itself. A breach there is resolved by branch 2 or 3 on the sign "
                f"observed there and the match is reported as inadmissible(P4, "
                f"Δmean); it is not re-run with a widened P4, and it is not re-run "
                f"at all -- re-running until a match passes is selection on the gate"
            ),
        )
    if delta < -P4_TOLERANCE:
        return P4Branch(
            branch=PROCEED_ONE_SIDED,
            delta_mean=delta,
            tolerance=P4_TOLERANCE,
            rationale=(
                f"Δmean = {delta:.4f} < −{P4_TOLERANCE} (§D3.6 branch 2): v1 spent "
                f"materially more compute than Komugi. The rated run proceeds, but "
                f"the anchor may be reported ONLY as one-sided -- 'Komugi − v1 ≥ g, "
                f"measured at a v1-favouring compute imbalance of "
                f"{abs(delta) / EVAL_CEILING:.4%} of the per-move budget'. Usable as "
                f"a lower bound for a P0 gate; it may NEVER be cited as a symmetric "
                f"'Komugi vs v1 at equal compute' number, in KOM-011 or anywhere else"
            ),
        )
    return P4Branch(
        branch=REFUSE,
        delta_mean=delta,
        tolerance=P4_TOLERANCE,
        rationale=(
            f"Δmean = {delta:.4f} > +{P4_TOLERANCE} (§D3.6 branch 3): Komugi spent "
            f"materially more. No rated anchor run at this ceiling. The "
            f"conservatism condition fails by more than the gate's own tolerance, "
            f"so the resulting number would flatter Komugi by an unmeasured amount "
            f"in exactly the direction §D3.7 identifies as the dangerous one. "
            f"KOM-023 reports that equal-evaluations parity is not achievable "
            f"between these two engines at {EVAL_CEILING} without a correction the "
            f"design does not pre-register, and KOM-011 may not carry a "
            f"single-number v1 anchor unless §D3 is re-pre-registered (a rev 4) "
            f"around the measured asymmetry -- before any rated game"
        ),
    )


@dataclass(frozen=True, slots=True)
class R5Result:
    """The pre-flight's whole output: one object, one JSON file, one gate.

    Args:
        games: Every game played, scored or voided, in play order.
        pairs: The **scored** pairs' means, for §D3.5's per-pair spread.
        status: :data:`~komugi.eval.v1bridge.COMPLETED`, ``aborted`` or
            ``voided``, straight from the match.
        reason: The match's reason string; ``""`` when it completed.
        pairs_attempted: Pairs the match began.
        pairs_scored: Pairs it scored. ``0`` for anything that did not complete:
            "voids are non-random, so the surviving pairs of an aborted match are
            exactly the biased sample the 2% gate exists to refuse" (§D6).
        ceiling: The per-move ceiling the run was played at, on both sides.
        run_seed: The match's ``run_seed``.
        config: The §D3.6 P5 configuration fields this run can vouch for.
    """

    games: tuple[R5Game, ...]
    pairs: tuple[R5Pair, ...]
    status: str
    reason: str
    pairs_attempted: int
    pairs_scored: int
    ceiling: int
    run_seed: int
    config: Mapping[str, Any]

    @property
    def scored_games(self) -> tuple[R5Game, ...]:
        """Return the games of scored pairs -- the only plies a mean sees."""
        return tuple(game for game in self.games if game.scored)

    @property
    def evals_k(self) -> tuple[int, ...]:
        """Return every Komugi ply of every scored pair, in play order."""
        return tuple(
            evals for game in self.scored_games for evals in game.evals_k
        )

    @property
    def evals_v1(self) -> tuple[int, ...]:
        """Return every v1 ply of every scored pair, in play order."""
        return tuple(
            evals for game in self.scored_games for evals in game.evals_v1
        )

    @property
    def mean_k(self) -> float | None:
        """Return ``mean_K``: P4's mean over every ply at which Komugi moved.

        Pooled over plies rather than averaged over per-pair means: P4 says "over
        every ply at which that side moved in a scored pair", and a mean of means
        would silently weight a 40-ply game like a 400-ply one.
        """
        return _mean(self.evals_k)

    @property
    def mean_v1(self) -> float | None:
        """Return ``mean_v1``, the same way."""
        return _mean(self.evals_v1)

    @property
    def delta_mean(self) -> float | None:
        """Return ``Δmean = mean_K − mean_v1``, or ``None`` if either is absent."""
        mean_k, mean_v1 = self.mean_k, self.mean_v1
        if mean_k is None or mean_v1 is None:
            return None
        return mean_k - mean_v1

    @property
    def pair_deltas(self) -> tuple[float, ...]:
        """Return each scored pair's own ``Δmean``, for the spread."""
        return tuple(
            pair.delta_mean for pair in self.pairs if pair.delta_mean is not None
        )

    @property
    def branch(self) -> P4Branch:
        """Return §D3.6's branch for this run's ``Δmean``.

        Raises:
            R5Missing: If this run has no ``Δmean`` (:func:`p4_branch`).
        """
        return p4_branch(self.delta_mean)

    def to_json(self) -> dict[str, Any]:
        """Return the machine-readable artefact -- one file, every R5 number.

        The branch is written **as well as** ``Δmean``, and
        :func:`rated_run_branch` re-derives it from ``Δmean`` and refuses an
        artefact where the two disagree. A stored decision that could not be
        checked against its own input is an operator's judgement wearing a field
        name, which is the thing A24 exists to remove.
        """
        deltas = self.pair_deltas
        artefact: dict[str, Any] = {
            "run": "R5",
            "design": "kom-023 §D3.5 / §D3.6 P4, acceptance A24",
            "purpose": (
                "the Δmean pre-flight: measure mean_K − mean_v1 at the rated "
                "per-move ceiling before the rated run, and dispatch on §D3.6's "
                "pre-registered three-way branch"
            ),
            "ceiling": self.ceiling,
            "run_seed": self.run_seed,
            "pairs_requested": R5_PAIRS,
            "games_requested": R5_GAMES,
            "status": self.status,
            "reason": self.reason,
            "pairs_attempted": self.pairs_attempted,
            "pairs_scored": self.pairs_scored,
            "games_played": len(self.games),
            "plies_K": len(self.evals_k),
            "plies_v1": len(self.evals_v1),
            "mean_K": self.mean_k,
            "mean_v1": self.mean_v1,
            DELTA_MEAN_FIELD: self.delta_mean,
            "mean_K_fraction_of_ceiling": (
                None if self.mean_k is None else self.mean_k / self.ceiling
            ),
            "mean_v1_fraction_of_ceiling": (
                None if self.mean_v1 is None else self.mean_v1 / self.ceiling
            ),
            "max_evals_K": max(self.evals_k, default=None),
            "max_evals_v1": max(self.evals_v1, default=None),
            "per_pair_delta_mean": {
                "min": min(deltas, default=None),
                "max": max(deltas, default=None),
                "range": (max(deltas) - min(deltas)) if deltas else None,
            },
            "d1_plies_total": sum(game.d1_plies for game in self.games),
            "d1_plies_v1_to_move_total": sum(
                game.d1_plies_v1_to_move for game in self.games
            ),
            "d1_ids_total": sum(game.d1_ids for game in self.games),
            "no_move_plies_total": sum(game.no_move_plies for game in self.games),
            "config": dict(self.config),
            "per_pair": [pair.to_json() for pair in self.pairs],
            "per_game": [game.to_json() for game in self.games],
        }
        artefact[BRANCH_FIELD] = (
            None if self.delta_mean is None else self.branch.to_json()
        )
        return artefact

    def write_json(self, path: str | Path) -> Path:
        """Write :meth:`to_json` to `path` and return it -- the committed artefact."""
        destination = Path(path)
        destination.write_text(
            json.dumps(self.to_json(), indent=2, sort_keys=False) + "\n",
            encoding="utf-8",
        )
        return destination


# ---------------------------------------------------------------------------
# A24's refusal: no committed R5, no rated run
# ---------------------------------------------------------------------------


class R5Missing(RuntimeError):
    """The rated run was asked to start without a committed ``Δmean``.

    A24: "a test asserts that the rated-run entry point **refuses to start** when
    R5's committed ``Δmean`` is absent". The message always names the artefact --
    a refusal that does not say *which* file is missing is a refusal an operator
    resolves by guessing.
    """


class RatedRunRefused(RuntimeError):
    """§D3.6 branch 3: the committed ``Δmean`` forbids a rated run at 257.

    Distinct from :class:`R5Missing` on purpose. "The number is not there" and
    "the number says no" have different repairs -- run R5, versus re-pre-register
    §D3 as a rev 4 around the measured asymmetry -- and a caller that caught one
    exception type for both would report the second as the first.
    """


def read_r5_artefact(path: str | Path) -> dict[str, Any]:
    """Return the committed R5 artefact at `path`.

    Raises:
        R5Missing: If the file does not exist, is not readable JSON, or is not a
            JSON object. Every message names `path`.
    """
    location = Path(path)
    if not location.is_file():
        raise R5Missing(
            f"no committed R5 result at {location}: run R5 (§D3.5 -- 10 pairs, "
            f"20 games, ceiling {EVAL_CEILING}, run_seed {R5_RUN_SEED:#x}) and "
            f"commit its artefact to tickets/KOM-023.md before any rated game. "
            f"Its absence blocks the rated run outright (A24)"
        )
    try:
        payload = json.loads(location.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise R5Missing(
            f"the committed R5 result at {location} is not readable JSON "
            f"({error}); the rated run does not start on an artefact that cannot "
            f"be parsed (A24)"
        ) from error
    if not isinstance(payload, dict):
        raise R5Missing(
            f"the committed R5 result at {location} is a "
            f"{type(payload).__name__}, not a JSON object carrying "
            f"{DELTA_MEAN_FIELD!r}"
        )
    return cast("dict[str, Any]", payload)


def rated_run_branch(path: str | Path) -> P4Branch:
    """Return the branch the committed R5 result at `path` dispatches the rated run on.

    A24, in full: the rated-run entry point **refuses to start** without a
    committed ``Δmean``, and it "dispatches on the three-way branch of §D3.6 --
    proceed, proceed-one-sided, or refuse -- rather than on an operator's
    judgement".

    The branch is re-derived from ``Δmean`` and then checked against the branch
    the artefact recorded. A disagreement is refused rather than resolved: one of
    the two was produced by something that is not :func:`p4_branch`, and there is
    no reading of A24 under which a rated run starts on a decision whose
    provenance is unknown.

    Args:
        path: The committed artefact (:meth:`R5Result.write_json`).

    Returns:
        The arm, which the caller must still honour -- see
        :func:`assert_rated_run_permitted` for the enforcing form.

    Raises:
        R5Missing: If the artefact is absent, unparseable, carries no
            ``delta_mean``, carries a null or non-numeric one, or records a
            branch that disagrees with the one ``delta_mean`` implies.
    """
    location = Path(path)
    payload = read_r5_artefact(location)
    if DELTA_MEAN_FIELD not in payload:
        raise R5Missing(
            f"the committed R5 result at {location} carries no "
            f"{DELTA_MEAN_FIELD!r} field: the rated run does not start (A24). "
            f"P4 is gated on a pre-registered measurement, and a rated run "
            f"started without one is a run gated on nothing"
        )
    raw = payload[DELTA_MEAN_FIELD]
    if raw is None:
        raise R5Missing(
            f"the committed R5 result at {location} reports "
            f"{DELTA_MEAN_FIELD}=null -- R5 ran but scored no pair, so no mean "
            f"exists. The rated run does not start (A24)"
        )
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise R5Missing(
            f"the committed R5 result at {location} reports "
            f"{DELTA_MEAN_FIELD}={raw!r}, which is not a number: P4's branch is "
            f"a function of a measured quantity and there is none here"
        )
    branch = p4_branch(float(raw))

    recorded = payload.get(BRANCH_FIELD)
    if isinstance(recorded, dict):
        stated = recorded.get("branch")
        if stated != branch.branch:
            raise R5Missing(
                f"the committed R5 result at {location} records branch "
                f"{stated!r} but its {DELTA_MEAN_FIELD} of {float(raw)!r} implies "
                f"{branch.branch!r} (§D3.6, tolerance {P4_TOLERANCE}). One of the "
                f"two did not come from the pre-registered branch function; the "
                f"rated run does not start on a decision of unknown provenance"
            )
    return branch


def assert_rated_run_permitted(path: str | Path) -> P4Branch:
    """Return the branch, raising unless the rated run may actually be played.

    The enforcing form of :func:`rated_run_branch`, for the rated-run entry
    point itself. Branch 2 returns -- the run proceeds -- and the returned
    :attr:`P4Branch.one_sided_only` is what the reporting side must honour.

    Raises:
        R5Missing: As :func:`rated_run_branch`.
        RatedRunRefused: On §D3.6 branch 3.
    """
    branch = rated_run_branch(path)
    if not branch.rated_run_permitted:
        raise RatedRunRefused(branch.rationale)
    return branch


# ---------------------------------------------------------------------------
# the driver: the bridge's own loop, instrumented through the mover seam
# ---------------------------------------------------------------------------


def r5_config(games: int = R5_GAMES, seed: int = R5_RUN_SEED) -> ArenaConfig:
    """Return the pre-registered R5 match config.

    §D3.5: the same start mode as §D4 and the same ``book_id``, ``run_seed =
    0xC0FFEE``, ``draw_score = 0.0``. The two arguments exist so a test can play
    one pair instead of ten; both default to the pre-registered literals, and a
    caller that takes the defaults cannot accidentally measure a different run.
    """
    return ArenaConfig(
        games=games,
        seed=seed,
        draw_score=R5_DRAW_SCORE,
        book_id=bridged_book_id(resolved_gungi_mode()),
    )


class _GameTally:
    """The mutable per-game record the two probes of one game share."""

    def __init__(self, pair: int, seat: int) -> None:
        self.pair = pair
        self.seat = seat
        self.evals: dict[str, list[int]] = {KOMUGI_SIDE: [], V1_SIDE: []}
        self.seeds: dict[int, int] = {}
        self.d1_plies = 0
        self.d1_plies_v1_to_move = 0
        self.d1_ids = 0
        self.no_move_plies = 0
        self.void_mode: str | None = None

    def freeze(self, *, scored: bool) -> R5Game:
        """Return the immutable :class:`R5Game` for this game."""
        return R5Game(
            pair=self.pair,
            seat=self.seat,
            evals_k=tuple(self.evals[KOMUGI_SIDE]),
            evals_v1=tuple(self.evals[V1_SIDE]),
            d1_plies=self.d1_plies,
            d1_plies_v1_to_move=self.d1_plies_v1_to_move,
            d1_ids=self.d1_ids,
            no_move_plies=self.no_move_plies,
            seeds=tuple(self.seeds[ply] for ply in sorted(self.seeds)),
            void_mode=self.void_mode,
            scored=scored,
        )


class _Collector:
    """Every game of one R5 run, keyed by ``(pair, seat)``.

    Keyed rather than appended because both probes call ``reset`` for the same
    game and a crash respawn (§D6 M1) builds a *new* probe mid-match: the two
    calls and the two probe generations must land on one record, or Komugi's
    plies and v1's would be filed as different games.
    """

    def __init__(self) -> None:
        self.order: list[tuple[int, int]] = []
        self.tallies: dict[tuple[int, int], _GameTally] = {}

    def begin(self, pair: int, seat: int) -> _GameTally:
        """Return the tally for game ``(pair, seat)``, creating it once."""
        key = (pair, seat)
        tally = self.tallies.get(key)
        if tally is None:
            tally = _GameTally(pair, seat)
            self.tallies[key] = tally
            self.order.append(key)
        return tally


class _R5Probe:
    """A :class:`~komugi.eval.v1bridge.Mover` that records and delegates.

    Structural, not nominal, exactly as :class:`~komugi.eval.gumbel_mover.GumbelMover`
    is: nothing here subclasses ``Mover``, and the annotation on
    :func:`_instrumented`'s factory is where ``mypy --strict`` checks the four
    methods.

    Three things are recorded, and the reason each has to be recorded *here*
    rather than read off the finished match:

    * **the per-move ``evals_used``**, taken from the wrapped mover's own
      ``evals`` list after each ``pick``. Both implementations append exactly one
      entry per successful pick -- ``GumbelMover`` from its counting evaluator,
      ``V1RemoteMover`` from the wire's ``evals_used`` -- but ``V1RemoteMover``
      *replaces* the list at ``reset``, so it is re-read every time rather than
      cached;
    * **diagnostic D-1**, which needs the position, and the position exists only
      inside the loop;
    * **the ``no_move``/M5 plies**, which leave as a
      :class:`~komugi.eval.v1bridge.BridgeVoided` and are otherwise visible only
      as a pair-level void with no per-game attribution.

    Args:
        inner: The real mover.
        collector: The run's shared record.
        side: :data:`KOMUGI_SIDE` or :data:`V1_SIDE`.

    Raises:
        TypeError: If `inner` does not report a per-move ``evals`` list. A mover
            that cannot say what it spent cannot be in a run whose entire output
            is what the two sides spent.
    """

    def __init__(self, inner: Mover, collector: _Collector, side: str) -> None:
        if not isinstance(getattr(inner, "evals", None), list):
            raise TypeError(
                f"the {side} mover {type(inner).__name__} reports no per-move "
                f"'evals' list, so R5 cannot measure the one quantity it exists "
                f"to measure (§D3.5); both GumbelMover and V1RemoteMover carry one"
            )
        self._inner = inner
        self._collector = collector
        self._side = side
        self._tally: _GameTally | None = None

    @property
    def inner(self) -> Mover:
        """Return the wrapped mover -- construction and diagnostics only."""
        return self._inner

    @property
    def last_win_state(self) -> tuple[int, int, int] | None:
        """Forward the wrapped mover's last view, for §D6 M6's diagnostic.

        ``play_bridged_game`` reads this off the mover with ``getattr``
        (``v1bridge.py:1948-1951``). A wrapper that did not forward it would
        silently zero the ``referee_ahead`` count for the whole run -- a
        diagnostic that fails quiet, which is the failure shape §D6 is written
        against.
        """
        view = getattr(self._inner, "last_win_state", None)
        if isinstance(view, tuple):
            return cast("tuple[int, int, int]", view)
        return None

    def _evals(self) -> list[int]:
        """Return the wrapped mover's live ``evals`` list, re-read every call."""
        return cast("list[int]", getattr(self._inner, "evals", []))

    def _record_void(self, void_mode: str) -> None:
        """Note the §D6 mode this game ended on, counting M5 as a ``no_move`` ply."""
        tally = self._tally
        if tally is None:
            return
        tally.void_mode = void_mode
        if void_mode == M5:
            tally.no_move_plies += 1

    def reset(self, *, pair: int, seat: int) -> None:
        """Open (or join) this game's record, then start the wrapped mover.

        The record is opened **before** the delegation so that a game whose very
        ``reset`` voids -- a v1 process that died between pairs -- still appears
        in the run with its mode, rather than vanishing from a report that claims
        to list every game played.
        """
        self._tally = self._collector.begin(pair, seat)
        try:
            self._inner.reset(pair=pair, seat=seat)
        except BridgeVoided as voided:
            self._record_void(voided.void.mode)
            raise

    def pick(self, state: GameState, *, ply: int, seed: int) -> int:
        """Record the ply, delegate, and record what the ply cost."""
        tally = self._tally
        if tally is None:
            raise RuntimeError(
                "R5's probe was asked to pick before reset opened a game: the "
                "ply would be filed under no game and would reach no mean"
            )
        tally.seeds[ply] = seed
        orphans = d1_orphans(state)
        if orphans:
            tally.d1_plies += 1
            tally.d1_ids += len(orphans)
            if self._side == V1_SIDE:
                tally.d1_plies_v1_to_move += 1

        evals = self._evals()
        before = len(evals)
        try:
            action = self._inner.pick(state, ply=ply, seed=seed)
        except BridgeVoided as voided:
            self._record_void(voided.void.mode)
            raise
        except BudgetBreach:
            # Komugi's side of §D3.5: an in-process refusal, resolved as M8 at
            # the arena boundary. Booked here so the game's row names the mode.
            self._record_void(M8)
            raise

        if len(evals) != before + 1:
            raise RuntimeError(
                f"the {self._side} mover played ply {ply} without appending a "
                f"per-move evals entry ({before} -> {len(evals)}): R5's mean "
                f"would be taken over a different set of plies than the one that "
                f"was played"
            )
        tally.evals[self._side].append(int(evals[-1]))
        return action

    def observe(self, action: int, record: MoveRecord) -> None:
        """Forward §D1's notification, booking a void that leaves through it."""
        try:
            self._inner.observe(action, record)
        except BridgeVoided as voided:
            self._record_void(voided.void.mode)
            raise

    def close(self) -> None:
        """Release the wrapped mover."""
        self._inner.close()


def _instrumented(
    entrant: BridgeEntrant, collector: _Collector, side: str
) -> BridgeEntrant:
    """Return `entrant` with its ``mover_factory`` wrapped in an :class:`_R5Probe`.

    ``dataclasses.replace`` rather than a fresh construction, so the identity,
    the family and the three budget fields the configuration block publishes
    (§D3.6 P5) travel through unaltered: an R5 run must be played by the same
    entrant the rated run will be played by.
    """

    def build() -> Mover:
        """Build the real mover, then the probe around it."""
        return _R5Probe(entrant.mover_factory(), collector, side)

    return replace(entrant, mover_factory=build)


def run_r5(
    komugi: BridgeEntrant,
    v1: BridgeEntrant,
    config: ArenaConfig | None = None,
    *,
    rated: bool = True,
    ceiling: int = EVAL_CEILING,
) -> R5Result:
    """Play the R5 pre-flight and return everything §D3.5 asks it to report.

    Ten pairs, twenty games, Komugi vs v1, at the **rated** per-move ceiling of
    257 on both sides, the same start mode and the same
    :func:`~komugi.eval.arena.move_seed_for` derivation the rated run uses, with
    ``run_seed = 0xC0FFEE``. The games are played by
    :func:`~komugi.eval.v1bridge.play_bridged_match`, so pair atomicity, §D6's
    void accounting, M1's respawn and the seeding are the rated run's own and not
    a second implementation of them.

    **v1's root Dirichlet noise is off** (B2). That is a property of the adapter
    the caller spawned, not something this driver can set on the wire; it is
    recorded in the artefact's configuration block as the pre-registration it is,
    and the v1-side runner is where it is enforced.

    Args:
        komugi: The Komugi entrant (:func:`komugi.eval.gumbel_mover.komugi_entrant`).
            Holds white in seat 0 of every pair.
        v1: The v1 entrant (:func:`komugi.eval.v1bridge.v1_entrant`).
        config: The match config; :func:`r5_config` by default. A test may shrink
            the pair count here, and only here.
        rated: Passed to :func:`~komugi.eval.v1bridge.play_bridged_match`, which
            asserts §D3.3's pinned budget (A15) and §D4 amendment 2's ``book_id``.
            R5 measures a *rated* quantity, so this defaults to true.
        ceiling: The per-move ceiling both sides were built with, for the record
            and for the post-run check below.

    Returns:
        The :class:`R5Result`. ``Δmean`` is ``None`` when no pair was scored --
        which is A24's blocking state and not a proceed.

    Raises:
        ValueError: If a scored ply reports more evaluations than `ceiling`. P1
            and M10 catch this per move on the v1 side and the counting evaluator
            makes it unreachable on Komugi's, so reaching it means a mean was
            about to be published over plies that were not played under the
            ceiling the artefact names.
    """
    match_config = r5_config() if config is None else config
    collector = _Collector()
    match = play_bridged_match(
        _instrumented(komugi, collector, KOMUGI_SIDE),
        _instrumented(v1, collector, V1_SIDE),
        match_config,
        rated=rated,
    )
    return _summarise_r5(
        collector,
        match,
        ceiling=ceiling,
        run_seed=match_config.seed,
        book_id=match_config.book_id,
        komugi=komugi,
        v1=v1,
    )


def _scored_pairs(collector: _Collector, match: BridgedMatch) -> frozenset[int]:
    """Return the pair indices whose games may reach a mean.

    §D3.6: "every mean in P4 and P5 is computed over the plies of **scored**
    pairs only". A pair is scored iff the match completed and no §D6 void names
    it -- an aborted or match-voided run keeps nothing, whatever its rate, so its
    scored set is empty.

    Raises:
        ValueError: If the set derived here disagrees with the match's own
            ``pairs_scored``. The two are computed from different things (the
            void records versus the driver's own counter), and a disagreement
            means R5's means and the harness's records are over different games.
    """
    if match.status != COMPLETED:
        return frozenset()
    voided = {void.pair for void in match.tally.voids if void.pair is not None}
    scored = frozenset(
        pair for pair, _seat in collector.order if pair not in voided
    )
    if len(scored) != match.tally.pairs_scored:
        raise ValueError(
            f"R5 derived {len(scored)} scored pairs from the void records but "
            f"the match scored {match.tally.pairs_scored}: the means would be "
            f"taken over a different set of games than the one that was rated"
        )
    return scored


def _summarise_r5(
    collector: _Collector,
    match: BridgedMatch,
    *,
    ceiling: int,
    run_seed: int,
    book_id: str,
    komugi: BridgeEntrant,
    v1: BridgeEntrant,
) -> R5Result:
    """Turn one played match plus its probe records into the :class:`R5Result`."""
    scored = _scored_pairs(collector, match)
    games = tuple(
        collector.tallies[key].freeze(scored=key[0] in scored)
        for key in collector.order
    )

    for game in games:
        if not game.scored:
            continue
        for side, spent in ((KOMUGI_SIDE, game.evals_k), (V1_SIDE, game.evals_v1)):
            over = [value for value in spent if value > ceiling]
            if over:
                raise ValueError(
                    f"pair {game.pair} seat {game.seat}: the {side} side reports "
                    f"{over!r} evaluations on a scored ply, above the ceiling of "
                    f"{ceiling} (§D3.3). P1/M10 should have voided this pair; a "
                    f"mean published over it would not be a mean at the ceiling "
                    f"the artefact names"
                )

    pairs: list[R5Pair] = []
    for pair in sorted(scored):
        spent_k = [
            value for game in games if game.pair == pair for value in game.evals_k
        ]
        spent_v1 = [
            value for game in games if game.pair == pair for value in game.evals_v1
        ]
        mean_k, mean_v1 = _mean(spent_k), _mean(spent_v1)
        pairs.append(
            R5Pair(
                pair=pair,
                mean_k=mean_k,
                mean_v1=mean_v1,
                delta_mean=(
                    None if mean_k is None or mean_v1 is None else mean_k - mean_v1
                ),
                plies_k=len(spent_k),
                plies_v1=len(spent_v1),
            )
        )

    return R5Result(
        games=games,
        pairs=tuple(pairs),
        status=match.status,
        reason=match.reason,
        pairs_attempted=match.tally.pairs_attempted,
        pairs_scored=match.tally.pairs_scored,
        ceiling=ceiling,
        run_seed=run_seed,
        config={
            "ceiling": ceiling,
            "m": komugi.m,
            "n": komugi.n,
            "gumbel_scale": komugi.gumbel_scale,
            "tree_reuse": "off",
            "root_noise": "off",
            "forced_move_convention": (
                "one root evaluation through the counted evaluator"
            ),
            "referee": "komugi",
            "gungi_mode": resolved_gungi_mode(),
            "book_id": book_id,
            "seed_derivation": "move_seed_for(config, pair, seat, ply)",
            "komugi_player": {
                "id": komugi.player.id,
                "family": komugi.player.family,
            },
            "v1_player": {"id": v1.player.id, "family": v1.player.family},
        },
    )
