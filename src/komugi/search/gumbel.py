"""Gumbel AlphaZero: the search, its configuration, and its result.

Reference: Danihelka, Guez, Schrittwieser and Silver, "Policy improvement by
planning with Gumbel", ICLR 2022. The implementation below is the AlphaZero
variant (no learned dynamics): the tree is expanded by actually playing moves
on a cloned :class:`~komugi.search.state.SearchState`.

The shape of one search
-----------------------
1. Evaluate the root once. Draw one Gumbel variate per action from a
   :func:`numpy.random.default_rng` seeded by :attr:`GumbelConfig.seed`. This
   is the *only* use of randomness in the whole search.
2. Take the ``m`` legal actions with the largest ``g(a) + logits(a)``. By the
   Gumbel top-k trick that is exactly a sample of ``m`` actions drawn without
   replacement from ``softmax(logits)`` -- not a heuristic shortlist.
3. Spend the ``n``-simulation budget on those candidates by **sequential
   halving**, comparing them on ``g(a) + logits(a) + sigma(q(a))`` and
   dropping the worse half each round.
4. Play the argmax of that same quantity among the survivors. Return a
   training target built from *completed* Q-values over the full legal action
   set.

What is deliberately missing, and why
-------------------------------------
* **No root Dirichlet noise.** Exploration at the root is the Gumbel draw.
  Adding noise to the logits would corrupt the sampling distribution that the
  policy-improvement guarantee is stated over, and it would do it invisibly.
* **No root temperature FIELD.** The action returned is an argmax over a
  sample, so the stochasticity is already in the sample. A second temperature
  on top would double-count it.

  This module used to add that :attr:`GumbelConfig.gumbel_scale` "is not that
  knob in disguise: a temperature divides the *logits* … whereas
  ``gumbel_scale`` multiplies the *noise*". That is true about which array the
  code multiplies and **false about the induced distribution**, and KOM-036
  §2.2 corrects it here rather than leaving the denial standing. Root candidate
  selection takes the top ``m`` of ``s·γ_a + L_a`` (below), and for ``s > 0``
  dividing every entry by ``s`` preserves the ordering::

      argsort_desc( s·γ_a + L_a )  ==  argsort_desc( γ_a + L_a/s )

  By the Gumbel top-k trick the right-hand side is exactly a sample of ``m``
  actions **without replacement from** ``softmax(L/s)`` -- the definition of a
  temperature ``T = s``. Every other use of the same draw is the
  halving/final score ``γ·s + L + σ(q)``, which divides the same way to
  ``γ + (L + σ)/s``: one temperature, applied at one site, to the *combined*
  score. So ``s < 1`` is a **colder** root (at ``s = 0.5`` every pairwise odds
  ratio in the prior is squared) and ``s > 1`` a hotter one that also dilutes
  the search's own Q evidence against the noise, which is the real and stated
  cost of a hot draft schedule.

  What the field's separateness still buys is not the denial but the
  containment: it is applied at exactly one site (the draw below) and provably
  cannot reach :func:`_improved_policy`, which never reads the Gumbel vector.
  Changing it changes which action is played and which candidates were
  simulated -- the trajectory distribution and the sims allocation -- and never
  the formula that produces the recorded label.
* **No forced playouts.** Sequential halving *is* the budget allocation.
  Forcing a minimum visit count on each child is a fix for a visit-count
  target, and there is no visit-count target here.
* **No visit-count policy target.** At branching near 229 with a 32-visit
  budget, a normalised visit vector is mostly zeros, and a target that is
  mostly zeros teaches the network that almost every legal move is impossible.
  The target here is ``softmax(logits + sigma(completedQ))`` over *all* legal
  actions: candidates contribute their measured Q, everything else keeps its
  prior shape re-centred by the root's own value estimate. Every legal action
  therefore keeps nonzero mass (under ``proof_mode = "interval_v1"`` the support
  is the admissible set instead, and excluded legal actions are exact zeros --
  the ``completed_q_proof_support_v1`` target, KOM-221).

Their absence is a design decision under test, not a gap waiting to be filled.
:class:`GumbelConfig` has no field that could switch Dirichlet noise, forced
playouts or a visit-count target on -- and exactly one, ``gumbel_scale``, that
**is** a root temperature by the derivation above, applied at one site, whose
default ``1.0`` is the identity. It is the sanctioned form of the concept
rather than a second one: there is no ``root_temperature`` field, because there
does not need to be.

Exact budget accounting
-----------------------
Sequential halving over ``m`` candidates runs the rounds ``m``, ``ceil(m/2)``,
..., ``1``. Round ``r``, with ``R - r`` rounds still to come and ``B``
simulations unspent, is given ``B // (R - r)`` of them -- and the final round
is given all of ``B``, so no simulation is ever left over. Within a round the
allocation is ``divmod`` across the survivors, with the remainder handed out
round-robin in survivor order (which after the first cut is best-first). The
consequence, and it is asserted by the tests, is that
``sims_used == n`` exactly whenever the root has two or more legal actions --
not ``n`` rounded down to a multiple of anything.

Wave scheduling (KOM-040 b2, default off)
-----------------------------------------
:class:`GumbelSearch` optionally takes a
:data:`~komugi.search.state.WaveEvaluator` alongside the single-position one.
Given it, sequential halving stops asking the network one leaf at a time:
within a round, at each visit index, every surviving root action descends to
its own pending leaf on the CPU and all of those leaves are submitted as *one*
call. Without it -- the default -- not a line of the schedule changes.

This is not an approximation traded for throughput; it is the same search.
Within a halving round the survivors' state is **disjoint**: the root action of
each simulation is dictated by the schedule rather than chosen by comparing
children, :func:`_select_interior` reads only the node it is standing on, and
the only cross-action comparison -- :meth:`GumbelSearch._root_scores` at the
round cut -- happens after every visit of the round in either schedule. Two
survivors' visits therefore touch disjoint subtrees and disjoint entries of the
root's per-action accumulators. Reordering them changes nothing, and because
each action's own visits stay in order (visit ``k`` of action ``X`` still sees
``X``'s backups ``1..k-1``) even the float accumulation order inside each
entry is unchanged, so the results are bit-identical rather than merely
statistically equivalent. That argument was verified against this code by the
design review -- see
``docs/reviews/2026-08-24-kom-040-worker-lane-design-review.md`` ("Wave
independence: VERIFIED") -- and it is pinned in CI by the stub-equivalence
golden in ``tests/search/unit/test_gumbel_waves.py``.

A descent that lands on a terminal leaf takes its value from the game and is
backed up on the spot, so it never enters the wave; a wave in which every
descent did that is empty and is not submitted. That is ordinary endgame
search flow, not an error -- the transport layer's refusal to send an empty
batch lives at the transport, and this path never reaches it (design review
D8').

The searched WDL shadow (KOM-173, default off)
----------------------------------------------
An evaluator may answer ``(policy_logits, value, wdl)`` instead of
``(policy_logits, value)``, where ``wdl`` is that leaf's ``(w, d, l)``
probability vector **in the leaf mover's frame**
(:data:`~komugi.search.state.EvalAnswer`). Given one, the search carries each
simulation's leaf vector up its path beside the scalar and accumulates it into
an ``(A, 3)`` shadow table on the **root node only**, which is where all three
readouts -- :attr:`SearchResult.root_wdl`, :attr:`SearchResult.root_wdl_pi`,
:attr:`SearchResult.played_wdl` -- live. An interior node never holds one, so
the cost is one 0.9 MB array per search rather than per node (design D2).

Three properties this rests on, each pinned by a test:

* **Frames.** One edge up to a parent with a different mover applies the
  *permutation* ``swap(w, d, l) = (l, d, w)`` -- the draw mass does not move,
  because ``d`` is swap-invariant -- at exactly the edges :func:`_backup`
  negates the scalar at. A same-mover edge (a draft double move) swaps nothing,
  just as it negates nothing. Terminal leaves enter as one-hots from
  :meth:`~komugi.search.state.SearchState.terminal_wdl`, in the same frame.
* **Selection never reads the vector.** ``_completed_q``, ``_sigma``,
  ``_improved_policy``, ``_select_interior`` and ``_root_scores`` are untouched
  and see only the scalar, so ``action``, ``policy_target``,
  ``root_prior_logits``, ``root_value``, ``visit_counts`` and ``sims_used`` are
  bit-identical to the two-tuple search at **any** ``draw_score`` and any MLH
  ``lambda`` (KOM-173 review F8).
* **The identity is conditional.** ``root_wdl[0] - root_wdl[2] ==
  root_value`` holds only where the leaf scalar *is* ``w - l`` of the leaf
  vector: the utility ``w - l`` is **odd** under the swap, so negamax and the
  permutation agree on it. A draw score contributes ``s*d``, which is even and
  therefore oscillates (that is kom-021's defect, measured in
  ``tests/search/unit/test_draw_utility_parity.py``), and the KOM-159
  moves-left term is scalar-only by construction. So the identity is a claim
  about ``draw_score == 0`` and ``lambda == 0``; the *bit-identity* of the
  scalar path above is a claim about every configuration.

With a two-tuple evaluator none of this allocates and the three readouts are
``NaN`` vectors: the arena, the gauntlet, the ladder, the anchor server and
every legacy stub keep the search they had, **number for number** -- every
field of :class:`SearchResult` a two-tuple search fills is bit-identical to
what it was before KOM-173. Not instruction for instruction, and the
difference is worth stating precisely: per evaluation the two-tuple path now
also pays one ``len`` (:func:`_unpack_evaluation`), a few ``None`` tests
against the vector it does not have (:meth:`GumbelSearch._evaluated_node` and
:func:`_backup`) and one extra call frame, and ``komugi.selfplay.adapter``'s
wave seam rebuilds its answer list. No allocation, no dtype change and no
arithmetic in any of it.

The two shapes may not be mixed within one search, and the refusal is
**symmetric**: a vector leaf under a scalar root and a scalar leaf under a
vector root both raise in :func:`_backup` (KOM-210 F3), because either way a
readout would be computed over a row that is missing or over an accumulator
that does not exist.

Bounded tactical discovery (KOM-220, ``discovery_mode``, default off)
---------------------------------------------------------------------
Steps 2 and 3 above are where KOM-219 found the search blind: the candidate set
is a sample of ``m`` actions from the prior and is never widened, and interior
selection ranks replies on that same prior, so an immediate marshal capture the
network gave ``1e-5`` to is unreachable **at any budget** -- ``n`` buys depth
inside the candidate set and never widens it. Self-play leaves such wins on the
board in 41-46% of games (KOM-215). The precedent is Teytaud & Teytaud (2010)'s
*decisive and anti-decisive moves*: a class of moves the rules can enumerate
exactly must not be left to sampling.

``discovery_mode = "root_marshal_v1"`` therefore runs, **before** the first
simulation and on both schedules, a survey of fixed and asserted size:

* the :class:`~komugi.search.tactical.TacticalProvider`'s candidates at the
  root are verified by clone/play, best-scoring first, and the first one that is
  terminal and won by the root mover is attached as a real terminal node and
  backed up -- a **witness**. It is added to the candidate set if the draw
  missed it (at most one action, so the set grows by at most one), and the
  survey stops there: a proven win at the root makes every other question moot;
* otherwise every **initial candidate** is played once, its child attached and
  backed up, and the provider's candidates *at that child* are verified the same
  way, so a reply that wins for the opponent is attached as a terminal node
  below the candidate it refutes. One witness per candidate.

``discovery_mode = "root_win_v1"`` (KOM-234) is the same survey with the first
bullet made exhaustive: the root candidate list is EVERY offered legal action,
ordered and verified exactly as above, and the provider is not asked at the root
at all. The KOM-232 census is the reason -- under ``root_marshal_v1`` every
available marshal capture was taken (31/31) and every win still missed was a
checkmate, by a drop, a stack or an ordinary move, none of which a capture
provider enumerates. The second bullet, the reply survey, is unchanged: an
exhaustive reply scan is ``|C| x |replies|`` transitions per candidate, which
fast-mode self-play cannot afford, and captures are the loss mode observed
there. The root scan costs at most ``num_legal`` transitions and **no network
row** -- a verified candidate is either non-terminal (nothing attached) or
terminal (read off the game) -- so the row bounds below are untouched and only
the transition bound grows, to ``num_legal + 2K``.

The survey is rules work plus at most ``K = min(m, num_legal)`` evaluations: it
costs at most ``2K`` extra simulations, at most ``1 + K`` provider calls and at
most ``K`` network rows, and sequential halving afterwards still spends exactly
``n`` ordinary simulations. The survey **proves** nothing by itself -- a
provider is allowed false positives -- it only attaches terminal evidence; the
proof layer (``proof_mode``, KOM-221) is what turns that evidence into an
excluded action or a played win, which is why discovery requires it. Everything
goes through the scalar :meth:`GumbelSearch._make_node` seam, so a wave-scheduled
search and a sequential one produce the same tree and the same result.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Final, Literal, cast

import numpy as np
import numpy.typing as npt

from komugi.search.modes import (
    SearchAlgorithmIdentity,
    search_algorithm_identity,
    validate_search_modes,
)
from komugi.search.proofs import root_edge_bounds
from komugi.search.state import (
    AnyEvaluator,
    AnyWaveEvaluator,
    EvalAnswer,
    SearchState,
    WdlAnswer,
)
from komugi.search.tactical import TacticalProvider

WDL_SIZE = 3
"""Width of a ``(w, d, l)`` vector -- the one place this module spells it."""

WDL_SIMPLEX_TOL = 1e-6
"""How far a ``(w, d, l)`` vector's mass may sit from 1 before it is refused.

Wide enough that a float32 head's softmax, summed in float64, never trips it --
three float32 probabilities agree with their exact sum to a few 1e-8 -- and
narrow enough that an unnormalised head (logits handed over where probabilities
were meant, or two slots filled and the third forgotten) cannot pass.
"""


@dataclass(frozen=True)
class GumbelConfig:
    """Everything the search needs to know, and deliberately nothing more.

    There is no ``dirichlet_alpha``, no ``root_temperature`` and no
    ``forced_playouts`` field, and adding one would be a change of algorithm
    rather than a change of tuning -- see the module docstring.

    Args:
        num_actions: Size ``A`` of the fixed action space. Every mask and
            every logit vector the search sees must have shape ``(A,)``.
        m: Number of root actions to sample without replacement. Clipped down
            to the number of legal actions at search time, so a config tuned
            for the opening is still valid in a forced endgame.
        n: Simulation budget for one search. Spent exactly (see the module
            docstring); ``n = 0`` is legal and yields a prior-shaped target.
        seed: Seed for the root Gumbel draw. Identical ``(state, config)``
            gives an identical :class:`SearchResult`.
        c_visit: The visit-count scale in ``sigma``. The paper's default of
            50 keeps the Q term comparable to logit differences once a child
            has been visited a few dozen times. Must be ``>= 0`` (ES-04): a
            negative coefficient **reverses** the monotone Q transform
            :func:`_sigma`, which is not a tuning choice but a different
            algorithm -- one that ranks the child it scored worst highest and
            still returns a confident-looking target.
        c_scale: The value scale in ``sigma``. ``sigma(q) = (c_visit +
            max_visit_count) * c_scale * q``, where ``max_visit_count`` is
            over the children of the node ``sigma`` is being applied at. Must
            be ``>= 0``, for the reason `c_visit` gives -- ``c_scale = -1``
            flips the chosen action outright. Zero stays legal on both, and is
            the honest spelling of "ignore the Q term": it flattens ``sigma``
            rather than inverting it, so the ordering it induces is the prior's
            and not the reverse of the search's.
        gumbel_scale: Multiplier on the root Gumbel *noise*, and on nothing
            else -- the logits are never touched, so this damps exploration
            without distorting what the network said. Full search uses the
            default ``1.0``, which is the paper's search exactly; DESIGN
            section 3.1's fast self-play mode calls for a reduced scale (exploration
            damped to protect the value targets its games produce), and the
            value it gets is a pre-registered P0b A/B, not a constant to be
            guessed here. Must be strictly positive: ``0.0`` would not be
            "the same search without noise" but a different algorithm --
            root selection would collapse from a sample of ``m`` actions
            without replacement from ``softmax(logits)`` into a deterministic
            top-``m`` argmax over the logits, which silently voids the
            policy-improvement guarantee the whole search is built on. A
            caller who genuinely wants argmax play should say so upstream,
            where it is visible, rather than by passing a zero in here.
        proof_mode: Which proof layer runs after the ordinary search
            (:mod:`komugi.search.modes`). ``"off"``, the default, is the search
            exactly as it was, bit for bit. ``"interval_v1"`` infers exact
            intervals from the terminals the tree reached and restricts the
            final action and the target's support to the admissible root
            actions (KOM-221).
        discovery_mode: Which discovery layer runs before sequential halving.
            ``"off"``, the default, is the search exactly as it was, bit for
            bit: no provider is consulted and one is refused at construction.
            ``"root_marshal_v1"`` runs the bounded tactical survey of KOM-220
            and ``"root_win_v1"`` the same survey with an exhaustive root scan
            (KOM-234; both are described in the module docstring). Either one
            **requires** a
            :class:`~komugi.search.tactical.TacticalProvider`, which
            :class:`GumbelSearch` refuses to run without rather than record a
            mode that silently did nothing. Requires ``proof_mode != "off"``.
    """

    num_actions: int
    m: int
    n: int
    seed: int
    c_visit: float = 50.0
    c_scale: float = 1.0
    gumbel_scale: float = 1.0
    proof_mode: str = "off"
    discovery_mode: str = "off"

    def __post_init__(self) -> None:
        """Reject configurations that cannot describe a search."""
        if self.num_actions < 1:
            raise ValueError(f"num_actions must be >= 1, got {self.num_actions!r}")
        if self.m < 1:
            raise ValueError(f"m must be >= 1, got {self.m!r}")
        if self.n < 0:
            raise ValueError(f"n must be >= 0, got {self.n!r}")
        if not math.isfinite(self.c_visit) or not math.isfinite(self.c_scale):
            raise ValueError(
                f"c_visit and c_scale must be finite, got "
                f"({self.c_visit!r}, {self.c_scale!r})"
            )
        if self.c_visit < 0.0:
            raise ValueError(
                f"c_visit must be >= 0, got {self.c_visit!r}; a negative "
                f"coefficient reverses the monotone Q transform `_sigma`, so "
                f"the search would prefer the child it scored worst"
            )
        if self.c_scale < 0.0:
            raise ValueError(
                f"c_scale must be >= 0, got {self.c_scale!r}; a negative "
                f"coefficient reverses the monotone Q transform `_sigma`, so "
                f"the search would prefer the child it scored worst"
            )
        if not math.isfinite(self.gumbel_scale) or self.gumbel_scale <= 0.0:
            raise ValueError(
                f"gumbel_scale must be finite and > 0, got "
                f"{self.gumbel_scale!r}; scale 0 is not a noiseless search but "
                f"a silent collapse of root sampling into top-m argmax"
            )
        # The mode pair is one vocabulary, validated in one place: an unknown
        # string, or a discovery layer with no proof layer to consume it, is a
        # config that names no algorithm (KOM-221).
        validate_search_modes(self.proof_mode, self.discovery_mode)


RescueScope = Literal["none", "original_candidates", "full_root"]
"""Which set the proof layer's final action was chosen from (KOM-221).

``"none"`` -- the ordinary survivor was admissible and was kept; the decision is
the legacy one. ``"original_candidates"`` -- the survivor was refuted and an
*eliminated* initial candidate was played in its place. ``"full_root"`` -- no
initial candidate was admissible either, so the choice ranged over every
admissible root edge.
"""


@dataclass(frozen=True, slots=True)
class SearchDiagnostics:
    """What the optional layers did, as an immutable record beside the result.

    Reported rather than inferred: a caller that records which algorithm
    produced a training row (:attr:`identity`) or that wants to know how often
    the proof layer overruled the ordinary search (:attr:`legacy_action`,
    :attr:`rescue_scope`) must not have to re-derive it from the numbers.
    Frozen and slotted because it is provenance: a record something downstream
    may key on is not a scratch buffer.

    Args:
        identity: The algorithm the two modes name
            (:func:`komugi.search.modes.search_algorithm_identity`).
        ordinary_sims_used: Simulations spent by sequential halving, i.e.
            exactly ``config.n`` whenever the root had two or more legal
            actions. **Not** :attr:`SearchResult.sims_used` once discovery is
            active: that field is this plus :attr:`tactical_sims_used`, because
            every backup the survey did is a root visit like any other.
        tactical_sims_used: Backups the KOM-220 survey performed -- one per
            initial candidate whose child it attached, plus one per terminal
            witness. Bounded by ``2K`` with ``K = min(m, num_legal)``, and
            exactly ``1`` when a root witness ended the survey.
        root_nn_rows: Network rows spent on the root. Always ``1``: the root is
            evaluated exactly once on every path out of :meth:`GumbelSearch.search`.
        survey_nn_rows: Network rows the survey spent, i.e. one per initial
            candidate whose child was non-terminal. Bounded by ``K``; a
            terminal child and every verified candidate cost none, because
            neither is handed to the evaluator.
        ordinary_nn_rows: Network rows spent on leaves by sequential halving,
            i.e. every row of this search except the root's and the survey's.
        provider_id: The :attr:`~komugi.search.tactical.TacticalProvider.provider_id`
            of the provider this search consulted, and ``None`` when discovery
            was off -- provenance, so a recorded row names the generator that
            produced it.
        provider_calls: Calls to ``candidate_actions``. At most ``1 + K``: one
            at the root, one per initial candidate with a non-terminal child.
            ``validate_state`` is not a call in this sense -- it is wiring.
            Under ``root_win_v1`` the root call is not made, so the bound is
            ``K``.
        candidate_verification_transitions: ``clone``/``play`` pairs spent
            verifying a root candidate or a reply. Rules work only: none of them
            asks the network anything. Under ``root_win_v1`` the root scan
            contributes at most ``num_legal`` of them -- every offered legal
            action, verified once -- so with a provider that names at most one
            reply per candidate the whole survey spends at most
            ``num_legal + 2K`` transitions: the scan, one reply probe per
            candidate, and :attr:`survey_root_transitions`.
        survey_root_transitions: ``clone``/``play`` pairs spent playing an
            initial candidate at the root, i.e. ``K`` on a full survey and
            ``0`` when a root witness ended it.
        terminal_witnesses: Terminal nodes the survey attached and backed up --
            a root witness (at most one, and then the survey stops) or one per
            refuted initial candidate.
        discovery_scope: Which set the survey covered:
            ``"initial_root_candidates_depth1"`` under an active discovery mode,
            ``"off"`` otherwise. It is a **scope**, not a guarantee: an action
            outside it keeps whatever the ordinary search learned about it.
        root_scan: The rule the ROOT half of the survey chose candidates by --
            ``"exhaustive"`` under ``root_win_v1`` (every offered legal action,
            and no provider call at the root at all) and ``"provider"``
            otherwise, which includes ``discovery_mode = "off"``: a search that
            ran no survey scanned no root exhaustively, and the default says so
            rather than inventing a third word for "nothing happened". It is the
            rule, not the event -- on a bypassed path (``bypassed_forced_root``,
            ``bypassed_n0``) the mode still names it and
            :attr:`discovery_status` is what says the scan never ran.
        discovery_status: What the survey did --- ``"off"``,
            ``"bypassed_forced_root"`` (one legal action, nothing to survey),
            ``"bypassed_n0"`` (no simulation budget), ``"root_witness"`` (a
            winning root candidate was verified and the survey stopped there)
            or ``"candidate_survey_complete"`` (the whole initial-candidate
            scope was surveyed, which says nothing about what was found).
        legacy_action: The action the ordinary search would have played, when
            the proof layer played a different one, and ``None`` otherwise --
            so "the layer changed nothing" is a value rather than a comparison.
        rescue_scope: See :data:`RescueScope`.
        proof_excluded_actions: How many offered legal root edges the proof
            layer dropped from the target's support.
    """

    identity: SearchAlgorithmIdentity
    ordinary_sims_used: int
    tactical_sims_used: int = 0
    root_nn_rows: int = 1
    survey_nn_rows: int = 0
    ordinary_nn_rows: int = 0
    provider_id: str | None = None
    provider_calls: int = 0
    candidate_verification_transitions: int = 0
    survey_root_transitions: int = 0
    terminal_witnesses: int = 0
    discovery_scope: str = "off"
    discovery_status: str = "off"
    root_scan: str = "provider"
    legacy_action: int | None = None
    rescue_scope: RescueScope = "none"
    proof_excluded_actions: int = 0


@dataclass(frozen=True, eq=False)
class SearchResult:
    """One search's output: what to play, and what to train on.

    Args:
        action: The action to play. Always legal in the searched state.
        policy_target: Shape ``(A,)`` ``float32`` training target, summing to
            ``1``. Zero exactly on the illegal actions -- every legal action,
            candidate or not, keeps nonzero mass -- in ``proof_mode = "off"``.
            Under ``"interval_v1"`` the support is the proof layer's admissible
            set and every excluded legal action is exactly ``0.0`` (KOM-221).
        root_prior_logits: Shape ``(A,)`` ``float32`` -- the **raw** logits the
            evaluator returned for the *root* position, exactly as it returned
            them (no mask, no softmax, no rescaling). Reported because the
            search evaluated the root anyway: a caller that needs the prior the
            search actually used -- self-play's ``policy_kl`` is the one that
            does -- would otherwise have to evaluate the same position a second
            time, which is one wasted network round trip on every move of every
            game (KOM-040 design review D9').
        root_value: The root's value estimate in ``[-1, 1]``, from the
            perspective of the player to move at the root. This is the
            ``v_mix`` of the paper, i.e. the network's own root value mixed
            with the prior-weighted mean Q of the *visited* children:
            ``(v + N * sum_{a: N(a)>0} pi(a) Q(a) / sum_{a: N(a)>0} pi(a))
            / (1 + N)`` with ``N = sum_a N(a)`` and ``pi`` the masked softmax
            of the root logits. It falls back to the bare network value when
            no child has been visited.

            **What is in it under KOM-159** (design review F5): the evaluator
            is free to fold a moves-left conversion term into each leaf's
            scalar (:func:`komugi.selfplay.adapter.mlh_utility_term`), and
            ``v_mix`` averages the visited children's Q, so at ``lambda > 0``
            this number carries the term -- and so does the self-play
            ``short_term_value`` column, which is built from it
            (``worker.py``'s ``finalize_rows``). ``wdl_target`` does not: the
            recorded outcome never passes through a utility at all. At
            ``lambda == 0``, which is every arena, gauntlet, ladder and anchor
            search, this is the number it always was.
        visit_counts: Shape ``(A,)`` ``int64`` root child visit counts. Sums
            to :attr:`sims_used`. Reported for diagnostics -- it is *not* the
            training target, and it is nonzero only on sampled candidates.
        sims_used: Simulations actually run, and always exactly
            ``visit_counts.sum()``. Equal to ``config.n`` whenever the root had
            two or more legal actions -- plus
            :attr:`SearchDiagnostics.tactical_sims_used` when the KOM-220
            discovery layer ran, because the survey's backups are root visits
            and a total that did not count them would not sum to the counts
            beside it.
        root_wdl: Shape ``(3,)`` ``float32`` -- the **vector** ``v_mix``
            (KOM-173 D4): ``(x_root + N * sum_{a: N(a)>0} pi(a) Qbar_wdl(a) /
            sum_{a: N(a)>0} pi(a)) / (1 + N)`` component-wise, with ``x_root``
            the root's own network vector and ``Qbar_wdl(a) = S(a) / N(a)`` the
            mean backed-up vector of root edge ``a``. Exactly
            :attr:`root_value`'s formula, one component at a time, so
            ``root_wdl[0] - root_wdl[2] == root_value`` at ``draw_score == 0``
            and ``lambda == 0`` (the utility ``w - l`` is odd under the frame
            swap; see the module docstring for why the two other utility terms
            break the identity without touching the arithmetic).
        root_wdl_pi: Shape ``(3,)`` ``float32`` -- ``sum_a pi'(a) Qtilde(a)``,
            the improved policy's own value in WDL space, with ``Qtilde(a) =
            Qbar_wdl(a)`` on visited edges and :attr:`root_wdl` elsewhere. The
            vector form of :func:`_completed_q`'s completion rule, weighted by
            the float64 improved policy this search built
            :attr:`policy_target` from.
        played_wdl: Shape ``(3,)`` ``float32`` -- ``Qbar_wdl`` of
            :attr:`action`, i.e. the deepest subtree's own answer. Falls back to
            :attr:`root_wdl` if the played action somehow has no visit.

    All three readouts are probability vectors (nonnegative, summing to ``1``)
    in the **root mover's frame**, and all three are ``NaN`` when the evaluator
    answered with two-tuples: the vector is opt-in, and a search that was never
    given one reports that rather than inventing a uniform. None of them feeds
    selection -- see :attr:`root_value` and the module docstring.

    Compares by value with array-aware equality, so two results of the same
    search can be checked with ``==`` instead of field by field; the default
    dataclass ``__eq__`` would raise on the arrays instead of answering. The
    three WDL readouts compare with ``equal_nan``, because "this search carried
    no vector" is a state two identical searches must agree on rather than a
    reason to call them different.
    """

    action: int
    policy_target: npt.NDArray[np.float32]
    root_prior_logits: npt.NDArray[np.float32]
    root_value: float
    visit_counts: npt.NDArray[np.int64]
    sims_used: int
    root_wdl: npt.NDArray[np.float32]
    root_wdl_pi: npt.NDArray[np.float32]
    played_wdl: npt.NDArray[np.float32]
    diagnostics: SearchDiagnostics | None = None
    """What the optional layers did, or ``None`` when none of them ran.

    Last and defaulted so that every existing positional and keyword
    construction of a :class:`SearchResult` keeps working unchanged, and part
    of equality so that two results that differ only in *how* they were
    produced do not compare equal (KOM-221).
    """

    def __eq__(self, other: object) -> bool:
        """Return whether `other` is a :class:`SearchResult` with equal fields."""
        if not isinstance(other, SearchResult):
            return NotImplemented
        return (
            self.action == other.action
            and self.root_value == other.root_value
            and self.sims_used == other.sims_used
            and bool(np.array_equal(self.policy_target, other.policy_target))
            and bool(np.array_equal(self.root_prior_logits, other.root_prior_logits))
            and bool(np.array_equal(self.visit_counts, other.visit_counts))
            and _wdl_equal(self.root_wdl, other.root_wdl)
            and _wdl_equal(self.root_wdl_pi, other.root_wdl_pi)
            and _wdl_equal(self.played_wdl, other.played_wdl)
            and self.diagnostics == other.diagnostics
        )


def _wdl_equal(
    left: npt.NDArray[np.float32], right: npt.NDArray[np.float32]
) -> bool:
    """Return whether two WDL readouts agree, counting ``NaN`` as ``NaN``.

    ``np.array_equal`` without ``equal_nan`` calls the two all-``NaN`` readouts
    of two identical two-tuple searches *unequal*, which would make every
    ``result == result`` assertion in the suite fail for a reason that has
    nothing to do with the search.
    """
    return bool(np.array_equal(left, right, equal_nan=True))


def _nan_wdl() -> npt.NDArray[np.float32]:
    """Return the ``(3,)`` ``float32`` ``NaN`` vector a vectorless search reports."""
    return np.full(WDL_SIZE, np.nan, dtype=np.float32)


def _masked_softmax(
    scores: npt.NDArray[np.float64], legal: npt.NDArray[np.bool_]
) -> npt.NDArray[np.float64]:
    """Return a softmax of `scores` over the legal actions, zero elsewhere.

    Illegal entries are removed before the max is taken rather than being set
    to ``-inf`` and exponentiated, so a caller's logits on illegal actions can
    be any finite garbage without shifting the result.
    """
    out = np.zeros(scores.shape[0], dtype=np.float64)
    if not bool(legal.any()):
        return out
    shifted = np.where(legal, scores - float(scores[legal].max()), -np.inf)
    exp = np.where(legal, np.exp(shifted), 0.0)
    return np.asarray(exp / float(exp.sum()), dtype=np.float64)


def _as_policy_target(
    improved: npt.NDArray[np.float64], legal: npt.NDArray[np.bool_]
) -> npt.NDArray[np.float32]:
    """Return `improved` as the ``float32`` target without losing legal mass.

    A decided position drives the improved policy to genuine extremes: a
    candidate backing up ``q = +1`` while the unvisited tail is completed with
    a much lower ``v_mix``, under a peaked prior, opens a score gap well past
    100 nats -- and ``exp(-104)`` is already below the smallest ``float32``
    subnormal ``1.4e-45``. Casting the ``float64`` softmax straight down would
    flush those legal actions to exact zero, which is both the one thing
    :attr:`SearchResult.policy_target` promises to do only on illegal actions
    and the precise failure -- a target that is mostly zeros -- this search
    was built to avoid.

    So every legal entry is floored at the smallest positive ``float32``
    normal and the result renormalised over the legal set before the cast.
    Illegal entries stay exactly ``0.0``.
    """
    tiny = float(np.finfo(np.float32).tiny)
    floored = np.where(legal, np.maximum(improved, tiny), 0.0)
    total = float(floored.sum())
    if total > 0.0:
        floored = floored / total
    return np.asarray(floored, dtype=np.float32)


class _Node:
    """One state in the tree, holding per-action statistics for its children.

    Statistics live on the *parent*, keyed by action, which is what lets the
    root report visit counts and Q-values for actions whose child was never
    created. ``child_value_sums`` is accumulated in this node's own
    :attr:`to_play` perspective, so ``Q(a) = W(a) / N(a)`` needs no further
    sign work at read time; the negation happens once per edge during backup.

    Two KOM-173 attributes, both optional and both ``None`` on the path this
    module took before it existed:

    * ``wdl_net`` -- this node's own ``(w, d, l)`` as ``float64``, in this
      node's frame: the evaluator's third element on a non-terminal node, the
      game's one-hot on a terminal one, and ``None`` when no vector was
      supplied.
    * ``child_wdl_sums`` -- the ``(A, 3)`` shadow accumulator, allocated by
      :meth:`GumbelSearch.search` on the **root only** (design D2). An interior
      node's stays ``None`` for the whole search, which is the memory invariant
      T11 pins: every readout is a root quantity, so a per-node table would add
      0.9 MB per node for nothing.
    """

    def __init__(
        self,
        *,
        to_play: int,
        is_terminal: bool,
        value: float,
        legal: npt.NDArray[np.bool_],
        logits: npt.NDArray[np.float32],
        wdl: npt.NDArray[np.float64] | None = None,
    ) -> None:
        self.to_play = to_play
        self.is_terminal = is_terminal
        self.value = value
        self.legal = legal
        self.logits = logits
        self.prior = _masked_softmax(logits.astype(np.float64), legal)
        num_actions = int(legal.shape[0])
        self.child_visits: npt.NDArray[np.int64] = np.zeros(
            num_actions, dtype=np.int64
        )
        self.child_value_sums: npt.NDArray[np.float64] = np.zeros(
            num_actions, dtype=np.float64
        )
        self.children: dict[int, _Node] = {}
        self.wdl_net: npt.NDArray[np.float64] | None = wdl
        self.child_wdl_sums: npt.NDArray[np.float64] | None = None


def _completed_q(node: _Node) -> tuple[npt.NDArray[np.float64], float]:
    """Return `(completed_q, v_mix)` for `node`'s children.

    ``completed_q[a]`` is the empirical ``W(a) / N(a)`` where the child has
    been visited, and ``v_mix`` -- the node's own mixed value estimate --
    everywhere else. That completion is the reason a never-visited action
    still gets a defensible score: it is scored as "about as good as this node
    looks overall", not as zero, which under a ``[-1, 1]`` value convention
    would mean "a draw" and would be a real claim about the move.
    """
    visits = node.child_visits
    visited = visits > 0
    q = np.zeros(visits.shape[0], dtype=np.float64)
    np.divide(
        node.child_value_sums,
        visits.astype(np.float64),
        out=q,
        where=visited,
    )

    total_visits = int(visits.sum())
    prior_mass = float(node.prior[visited].sum()) if total_visits > 0 else 0.0
    if total_visits == 0 or prior_mass <= 0.0:
        v_mix = node.value
    else:
        weighted = float((node.prior[visited] * q[visited]).sum())
        v_mix = (node.value + total_visits * weighted / prior_mass) / (
            1.0 + total_visits
        )
    completed = np.asarray(np.where(visited, q, v_mix), dtype=np.float64)
    return completed, v_mix


def _unpack_evaluation(
    answer: EvalAnswer,
) -> tuple[npt.NDArray[np.float32], float, WdlAnswer | None]:
    """Return `answer` as ``(logits, value, wdl_or_None)``.

    The one place the two accepted answer shapes
    (:data:`~komugi.search.state.EvalAnswer`) are told apart, and they are told
    apart by ``len`` alone -- no ``isinstance``, no attribute probing, one
    integer comparison per evaluation. A legacy two-tuple evaluator therefore
    pays a length check and nothing else, and the searched-WDL machinery below
    never allocates for it.
    """
    logits = answer[0]
    value = answer[1]
    if len(answer) == 2:
        return logits, value, None
    # Indexed through a widened alias because mypy cannot narrow a union of
    # tuple types on `len` in every supported version; the length is already
    # established above.
    fields: tuple[object, ...] = answer
    return logits, value, cast("WdlAnswer", fields[2])


def _as_wdl(vector: WdlAnswer) -> npt.NDArray[np.float64]:
    """Return `vector` as an owned ``(3,)`` ``float64`` array, or refuse it.

    Owned rather than a view, because the accumulator adds into its own storage
    and a node that aliased the evaluator's buffer would drift the moment that
    buffer was reused. Checked here, at the seam, so a mis-shaped vector fails
    where it entered rather than as a broadcast error three frames later.

    **The one WDL checkpoint in this module** (KOM-179): every vector that
    enters the search -- the evaluator's third element, on the branching, wave
    and forced paths alike, and a terminal state's one-hot -- is admitted
    through here and nowhere else, so there is no second opinion to drift. A
    vector that is not a distribution is refused for the same reason a
    non-finite logit is: ``w - l`` off a negative or unnormalised triple is a
    number the readouts will report, the identity ``root_wdl[0] - root_wdl[2]
    == root_value`` will quietly stop holding, and the only symptom is training
    data nobody can trace back to here.
    """
    out = np.asarray(vector, dtype=np.float64).reshape(-1)
    if out.shape != (WDL_SIZE,):
        raise ValueError(
            f"a wdl (w, d, l) vector must have shape ({WDL_SIZE},), got "
            f"{np.asarray(vector).shape}"
        )
    if not bool(np.isfinite(out).all()):
        raise ValueError(f"a wdl (w, d, l) vector must be finite, got {out.tolist()}")
    if bool((out < 0.0).any()):
        raise ValueError(
            f"a wdl (w, d, l) vector must have no negative entry, got {out.tolist()}"
        )
    total = float(out.sum())
    if abs(total - 1.0) > WDL_SIMPLEX_TOL:
        raise ValueError(
            f"a wdl (w, d, l) vector must sum to 1 within {WDL_SIMPLEX_TOL}, got "
            f"{out.tolist()}, which sums to {total}"
        )
    return np.array(out, dtype=np.float64, copy=True)


def _terminal_wdl(state: SearchState) -> npt.NDArray[np.float64]:
    """Return `state`'s terminal one-hot, in the mover's frame at `state`.

    The vector counterpart of :meth:`~komugi.search.state.SearchState.terminal_value`
    and, like it, read straight off the game: contempt never enters
    (design D5), so a mated mover reads ``(0, 0, 1)`` and a draw ``(0, 1, 0)``
    whatever ``draw_score`` the scalar path is using.
    """
    return _as_wdl(state.terminal_wdl())


def _wdl_readouts(
    root: _Node, improved: npt.NDArray[np.float64], played: int
) -> tuple[
    npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]
]:
    """Return ``(root_wdl, root_wdl_pi, played_wdl)`` for a searched root.

    The vector readouts of KOM-173 D4, computed once at the end of a search and
    from root quantities only:

    * ``root_wdl`` is :func:`_completed_q`'s ``v_mix`` component-wise -- the
      root's own vector mixed with the prior-weighted mean of the *visited*
      edges' mean vectors -- which makes it a convex combination of probability
      vectors and therefore one itself;
    * ``root_wdl_pi`` weights the same completion rule by the improved policy
      ``pi'`` (the float64 array :attr:`SearchResult.policy_target` is built
      from, before the float32 floor-and-renormalise), so it is a convex
      combination too: ``pi'`` is zero off the legal set, where the completed
      rows are the fallback rather than data;
    * ``played_wdl`` is the played edge's mean vector, i.e. the deepest subtree
      alone, falling back to ``root_wdl`` on the visit count a survivor cannot
      have.

    All three come back ``float64``-computed and ``float32``-cast, in the root
    mover's frame. Returns ``NaN`` vectors when the search carried no vector at
    all -- the accumulator is ``None``, which is the two-tuple evaluator path.

    Everything below the accumulator itself is sized by the **visited** edges --
    at most ``m``, so at most 32 rows -- rather than by ``A``: a full ``(A, 3)``
    temporary would be another 0.9 MB per search at ``A = 37341``, for a table
    whose rows are zero wherever no simulation went. The completion of the
    unvisited legal tail is therefore done by its total ``pi'`` mass
    (``1 - sum_{visited} pi'``, since ``pi'`` is a distribution over the legal
    set and is zero off it) instead of by materialising a row per action.
    """
    sums = root.child_wdl_sums
    x_root = root.wdl_net
    if sums is None or x_root is None:
        return _nan_wdl(), _nan_wdl(), _nan_wdl()

    visits = root.child_visits
    visited = np.flatnonzero(visits > 0)
    total_visits = int(visits.sum())
    counts = visits[visited].astype(np.float64).reshape(-1, 1)
    means = sums[visited] / counts if visited.size else np.zeros((0, WDL_SIZE))

    prior = root.prior[visited]
    prior_mass = float(prior.sum()) if total_visits > 0 else 0.0
    if total_visits == 0 or prior_mass <= 0.0:
        root_wdl = np.array(x_root, dtype=np.float64, copy=True)
    else:
        weighted = (prior.reshape(-1, 1) * means).sum(axis=0)
        root_wdl = (x_root + total_visits * weighted / prior_mass) / (
            1.0 + total_visits
        )

    visited_mass = float(improved[visited].sum())
    root_wdl_pi = (improved[visited].reshape(-1, 1) * means).sum(axis=0) + (
        1.0 - visited_mass
    ) * root_wdl
    played_count = int(visits[played])
    played_wdl = sums[played] / played_count if played_count > 0 else root_wdl
    return (
        np.asarray(root_wdl, dtype=np.float32),
        np.asarray(root_wdl_pi, dtype=np.float32),
        np.asarray(played_wdl, dtype=np.float32),
    )


def _sigma(
    completed_q: npt.NDArray[np.float64], max_visits: int, config: GumbelConfig
) -> npt.NDArray[np.float64]:
    """Return the paper's monotone Q transform ``(c_visit + max N) * c_scale * q``."""
    scale = (config.c_visit + max_visits) * config.c_scale
    return np.asarray(completed_q * scale, dtype=np.float64)


def _improved_policy(
    node: _Node, config: GumbelConfig
) -> tuple[npt.NDArray[np.float64], float]:
    """Return `(pi', v_mix)`: the completed-Q improved policy at `node`.

    ``pi' = softmax(logits + sigma(completedQ))`` restricted to legal actions.
    This is both the interior selection rule's target distribution and, at the
    root, the training target itself.
    """
    completed_q, v_mix = _completed_q(node)
    max_visits = int(node.child_visits.max()) if node.child_visits.size else 0
    scores = node.logits.astype(np.float64) + _sigma(completed_q, max_visits, config)
    return _masked_softmax(scores, node.legal), v_mix


def _admissible_actions(
    offered: list[int], bounds: dict[int, tuple[float, float]]
) -> list[int]:
    """Return the root edges the proof intervals have not refuted (KOM-221).

    The rules, in ascending action order and in this order of precedence:

    * a **proven win** decides the position outright -- if any edge has
      ``L == 1.0`` exactly, the admissible set is every such edge and nothing
      else, because no other move can do better than winning;
    * otherwise the **lost** edges (``U == -1.0``) are dropped, unless *every*
      edge is lost, in which case they are all kept: a position with no way out
      still has to return a move, and refusing to choose is not one;
    * of what survives, an edge is dropped when its upper bound is strictly
      below the best lower bound, i.e. when it is proven no better than a move
      that is proven at least that good. Equal bounds are retained, so a
      tie between two unresolved moves is left to the ordinary search.

    The result is never empty, and it only shrinks as evidence accumulates.
    """
    proven = [action for action in offered if bounds[action][0] == 1.0]
    if proven:
        return proven
    alive = [action for action in offered if bounds[action][1] != -1.0]
    if not alive:
        alive = list(offered)
    best_lower = max(bounds[action][0] for action in alive)
    return [action for action in alive if bounds[action][1] >= best_lower]


def _support_target(
    scores: npt.NDArray[np.float64], support: npt.NDArray[np.bool_]
) -> npt.NDArray[np.float64]:
    """Return the softmax of `scores` over `support`, floored and renormalised.

    :func:`_as_policy_target`'s treatment of the ``float32`` underflow floor,
    applied over an arbitrary support mask instead of over the legal mask: the
    proof layer's target is exactly zero on the legal-but-excluded actions --
    that is the point of it -- so the floor must not reintroduce them. Returned
    in ``float64`` because the WDL readouts weight by this same distribution
    before it is cast (KOM-221).
    """
    distribution = _masked_softmax(scores, support)
    tiny = float(np.finfo(np.float32).tiny)
    floored = np.where(support, np.maximum(distribution, tiny), 0.0)
    total = float(floored.sum())
    if total > 0.0:
        floored = floored / total
    return np.asarray(floored, dtype=np.float64)


def _proof_decision(
    root: _Node,
    initial_candidates: list[int],
    survivor: int,
    gumbel: npt.NDArray[np.float64],
    config: GumbelConfig,
) -> tuple[int, npt.NDArray[np.float64], RescueScope, int]:
    """Return `(action, target, rescue_scope, excluded)` for a searched `root`.

    The KOM-221 final decision, run after an ordinary search and reading the
    tree without touching it. Sequential halving eliminates candidates
    irreversibly, so a survivor whose subtree later reaches a terminal
    refutation is still played and the recorded target then favours a move the
    search will not play (KOM-219 C3). Here instead:

    * :func:`komugi.search.proofs.root_edge_bounds` infers exact intervals from
      the reached terminals alone -- no evaluator value, no visit count -- and
      :func:`_admissible_actions` turns them into the admissible set ``E``;
    * the ranking is the ordinary root score ``g(a) + logits(a) +
      sigma(Q(a))`` with the same ``sigma`` and the same ``max N`` the halving
      used, except that an edge whose interval is *exact* has its completed Q
      replaced by that exact value: a proof outranks an estimate;
    * the survivor is kept if it is admissible. Otherwise an eliminated
      **initial candidate** is rescued -- the highest score among the
      admissible ones, ties to the earliest candidate -- and only if no initial
      candidate is admissible at all does the choice range over the whole
      admissible set, ties to the lowest action id. It is never a policy
      argmax and the noise is never redrawn;
    * the target is ``softmax(logits + sigma(Q_exact))`` over ``E`` only, with
      **no** Gumbel term, so that for a fixed tree it does not depend on the
      draw -- and exactly ``0.0`` on every action outside ``E``, including the
      legal ones the intervals excluded.

    Nothing on `root` is mutated: not the mask, not the logits, not the visit
    counts, not the value sums.
    """
    bounds = root_edge_bounds(root)
    offered = [int(action) for action in np.flatnonzero(root.legal)]
    admissible = _admissible_actions(offered, bounds)

    completed_q, _ = _completed_q(root)
    q_exact = completed_q.copy()
    for action, (lower, upper) in bounds.items():
        if lower == upper:
            q_exact[action] = lower
    max_visits = int(root.child_visits.max()) if root.child_visits.size else 0
    logits = root.logits.astype(np.float64)
    sigma = _sigma(q_exact, max_visits, config)
    scores = gumbel + logits + sigma

    admissible_set = set(admissible)
    rescue_scope: RescueScope
    if survivor in admissible_set:
        final = survivor
        rescue_scope = "none"
    else:
        pool = [action for action in initial_candidates if action in admissible_set]
        if pool:
            final = _highest_score(pool, scores)
            rescue_scope = "original_candidates"
        else:
            final = _highest_score(admissible, scores)
            rescue_scope = "full_root"

    support = np.zeros(root.legal.shape[0], dtype=np.bool_)
    support[np.asarray(admissible, dtype=np.int64)] = True
    target = _support_target(logits + sigma, support)
    return final, target, rescue_scope, len(offered) - len(admissible)


def _highest_score(
    actions: list[int], scores: npt.NDArray[np.float64]
) -> int:
    """Return the highest-scoring action, ties falling to the earliest given.

    A strict ``>`` scan rather than :func:`numpy.argmax` over a gathered array,
    because the tie-break is defined against the *caller's* order -- candidate
    order when an eliminated candidate is being rescued, ascending action id
    when the whole root is -- and not against the action space.
    """
    best = actions[0]
    for action in actions[1:]:
        if scores[action] > scores[best]:
            best = action
    return best


def _select_interior(node: _Node, config: GumbelConfig) -> int:
    """Return the deterministically selected action at a non-root `node`.

    ``argmax_a pi'(a) - N(a) / (1 + sum_b N(b))``: the action whose share of
    the improved policy most exceeds its share of the visits so far. No
    Gumbel noise, no exploration constant, no tie-breaking randomness -- the
    Gumbel draw belongs to the root and nowhere else, and ties fall to the
    lowest action index via :func:`numpy.argmax`.
    """
    improved, _ = _improved_policy(node, config)
    total_visits = float(node.child_visits.sum())
    score = improved - node.child_visits.astype(np.float64) / (1.0 + total_visits)
    return int(np.argmax(np.where(node.legal, score, -np.inf)))


def _backup(path: list[tuple[_Node, int]], leaf: _Node) -> None:
    """Accumulate `leaf`'s value along `path`, negamaxing at every mover change.

    `path` is the ``(parent, edge)`` sequence walked from the root down to
    `leaf`, root first. Shared verbatim by the sequential and the
    wave-scheduled paths: the one piece of arithmetic in this module where a
    duplicated copy could drift into a sign error that no shape or count
    assertion would catch.

    **The KOM-173 shadow rides the same walk.** `leaf`'s ``wdl_net`` is carried
    up beside the scalar and *swapped* -- ``(w, d, l) -> (l, d, w)`` -- at
    exactly the edges the scalar is negated at, so the two frame changes cannot
    drift apart: they are the same ``if``. Only the last step writes, and it
    writes into the ROOT's ``child_wdl_sums`` at the root edge this simulation
    entered by, because that is the only edge every readout is about. The
    scalar arithmetic above is untouched (and unreachable from the vector), so a
    search whose evaluator supplies no vector runs the numbers it always ran.

    **Mixed arity is refused in both directions** (KOM-210 F3). The accumulator
    exists iff the ROOT answered with a vector, so a missing leaf vector under a
    live accumulator would leave a hole in a row the readouts average over, and
    a leaf vector with no accumulator means every leaf paid :func:`_as_wdl` for
    readouts that can only come back ``NaN``. Both are a two-tuple seam paired
    with a three-tuple one, both are a construction error rather than a
    position, and both raise here -- at the first backup, which is the first
    moment the two seams have each been heard from.
    """
    value = leaf.value
    vector = leaf.wdl_net
    below = leaf
    for parent, parent_edge in reversed(path):
        # One negation per change of mover: the leaf value is in the leaf's
        # frame, and each step up re-expresses it in the parent's.
        if below.to_play != parent.to_play:
            value = -value
            if vector is not None:
                # A permutation, not a negation: `d` is swap-invariant. This is
                # a reversed *view*, so nothing writes through it -- the sum
                # below reads it into the root's own array (match/tree.py's
                # `swap` carries the same warning).
                vector = vector[::-1]
        parent.child_visits[parent_edge] += 1
        parent.child_value_sums[parent_edge] += value
        below = parent

    root, root_edge = path[0]
    sums = root.child_wdl_sums
    if sums is None:
        if vector is not None:
            raise ValueError(
                "this simulation's leaf carried a (w, d, l) vector but the "
                "searched-WDL accumulator does not exist (the root's evaluation "
                "was a two-tuple): an evaluator answered with a three-tuple "
                "where another answered with a two-tuple, which most often "
                "means a two-tuple scalar seam was paired with a three-tuple "
                "wave seam. One answer shape per search, on both seams"
            )
        return
    if vector is None:
        raise ValueError(
            "the searched-WDL accumulator is live (the root's evaluation "
            "carried a (w, d, l) vector) but this simulation's leaf has none: "
            "an evaluator answered with a two-tuple where another answered "
            "with a three-tuple, or a terminal leaf's state cannot supply "
            "terminal_wdl(). One answer shape per search, on both seams"
        )
    # `vector` is now in the ROOT's frame: the loop above applied one swap per
    # mover change on the way up, ending at the root.
    sums[root_edge] += vector


@dataclass(frozen=True)
class _PendingLeaf:
    """A descent that reached an unexpanded, non-terminal position.

    Everything the search needs to finish the simulation once the network has
    answered: where the leaf sits (`path`, whose last entry is the parent and
    the edge the new node hangs off), the position itself, and the two facts
    about it -- `legal` and `to_play` -- that were read at descent time, before
    the evaluator was given the state to look at.
    """

    action: int
    path: list[tuple[_Node, int]]
    state: SearchState
    legal: npt.NDArray[np.bool_]
    to_play: int


DISCOVERY_SCOPE: Final[str] = "initial_root_candidates_depth1"
"""What an active KOM-220 survey covers: the root, and each candidate's replies.

Recorded as :attr:`SearchDiagnostics.discovery_scope` so that a reader of a
search record knows what the survey did *not* look at. Depth 1 means the survey
never descends past a candidate's immediate reply, and *initial* means it covers
the actions the Gumbel draw sampled -- plus, at the root only, the provider's
own candidates under ``root_marshal_v1`` and the whole legal set under
``root_win_v1`` (:attr:`SearchDiagnostics.root_scan` is which of the two ran, so
the scope string stays one published word). A root action outside that set --
which under ``root_win_v1`` can only be a REPLY, the root having no outside -- is
exactly as visible to this
search as it was before KOM-220, which is the boundary
``test_gumbel_tactical_discovery.py`` pins rather than papers over.
"""


@dataclass(slots=True)
class _DiscoveryStats:
    """What one KOM-220 survey spent and found, counted as it went.

    Mutable and private, unlike the :class:`SearchDiagnostics` record it is
    copied into: it is the survey's running tally, and the survey is the only
    thing that writes it.
    """

    status: str
    tactical_sims_used: int = 0
    survey_nn_rows: int = 0
    provider_calls: int = 0
    candidate_verification_transitions: int = 0
    survey_root_transitions: int = 0
    terminal_witnesses: int = 0


def _draft_is_open(state: SearchState) -> bool:
    """Return whether **either** side still holds drafting rights at `state`.

    Either, not the mover's: ``Game.win_state`` is all-zero while ``draft_w or
    draft_b`` (``engine/game.py``), so a position where only the opponent is
    still drafting cannot be terminal either, and every probe of the KOM-234
    exhaustive root scan is a transition spent on a provably empty result. The
    KOM-233 match filter skips the draft for the same reason and by the same
    rule (``search/match/search.py:_draft_is_open``), which this mirrors.

    Duck-typed, and quiet about a state that cannot answer. The search layer may
    not import the engine or the self-play adapter, the draft is a *phase* and
    :class:`~komugi.search.state.SearchState` deliberately knows nothing about
    phases, and a toy test state has no draft at all. So: the engine ``Game``
    behind the state if it is reachable (``in_draft(side)``, per side --
    :attr:`komugi.selfplay.adapter.GameState.game` is that handle for the state
    self-play searches with, and ``.inner.game`` for a wrapper that carries one),
    the state's own mover-relative ``in_draft()`` if it is not, and ``False`` --
    scan, do not skip -- if neither exists. Guessing "drafting" for a state that
    cannot say would turn the scan off silently, which is the one failure mode
    worth refusing.
    """
    for holder in (state, getattr(state, "inner", None)):
        if holder is None:
            continue
        game: Any = getattr(holder, "game", None)
        per_side: Any = getattr(game, "in_draft", None)
        if per_side is not None:
            return bool(per_side(0)) or bool(per_side(1))
    mover_relative: Any = getattr(state, "in_draft", None)
    if mover_relative is not None:
        return bool(mover_relative())
    return False


def _halving_sizes(m: int) -> list[int]:
    """Return the survivor count of each sequential-halving round, ending at 1.

    ``m = 5`` gives ``[5, 3, 2, 1]``. The final ``1`` is a real round: it is
    where the budget left over by integer division gets spent, on the action
    the search is about to return.
    """
    sizes = [m]
    while sizes[-1] > 1:
        sizes.append((sizes[-1] + 1) // 2)
    return sizes


class GumbelSearch:
    """A reusable search: one evaluator, one config, many positions.

    Instances are stateless between :meth:`search` calls -- no transposition
    table, no tree reuse. Two calls with the same state and config return
    equal results, which is the property the self-play loop's reproducibility
    rests on.

    Args:
        evaluator: The single-position network wrapper. Used for the root
            evaluation and the forced-move path always, and for every leaf
            unless `wave_evaluator` is given. Either answer shape is accepted
            (:data:`~komugi.search.state.AnyEvaluator`): a
            :data:`~komugi.search.state.WdlEvaluator`'s third element arms the
            searched-WDL readouts (module docstring), and a plain
            :data:`~komugi.search.state.Evaluator` leaves them ``NaN`` and the
            rest of the search bit-identical. **This** seam is what decides,
            because the root is always evaluated through it.
        config: The search's budget, candidate count and seed.
        wave_evaluator: Optional batched seam (KOM-040 b2). ``None``, the
            default, is the search exactly as it was: leaves are evaluated one
            at a time through `evaluator` and nothing about the schedule
            changes. Given one, sequential halving batches each round's leaves
            into waves (see the module docstring) -- the same search, with
            fewer round trips. The root evaluation and the forced-move path
            stay on `evaluator` either way: there is nothing to batch at a
            root, and the forced-terminal-root carve-out is defined against
            the single-position contract. Its answer shape must MATCH
            `evaluator`'s, and a mismatch either way is refused at the first
            backup rather than silently averaged over (KOM-210 F3): the root
            decided whether the searched-WDL accumulator exists, so a wave that
            dropped the vector leaves a hole in it, and a wave that carried one
            the root did not has no accumulator to be read into.
        tactical_provider: The rules-level seam the KOM-220 discovery layer
            consults (:class:`~komugi.search.tactical.TacticalProvider`).
            Required by, and only usable with,
            ``config.discovery_mode != "off"``: the two are validated against
            each other at construction, in both directions, so neither a
            discovery mode with nothing to consult nor a provider nothing will
            consult can be built. Not an evaluator -- it is never asked for a
            value, only for actions the search is obliged to look at.
    """

    def __init__(
        self,
        evaluator: AnyEvaluator,
        config: GumbelConfig,
        *,
        wave_evaluator: AnyWaveEvaluator | None = None,
        tactical_provider: TacticalProvider | None = None,
    ) -> None:
        # Every refusal below happens BEFORE anything is stored, drawn or
        # evaluated: a mode and a provider that disagree is a wiring error, and
        # a wiring error must fail where it is written rather than at some
        # interior node mid-search (the same rule
        # `TacticalProvider.validate_state` exists for).
        if config.discovery_mode == "off":
            # Not merely useless: the provenance identity records the mode pair,
            # so a provider that is present and never consulted would leave a
            # run claiming a generator that had no effect on it.
            if tactical_provider is not None:
                raise ValueError(
                    "a tactical provider was given but discovery_mode is 'off', "
                    "so it would never be consulted; pass "
                    "discovery_mode='root_marshal_v1' or "
                    "discovery_mode='root_win_v1' to use it, or drop it"
                )
        elif tactical_provider is None:
            # The mirror image, and the one KOM-221 raised unconditionally while
            # this search still had no parameter to receive a provider through:
            # a discovery mode that silently did nothing would still be recorded
            # in the identity as though it had run.
            raise ValueError(
                f"discovery_mode {config.discovery_mode!r} needs a tactical "
                f"provider; pass tactical_provider=..., or "
                f"discovery_mode='off'"
            )
        if tactical_provider is not None:
            # Read once, here, because it is the *identity* the diagnostics of
            # every search this instance runs will carry. A blank or non-string
            # one names nothing, and it would only be noticed in a manifest.
            provider_id = getattr(tactical_provider, "provider_id", None)
            if not isinstance(provider_id, str) or not provider_id:
                raise ValueError(
                    f"a tactical provider's provider_id must be a non-empty "
                    f"str, got {provider_id!r}"
                )
        self._config = config
        self._evaluator = evaluator
        self._wave_evaluator = wave_evaluator
        self._provider = tactical_provider
        """The KOM-220 tactical seam, or ``None`` when discovery is off."""
        self._nn_rows = 0
        """Evaluator answers turned into nodes during the current search."""

    def search(self, state: SearchState) -> SearchResult:
        """Run one search from `state` and return the move plus the target.

        `state` is never mutated: everything below works on clones. Raises
        :class:`ValueError` if the root has no legal action at all, or is
        terminal while claiming to have several -- neither is a position a
        search can say anything about, and returning a target that sums to
        zero would push the contradiction into the training data.

        **Root-first evaluation is an invariant of this method** (KOM-159 R9),
        not an accident of the order the statements happen to be in: the ROOT
        is the first state handed to the evaluator on every path out of here --
        the forced-move branch evaluates the root and nothing else
        (:meth:`_forced_move`), and the searching branch builds the root node
        before a single simulation runs, under both the sequential and the wave
        schedule. A stateful evaluator may therefore treat its first call after
        a reset as "this is the root", which is exactly what
        :class:`komugi.selfplay.adapter.SearchContext` does with it.
        ``tests/search/unit/test_gumbel_unit.py`` pins the order with a
        recording stub, so a future scheduler that expanded a leaf first would
        fail there rather than silently re-centre self-play's conversion term
        on a child.
        """
        config = self._config
        num_actions = config.num_actions
        provider = self._provider if config.discovery_mode != "off" else None
        # BEFORE the legal mask, the evaluator and the RNG, and on every path
        # out of here including the forced and the zero-budget ones: an
        # incompatible provider/state pairing is a wiring error, and a wiring
        # error that only shows up on the positions that happen to reach the
        # survey is one that ships (KOM-220). Nothing has been read off `state`
        # at this point, which is the whole point of the ordering.
        if provider is not None:
            provider.validate_state(state)
        # Per search, not per instance: the counter answers "how many rows did
        # THIS search cost", and every path below feeds it through the one
        # evaluation checkpoint (:meth:`_evaluated_node`).
        self._nn_rows = 0
        mask = self._legal_mask(state)
        legal_actions = np.flatnonzero(mask)
        num_legal = int(legal_actions.shape[0])

        if num_legal == 0:
            raise ValueError(
                "search called on a state with no legal actions: nothing to "
                "return, and no policy target that sums to one exists"
            )
        if num_legal == 1:
            return self._forced_move(state, int(legal_actions[0]), mask)

        root_state = state.clone()
        if root_state.is_terminal():
            raise ValueError(
                f"search called on a terminal state that reports {num_legal} "
                f"legal actions"
            )
        # THE FIRST EVALUATOR CALL OF THE SEARCH, on both schedules and by
        # construction: `_sequential_halving` below cannot reach a leaf without
        # a root node to descend from, so nothing above this line asks the
        # network anything. See this method's docstring (KOM-159 R9).
        root = self._make_node(root_state)
        if root.wdl_net is not None:
            # The whole of KOM-173's memory cost: ONE (A, 3) float64 table, on
            # the root, allocated only because this evaluator answered with a
            # vector. Its existence is also the flag every simulation reads to
            # decide whether to carry one (design D2).
            root.child_wdl_sums = np.zeros((num_actions, WDL_SIZE), dtype=np.float64)

        # The one and only draw. Over the whole action space rather than the
        # legal subset so that the variate an action gets does not depend on
        # how many illegal actions happen to precede it.
        #
        # ``gumbel_scale`` is applied here, at the draw, and nowhere else:
        # this array is the same ``g`` that ranks the root candidates below
        # and that sequential halving keeps re-reading, so scaling it once
        # scales it consistently everywhere it is used. The default 1.0
        # leaves every downstream number bit-identical -- multiplying a
        # float64 by exactly 1.0 is the identity -- and any other scale gives
        # variates that are exactly a linear rescaling of that same draw.
        rng = np.random.default_rng(config.seed)
        gumbel = np.asarray(
            config.gumbel_scale * rng.gumbel(size=num_actions), dtype=np.float64
        )

        m = min(config.m, num_legal)
        draw = np.where(mask, gumbel + root.logits.astype(np.float64), -np.inf)
        order = np.argsort(-draw, kind="stable")
        survivors = [int(action) for action in order[:m]]

        # The KOM-220 survey, between the draw and the first simulation and on
        # both schedules. It may append at most one action to `survivors` (a
        # root witness the draw missed), so it runs before the rescue-pool copy
        # below; it spends no part of the `n` budget, and `_sequential_halving`
        # is handed exactly what it would have been handed without it.
        discovery = _DiscoveryStats(status="off")
        candidate_count = len(survivors)
        if provider is not None:
            if config.n == 0:
                discovery.status = "bypassed_n0"
            else:
                discovery = self._survey(
                    provider, root, root_state, mask, survivors, gumbel
                )

        # Taken BEFORE `_sequential_halving`, which cuts `survivors` down to the
        # final round's set in place. The proof layer's rescue pool is the set
        # the draw originally offered -- an eliminated candidate is exactly what
        # it may have to bring back (KOM-221) -- so the copy has to be made on
        # this side of the call.
        initial_candidates = list(survivors)
        sims_used = self._sequential_halving(root, root_state, survivors, gumbel)

        improved, v_mix = _improved_policy(root, config)
        survivor = survivors[int(np.argmax(self._root_scores(root, survivors, gumbel)))]
        if config.proof_mode == "off":
            # After the action is chosen and from the same `improved` the target
            # is built from, so the readouts describe exactly the tree that was
            # played from -- and reading them cannot move it (KOM-173 D3).
            root_wdl, root_wdl_pi, played_wdl = _wdl_readouts(root, improved, survivor)
            return SearchResult(
                action=survivor,
                policy_target=_as_policy_target(improved, root.legal),
                # The root node's own logits, i.e. the evaluator's answer for this
                # position -- handed back rather than recomputed, which is the
                # whole point of the field.
                root_prior_logits=root.logits,
                root_value=v_mix,
                visit_counts=root.child_visits.copy(),
                sims_used=sims_used,
                root_wdl=root_wdl,
                root_wdl_pi=root_wdl_pi,
                played_wdl=played_wdl,
            )

        # The active proof layer (KOM-221). It reads the finished tree and
        # nothing else: no extra simulation is run, no node is mutated, the
        # Gumbel draw is not redrawn, and `sims_used`, `visit_counts`,
        # `root_prior_logits` and `root_value` are therefore bit-identical to
        # what the branch above would have returned for the same search. What
        # can differ is only the decision and the target's support.
        final, proof_target, rescue_scope, excluded = _proof_decision(
            root, initial_candidates, survivor, gumbel, config
        )
        # `_support_target` already floored and renormalised over the admissible
        # support, so this is a cast and not a second normalisation: the exact
        # zeros on the excluded legal actions stay exactly zero.
        policy_target = np.asarray(proof_target, dtype=np.float32)
        # The policy-weighted readout follows the target that is actually
        # recorded, and the played readout the edge that is actually played, for
        # the same KOM-173 D3 reason as above -- a readout that described a
        # distribution nobody trains on would be a number about nothing.
        root_wdl, root_wdl_pi, played_wdl = _wdl_readouts(root, proof_target, final)
        # The KOM-220 bounds, asserted rather than trusted: each one is a claim
        # the survey's own structure makes (at most one child and at most one
        # witness per initial candidate; one provider call at the root and one
        # per candidate; no evaluation for a terminal state), and a survey that
        # broke one would be spending budget nobody accounted for.
        assert discovery.tactical_sims_used <= 2 * candidate_count
        assert discovery.provider_calls <= 1 + candidate_count
        assert discovery.survey_nn_rows <= candidate_count
        ordinary_nn_rows = self._nn_rows - 1 - discovery.survey_nn_rows
        assert 0 <= ordinary_nn_rows <= config.n
        total_sims = sims_used + discovery.tactical_sims_used
        assert total_sims == int(root.child_visits.sum())
        diagnostics = SearchDiagnostics(
            identity=search_algorithm_identity(
                config.proof_mode, config.discovery_mode
            ),
            ordinary_sims_used=sims_used,
            tactical_sims_used=discovery.tactical_sims_used,
            # The root is evaluated exactly once on every path out of here, so
            # every other row this search spent went to a survey child or to a
            # leaf.
            root_nn_rows=1,
            survey_nn_rows=discovery.survey_nn_rows,
            ordinary_nn_rows=ordinary_nn_rows,
            provider_id=None if provider is None else provider.provider_id,
            provider_calls=discovery.provider_calls,
            candidate_verification_transitions=(
                discovery.candidate_verification_transitions
            ),
            survey_root_transitions=discovery.survey_root_transitions,
            terminal_witnesses=discovery.terminal_witnesses,
            discovery_scope=DISCOVERY_SCOPE if provider is not None else "off",
            discovery_status=discovery.status,
            root_scan=(
                "exhaustive"
                if config.discovery_mode == "root_win_v1"
                else "provider"
            ),
            # A value rather than a comparison a caller has to re-derive: `None`
            # says the layer changed nothing, an action says which one it
            # overruled.
            legacy_action=survivor if final != survivor else None,
            rescue_scope=rescue_scope,
            proof_excluded_actions=excluded,
        )
        return SearchResult(
            action=final,
            policy_target=policy_target,
            root_prior_logits=root.logits,
            root_value=v_mix,
            visit_counts=root.child_visits.copy(),
            # Every backup is a root visit, the survey's included, so the total
            # is the one number that still equals `visit_counts.sum()`.
            sims_used=total_sims,
            root_wdl=root_wdl,
            root_wdl_pi=root_wdl_pi,
            played_wdl=played_wdl,
            diagnostics=diagnostics,
        )

    def _survey(
        self,
        provider: TacticalProvider,
        root: _Node,
        root_state: SearchState,
        mask: npt.NDArray[np.bool_],
        candidates: list[int],
        gumbel: npt.NDArray[np.float64],
    ) -> _DiscoveryStats:
        """Run the KOM-220 tactical survey, attaching what it verifies.

        Two passes, and the first one can end the whole thing:

        **The root survey.** The root candidate list -- the provider's
        candidates under ``root_marshal_v1``, and EVERY offered legal action
        under ``root_win_v1`` (KOM-234), where no provider call is made at all
        and nothing at all is scanned while either side still holds drafting
        rights (KOM-235: no terminal exists there to be found) --
        is ordered by the search's own root score ``g(a) + logits(a)`` --
        descending, ties to the lower action id -- and played on a clone one at
        a time, the same ordering rule in both modes. The first
        one that is terminal and won by the mover *at the root* is attached as a
        real terminal node and backed up: a **witness**. It is appended to
        `candidates` if the draw did not sample it (which is the C1 repair: the
        candidate set is widened by a tactical signal rather than by more
        budget), and the survey returns immediately, because no move can beat
        winning and there is nothing a reply survey could add to that.

        **The child survey.** Identical in both discovery modes -- the reply
        half is provider-based, because an exhaustive reply scan would cost
        ``|C| x |replies|`` transitions per candidate. Otherwise every action
        the draw sampled -- the candidate
        list as it stood on entry, in its original order -- is played
        once, its child attached (one evaluation, unless the child is terminal
        and costs none) and backed up, and the provider is asked for the
        candidates *at that child*. Each is played on a clone, and the first one
        that is terminal and won by the mover at the child -- the opponent
        replying -- is attached below it and backed up. That is the C2 repair:
        the refutation is in the tree before halving starts, so the proof layer
        can exclude the move it refutes instead of the search having to sample
        it.

        The survey **proves** nothing on its own: a provider may return false
        positives, so a candidate that is not terminal, or terminal and not won
        by the mover in question, attaches nothing at all and has cost one
        clone/play. What it attaches is real terminal evidence, and
        :func:`komugi.search.proofs.root_edge_bounds` is what reads it.

        `candidates` is mutated in place by at most one append. Everything else
        the survey touches is a clone; `root_state` is not played on.
        """
        want_wdl = root.child_wdl_sums is not None
        stats = _DiscoveryStats(status="candidate_survey_complete")
        initial_candidates = list(candidates)

        # The OFFERED root mask -- the same array the draw and the halving see,
        # already narrowed by whatever the state declined to offer (a DRAFTMIN
        # filter, say). The provider never widens it, so a candidate it returns
        # is an action this search could have played anyway.
        if self._config.discovery_mode == "root_win_v1":
            # KOM-234: the root half is exhaustive. Every offered legal action
            # is a candidate, so a win of any move type -- checkmate by a drop,
            # by a stack, by an ordinary move -- is verified, not just the
            # captures a provider enumerates. The provider is not asked at the
            # root at all (`provider_calls` stays 0 for this step): there is no
            # answer it could give that would widen `flatnonzero(mask)`, and a
            # call whose answer is discarded would still be a call a diagnostics
            # reader has to explain. `flatnonzero` is ascending, which is the
            # tie order the sort below relies on.
            #
            # Except in the DRAFT (KOM-235 F1): `Game.win_state` suppresses every
            # terminal while `draft_w or draft_b`, so no probe taken there can be
            # terminal and the whole scan is transitions spent on a result that
            # is empty by construction (measured: 45-50% of a game's scan cost).
            # An empty root list is exactly that result, reached without paying
            # for it -- no transition, no witness, `root_scan` still
            # "exhaustive", and the child survey below untouched.
            root_candidates = (
                ()
                if _draft_is_open(root_state)
                else tuple(int(action) for action in np.flatnonzero(mask))
            )
        else:
            root_candidates = self._verified_candidates(
                provider.candidate_actions(root_state, mask), mask
            )
            stats.provider_calls += 1
        logits = root.logits.astype(np.float64)
        # Best-first, so that when several candidates win the one the search
        # itself rates highest is the one attached. `sorted` is stable and the
        # ids arrive ascending, so ties fall to the lower id.
        for action in sorted(root_candidates, key=lambda a: -(gumbel[a] + logits[a])):
            probe = root_state.clone()
            probe.play(action)
            stats.candidate_verification_transitions += 1
            if not probe.is_terminal():
                continue
            if self._verified_winner(provider, probe) != root.to_play:
                continue
            # Terminal ingress: `_make_node` reads the value off the GAME and
            # asks the evaluator nothing, so a witness costs no network row. The
            # value is in the child's frame and `_backup` negates it at the
            # mover change, exactly as a reached terminal leaf would be.
            witness = self._make_node(probe, want_wdl=want_wdl)
            root.children[action] = witness
            _backup([(root, action)], witness)
            stats.tactical_sims_used += 1
            stats.terminal_witnesses += 1
            if action not in candidates:
                candidates.append(action)
            stats.status = "root_witness"
            return stats

        for action in initial_candidates:
            child_state = root_state.clone()
            child_state.play(action)
            stats.survey_root_transitions += 1
            child = root.children.get(action)
            if child is None:
                rows_before = self._nn_rows
                child = self._make_node(child_state, want_wdl=want_wdl)
                stats.survey_nn_rows += self._nn_rows - rows_before
                root.children[action] = child
            _backup([(root, action)], child)
            stats.tactical_sims_used += 1
            if child.is_terminal:
                # Nothing to reply to, and `child.legal` is empty: asking the
                # provider for the candidates of a finished game is not a
                # question with an answer.
                continue

            replies = self._verified_candidates(
                provider.candidate_actions(child_state, child.legal), child.legal
            )
            stats.provider_calls += 1
            for reply in replies:
                probe = child_state.clone()
                probe.play(reply)
                stats.candidate_verification_transitions += 1
                if not probe.is_terminal():
                    continue
                if self._verified_winner(provider, probe) != child.to_play:
                    continue
                grandchild = self._make_node(probe, want_wdl=want_wdl)
                child.children[reply] = grandchild
                _backup([(root, action), (child, reply)], grandchild)
                stats.tactical_sims_used += 1
                stats.terminal_witnesses += 1
                # One witness per candidate: a second refutation of a move
                # already proven lost buys nothing and costs a backup.
                break

        return stats

    def _verified_candidates(
        self, candidates: tuple[int, ...], mask: npt.NDArray[np.bool_]
    ) -> tuple[int, ...]:
        """Return `candidates`, or refuse them against
        :class:`~komugi.search.tactical.TacticalProvider`'s contract.

        Checked **before any transition is spent**, and checked here rather than
        trusted, because every one of these violations is silent downstream: a
        ``bool`` is an ``int`` that indexes ``mask`` as ``0`` or ``1``, an
        unsorted or duplicated list turns "at most one witness per candidate"
        into an unbounded loop, and an id outside the mask is an action this
        search may not play but would attach a child for. A provider is a seam
        the rules layer reaches the search through; it is not a trusted caller.
        """
        if not isinstance(candidates, tuple):
            raise ValueError(
                f"a tactical provider must return a tuple of action ids, got "
                f"{type(candidates).__name__}"
            )
        num_actions = self._config.num_actions
        previous = -1
        for item in candidates:
            # `bool` first: `isinstance(True, int)` is True, so a provider that
            # answered with flags would otherwise be read as action 1.
            if isinstance(item, bool) or not isinstance(item, (int, np.integer)):
                raise ValueError(
                    f"a tactical provider's action ids must be ints, got "
                    f"{item!r} of type {type(item).__name__}"
                )
            action = int(item)
            # Range first, so that the mask below is indexed with something it
            # has a slot for and a negative id is named as what it is.
            if action < 0 or action >= num_actions:
                raise ValueError(
                    f"a tactical provider returned action id {action}, outside "
                    f"the action space [0, {num_actions})"
                )
            if action <= previous:
                raise ValueError(
                    f"a tactical provider's action ids must be strictly "
                    f"ascending and unique, got {action} after {previous} in "
                    f"{candidates!r}"
                )
            if not bool(mask[action]):
                raise ValueError(
                    f"a tactical provider returned action id {action}, which "
                    f"the legal mask it was given does not offer; a provider "
                    f"never widens the mask"
                )
            previous = action
        return tuple(int(item) for item in candidates)

    def _verified_winner(
        self, provider: TacticalProvider, terminal_state: SearchState
    ) -> int | None:
        """Return the provider's winner for `terminal_state`, or refuse it.

        ``0``, ``1`` or ``None`` (a real draw) and nothing else. ``True`` is
        refused for the reason it is refused above: it would compare equal to
        player ``1`` and silently attach a witness for the wrong side, which is
        a backed-up ``+1`` on a move that does not win.
        """
        winner = provider.terminal_winner(terminal_state)
        if winner is None:
            return None
        if isinstance(winner, bool) or not isinstance(winner, (int, np.integer)):
            raise ValueError(
                f"a tactical provider's terminal_winner must return 0, 1 or "
                f"None, got {winner!r} of type {type(winner).__name__}"
            )
        player = int(winner)
        if player not in (0, 1):
            raise ValueError(
                f"a tactical provider's terminal_winner must return 0, 1 or "
                f"None, got {player}"
            )
        return player

    def _sequential_halving(
        self,
        root: _Node,
        root_state: SearchState,
        survivors: list[int],
        gumbel: npt.NDArray[np.float64],
    ) -> int:
        """Spend the whole budget on `survivors`, cutting them down to one.

        Mutates `survivors` in place to the final round's set and returns the
        number of simulations run, which is exactly ``config.n``.

        The two schedules differ only in *when* the network is asked, never in
        what is asked or in what the answers do to the tree: the round's
        allocation arithmetic below is shared, and the wave branch reorders
        provably independent simulations (module docstring). With no wave
        evaluator the loop is the original one, unrestructured.
        """
        wave_evaluator = self._wave_evaluator
        sizes = _halving_sizes(len(survivors))
        budget_left = self._config.n
        sims_used = 0

        for round_index, _size in enumerate(sizes):
            rounds_left = len(sizes) - round_index
            # The last round takes the remainder outright; that is what makes
            # the accounting exact instead of merely close.
            allocation = (
                budget_left if rounds_left == 1 else budget_left // rounds_left
            )
            per_action, remainder = divmod(allocation, len(survivors))
            if wave_evaluator is None:
                for position, action in enumerate(survivors):
                    visits = per_action + (1 if position < remainder else 0)
                    for _ in range(visits):
                        self._simulate(root, root_state, action)
                        sims_used += 1
            else:
                sims_used += self._run_waves(
                    wave_evaluator,
                    root,
                    root_state,
                    survivors,
                    per_action=per_action,
                    remainder=remainder,
                )
            budget_left -= allocation

            next_size = sizes[round_index + 1] if rounds_left > 1 else len(survivors)
            if next_size < len(survivors):
                scores = self._root_scores(root, survivors, gumbel)
                keep = np.argsort(-scores, kind="stable")[:next_size]
                survivors[:] = [survivors[int(index)] for index in keep]

        return sims_used

    def _root_scores(
        self,
        root: _Node,
        actions: list[int],
        gumbel: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Return ``g(a) + logits(a) + sigma(completedQ(a))`` for `actions`.

        The single quantity sequential halving cuts on and the returned action
        is the argmax of. Using completed Q rather than raw Q means a
        candidate that ran out of budget before being visited is compared on
        the root's own value estimate instead of on a zero.
        """
        completed_q, _ = _completed_q(root)
        max_visits = int(root.child_visits.max()) if root.child_visits.size else 0
        sigma = _sigma(completed_q, max_visits, self._config)
        index = np.asarray(actions, dtype=np.int64)
        return np.asarray(
            gumbel[index] + root.logits.astype(np.float64)[index] + sigma[index],
            dtype=np.float64,
        )

    def _simulate(self, root: _Node, root_state: SearchState, action: int) -> None:
        """Run one simulation that starts by playing `action` at the root.

        The root action is dictated by sequential halving; every action below
        it comes from :func:`_select_interior`. The descent stops at the first
        edge with no child yet (which gets expanded) or at a terminal node
        (which is simply re-scored, costing no evaluator call), and the leaf
        value is then negamaxed back along the path.
        """
        state = root_state.clone()
        node = root
        edge = action
        path: list[tuple[_Node, int]] = []

        while True:
            path.append((node, edge))
            state.play(edge)
            child = node.children.get(edge)
            if child is None:
                leaf = self._make_node(
                    state, want_wdl=root.child_wdl_sums is not None
                )
                node.children[edge] = leaf
                break
            if child.is_terminal:
                leaf = child
                break
            node = child
            edge = _select_interior(child, self._config)

        _backup(path, leaf)

    def _run_waves(
        self,
        wave_evaluator: AnyWaveEvaluator,
        root: _Node,
        root_state: SearchState,
        survivors: list[int],
        *,
        per_action: int,
        remainder: int,
    ) -> int:
        """Run one halving round's visits as waves; return the simulation count.

        The round's allocation is the caller's ``divmod``: every survivor gets
        `per_action` visits and the first `remainder` *positions* of `survivors`
        -- best-first order after the first cut, exactly as the sequential
        schedule reads it -- get one more. That is sliced the other way here:
        one wave per visit index, holding every survivor that still has a visit
        at that index. So the round runs ``per_action`` full waves and, when
        `remainder` is nonzero, one short one, and no wave is ever wider than
        the round's survivor count.

        Waves are submitted in ascending action order, which is the order the
        answers are expanded and backed up in. The order is arbitrary -- the
        survivors' state is disjoint (module docstring) -- so it is fixed at
        "ascending" simply so that two runs of the same search build their
        waves identically.
        """
        ran = 0
        waves = per_action + (1 if remainder else 0)
        for index in range(waves):
            actions = sorted(
                survivors if index < per_action else survivors[:remainder]
            )
            self._run_one_wave(wave_evaluator, root, root_state, actions)
            ran += len(actions)
        return ran

    def _run_one_wave(
        self,
        wave_evaluator: AnyWaveEvaluator,
        root: _Node,
        root_state: SearchState,
        actions: list[int],
    ) -> None:
        """Run one simulation per action in `actions`, batching the evaluations.

        Three passes over the wave: descend every action to its leaf; submit
        the leaves that need the network as a single call; expand and back up
        the answers. A descent that reached a terminal leaf is complete without
        the network and is backed up in the first pass -- its value comes from
        the game, and waiting for a wave it is not in would only delay it. If
        *every* descent did that the wave is empty and nothing is submitted,
        which is a normal endgame occurrence rather than an error (design
        review D8').
        """
        pending: list[_PendingLeaf] = []
        for action in actions:
            path, state, leaf = self._descend(root, root_state, action)
            if leaf is not None:
                _backup(path, leaf)
                continue
            pending.append(
                _PendingLeaf(
                    action=action,
                    path=path,
                    state=state,
                    legal=self._expansion_mask(state),
                    to_play=int(state.to_play()),
                )
            )

        if not pending:
            return

        # Never the first evaluation of the search: this runs inside
        # `_sequential_halving`, which `search` reaches only after
        # `_make_node(root_state)` has already gone through `self._evaluator`.
        # A stateful evaluator's "the first call is the root" reading therefore
        # holds on the wave schedule too (KOM-159 R9, `search`'s docstring).
        answers = wave_evaluator([row.state for row in pending])
        if len(answers) != len(pending):
            raise ValueError(
                f"wave evaluator returned {len(answers)} answers for a wave of "
                f"{len(pending)} positions (root actions "
                f"{[row.action for row in pending]}); answers must be "
                f"positionally aligned with the states submitted"
            )

        for row, answer in zip(pending, answers, strict=True):
            raw_logits, value, wdl = _unpack_evaluation(answer)
            node = self._evaluated_node(
                to_play=row.to_play,
                legal=row.legal,
                raw_logits=raw_logits,
                value=value,
                wdl=wdl,
            )
            parent, parent_edge = row.path[-1]
            parent.children[parent_edge] = node
            _backup(row.path, node)

    def _descend(
        self, root: _Node, root_state: SearchState, action: int
    ) -> tuple[list[tuple[_Node, int]], SearchState, _Node | None]:
        """Walk one simulation down to its leaf without asking the network.

        The descent :meth:`_simulate` makes, stopped one step earlier: the same
        root action, the same :func:`_select_interior` at every interior node,
        the same stopping rule. Returns the ``(parent, edge)`` path, the leaf
        *position*, and the leaf node -- which is ``None`` exactly when the
        descent reached an unexpanded, non-terminal position, i.e. the one case
        that needs an evaluation. A terminal leaf is built and attached here,
        because building it costs no evaluator call.
        """
        state = root_state.clone()
        node = root
        edge = action
        path: list[tuple[_Node, int]] = []

        while True:
            path.append((node, edge))
            state.play(edge)
            child = node.children.get(edge)
            if child is None:
                if state.is_terminal():
                    leaf = self._make_node(
                        state, want_wdl=root.child_wdl_sums is not None
                    )
                    node.children[edge] = leaf
                    return path, state, leaf
                return path, state, None
            if child.is_terminal:
                return path, state, child
            node = child
            edge = _select_interior(child, self._config)

    def _make_node(self, state: SearchState, *, want_wdl: bool = False) -> _Node:
        """Return a node for `state`, evaluating it unless it is terminal.

        `want_wdl` asks a **terminal** state for its one-hot
        (:meth:`~komugi.search.state.SearchState.terminal_wdl`) as well as its
        scalar, and is set by the simulation paths exactly when the root
        allocated the searched-WDL accumulator. It is off by default so that a
        state used only with a legacy two-tuple evaluator is never asked for a
        method it need not have: a terminal leaf's vector would have nowhere to
        go. A non-terminal state's vector comes from the evaluator's answer
        instead, whether or not this flag is set.
        """
        to_play = int(state.to_play())
        if state.is_terminal():
            return _Node(
                to_play=to_play,
                is_terminal=True,
                value=float(state.terminal_value()),
                legal=np.zeros(0, dtype=np.bool_),
                logits=np.zeros(0, dtype=np.float32),
                wdl=_terminal_wdl(state) if want_wdl else None,
            )

        mask = self._expansion_mask(state)
        raw_logits, value, wdl = _unpack_evaluation(self._evaluator(state))
        return self._evaluated_node(
            to_play=to_play,
            legal=mask,
            raw_logits=raw_logits,
            value=value,
            wdl=wdl,
        )

    def _expansion_mask(self, state: SearchState) -> npt.NDArray[np.bool_]:
        """Return a non-terminal leaf's legal mask, refusing an empty one."""
        mask = self._legal_mask(state)
        if not bool(mask.any()):
            raise ValueError(
                "state is not terminal but has no legal actions: the search "
                "has no move to make and no result to back up"
            )
        return mask

    def _evaluated_node(
        self,
        *,
        to_play: int,
        legal: npt.NDArray[np.bool_],
        raw_logits: npt.NDArray[np.float32],
        value: float,
        wdl: WdlAnswer | None = None,
    ) -> _Node:
        """Return the node an evaluation describes, checking what it returned.

        The single checkpoint **every** evaluator seam passes through -- the
        root and every sequential leaf (:meth:`_make_node`), every wave answer
        (:meth:`_run_one_wave`) and the forced-move path
        (:meth:`_forced_move`) -- so an answer is validated identically
        wherever it entered: same checks, same messages, no second opinion to
        drift. `wdl` is the evaluator's optional ``(w, d, l)`` for this leaf, in
        the leaf mover's frame; it is checked by :func:`_as_wdl`, which is this
        module's one WDL checkpoint, and stored as an owned ``float64`` copy.

        The scalar is checked here too (KOM-179). A ``NaN`` value used to be
        ``float()``-ed through in silence and had no local symptom at all: it
        entered ``child_value_sums``, came back out as a ``NaN`` ``v_mix``, and
        the search returned a ``NaN`` ``root_value`` beside an **all-NaN**
        policy target -- a training row that poisons a batch and names nothing
        that produced it.
        """
        # KOM-221 diagnostics: every evaluator answer passes through here exactly
        # once (root, sequential leaves, wave answers, forced move), so this is
        # the one place a network row is counted.
        self._nn_rows += 1
        logits = np.asarray(raw_logits, dtype=np.float32)
        if logits.shape != (self._config.num_actions,):
            raise ValueError(
                f"evaluator returned logits of shape {logits.shape}, expected "
                f"{(self._config.num_actions,)}"
            )
        # Checked on the raw output, before the mask is applied: a NaN on a
        # legal action turns the whole target into NaN with no other symptom,
        # and a -inf on one makes it indistinguishable from the masked-out
        # entries in the candidate ordering, which can put an illegal action
        # in the candidate set and then play it.
        if not bool(np.isfinite(logits).all()):
            raise ValueError("evaluator returned non-finite logits")
        scalar = float(value)
        if not math.isfinite(scalar):
            raise ValueError(f"evaluator returned a non-finite value: {scalar}")
        return _Node(
            to_play=to_play,
            is_terminal=False,
            value=scalar,
            legal=legal,
            logits=logits,
            wdl=None if wdl is None else _as_wdl(wdl),
        )

    def _forced_move(
        self, state: SearchState, action: int, legal: npt.NDArray[np.bool_]
    ) -> SearchResult:
        """Return the result for a position with exactly one legal action.

        `legal` is the caller's already-computed mask for `state`, passed in
        rather than recomputed so the forced path still asks the state for its
        legal set exactly once.

        No simulation is run -- searching a forced move spends budget to
        rediscover the only move there is. The value is still worth having,
        because the self-play loop trains on it, so the root is scored once.
        The target is the one-hot of the forced move, which is the honest
        distribution here rather than an artefact of skipping work.

        The root is evaluated on **both** branches, including the one where the
        game has already adjudicated the position and its ``root_value`` comes
        from the game rather than the network: the evaluation is what supplies
        :attr:`SearchResult.root_prior_logits`, and dropping it would leave the
        field with no honest value to hold. That costs nothing anyone was
        saving -- before KOM-040 D9' the self-play worker ran its own root prior
        evaluation on *every* move, this one included, so evaluating here keeps
        ``policy_kl`` semantics identical while every other move of the game
        gets one evaluation cheaper. The net is strictly fewer evaluations per
        game.

        **The WDL readouts here** (KOM-173 D4) are all three the root's own
        vector -- the evaluator's, or the game's one-hot when the position is
        already adjudicated -- for the same reason ``root_value`` is the root's
        own number: no simulation ran, so there is nothing to average and
        nothing that could disagree with it. ``NaN`` when the evaluator answered
        with a two-tuple, as everywhere else.
        """
        target = np.zeros(self._config.num_actions, dtype=np.float32)
        target[action] = 1.0
        # The ONLY evaluator call on this branch, and therefore trivially the
        # first: the forced-move path runs no simulation, so a stateful
        # evaluator sees exactly one state per search here and it is the root
        # (KOM-159 R9, `search`'s docstring).
        raw_logits, value, wdl = _unpack_evaluation(self._evaluator(state.clone()))
        # Through the SAME checkpoint every other path uses (KOM-179). This
        # branch used to `np.asarray` the logits and hand them straight back:
        # a width-1 vector was returned as `root_prior_logits` -- a field every
        # caller reads as ``(num_actions,)`` -- and a NaN logit or a NaN value
        # was accepted outright. One evaluation still, and the node built here
        # is scratch: nothing is simulated from it.
        root = self._evaluated_node(
            to_play=int(state.to_play()),
            legal=legal,
            raw_logits=raw_logits,
            value=value,
            wdl=wdl,
        )
        terminal = state.is_terminal()
        root_value = float(state.terminal_value()) if terminal else root.value
        # `None` here exactly when the evaluator answered with a two-tuple:
        # the checkpoint stores the vector it validated and nothing else.
        net_wdl = root.wdl_net
        if net_wdl is None:
            root_wdl = _nan_wdl()
        else:
            vector = _terminal_wdl(state) if terminal else net_wdl
            root_wdl = np.asarray(vector, dtype=np.float32)
        # There is nothing for the proof layer to decide here -- one legal move
        # is played whatever any interval says -- but the provenance still has
        # to say which algorithm produced the row, and the row counts still have
        # to add up: one root evaluation, no simulation, no leaf (KOM-221).
        #
        # The KOM-220 survey is bypassed here for the same reason: it exists to
        # widen a candidate set and to refute candidates, and a position with
        # one legal action has neither. The provider has still been validated
        # against the state (`search`), and it is still named in the record --
        # "the survey did not run, and here is what would have run it" is a
        # value, not an absence.
        config = self._config
        diagnostics: SearchDiagnostics | None = None
        if config.proof_mode != "off":
            discovery_active = config.discovery_mode != "off"
            diagnostics = SearchDiagnostics(
                identity=search_algorithm_identity(
                    config.proof_mode, config.discovery_mode
                ),
                ordinary_sims_used=0,
                root_nn_rows=1,
                ordinary_nn_rows=self._nn_rows - 1,
                provider_id=(
                    self._provider.provider_id
                    if discovery_active and self._provider is not None
                    else None
                ),
                discovery_scope=DISCOVERY_SCOPE if discovery_active else "off",
                discovery_status=(
                    "bypassed_forced_root" if discovery_active else "off"
                ),
                # The rule the root half runs under, which the mode fixes even
                # on the paths where the survey is bypassed before it can apply
                # it; `discovery_status` is what says whether it ran.
                root_scan=(
                    "exhaustive"
                    if config.discovery_mode == "root_win_v1"
                    else "provider"
                ),
            )
        return SearchResult(
            action=action,
            policy_target=target,
            root_prior_logits=root.logits,
            root_value=root_value,
            visit_counts=np.zeros(self._config.num_actions, dtype=np.int64),
            sims_used=0,
            root_wdl=root_wdl,
            # Separate arrays holding the same numbers: three fields a caller
            # may read independently must not be three names for one buffer.
            root_wdl_pi=root_wdl.copy(),
            played_wdl=root_wdl.copy(),
            diagnostics=diagnostics,
        )

    def _legal_mask(self, state: SearchState) -> npt.NDArray[np.bool_]:
        """Return `state`'s legal mask, checked against the configured shape."""
        mask = np.asarray(state.legal_mask(), dtype=np.bool_)
        if mask.shape != (self._config.num_actions,):
            raise ValueError(
                f"legal_mask returned shape {mask.shape}, expected "
                f"{(self._config.num_actions,)}"
            )
        return mask
