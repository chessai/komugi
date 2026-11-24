"""Frozen configuration objects.

These replace v1's ``dotdict`` pattern, where a typo in a key silently created
a new attribute. Every config here is a frozen, slotted dataclass with full
annotations, so unknown keyword arguments are a ``TypeError`` at construction
time and mutation is an ``AttributeError`` (both for free from ``dataclasses``
-- do not add ``**kwargs`` to any of these).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

# The ONE komugi import this module is allowed: ``komugi.engine.repetition`` is
# stdlib-only by construction (an Enum and two ints), so naming a repetition
# rule in a config costs nothing -- no Cython extension, no torch, no engine.
# Keep it that way: a config import that dragged in the engine would be paid for
# by every process that reads a config, spawned workers included.
from komugi.engine.repetition import RepetitionRule

if TYPE_CHECKING:
    # Types only. :mod:`komugi.search.modes` is itself stdlib-only, but
    # importing it at module scope would execute ``komugi.search.__init__``,
    # which pulls in the whole search (and numpy) -- the exact cost the import
    # rule above exists to keep off every process that reads a config. The
    # validator is therefore imported inside the one function that calls it.
    from komugi.search.modes import SearchAlgorithmIdentity

_MAX_MIN_PLACEMENTS: Final[int] = 25
"""Ceiling on :attr:`SelfPlayConfig.min_placements` -- one gungi hand.

A side drafts from a 26-piece hand, so a threshold above the hand is one no
draft can satisfy: every root would withhold 終 until the last piece, where
``draft_root_mask``'s fallback hands the mask back untouched. That is a safety
net for the position the engine forces, not a curriculum, and a run configured
into it would be masking nothing while reporting a threshold. v1's curriculum
ran ``10 -> 7 -> 4``, comfortably inside this bound.
"""

_MIN_GUMBEL_SCALE: Final[float] = 1e-6
_MAX_GUMBEL_SCALE: Final[float] = 1e6
"""Bounds on each Gumbel scale, so their **product** cannot degenerate.

Generous on purpose -- a temperature outside ``[1e-6, 1e6]`` is not a schedule
but a degenerate search, and ``tests/search/unit/test_gumbel_unit.py`` already
explores ``1e-6`` as the collapse limit. Every accepted pair composes into
``[1e-12, 1e12]``, comfortably inside float64's normal range in both
directions, which is what makes
:meth:`komugi.search.gumbel.GumbelConfig.__post_init__` unreachable from the
self-play path (KOM-036 §3.2).
"""

_CAP_ADJUDICATION_MODES: Final[tuple[str, ...]] = ("off", "material")
"""The values :attr:`SelfPlayConfig.cap_adjudication` may take (KOM-245).

``off`` is the cap draw every campaign so far has trained on; ``material``
labels a capped game by its final material difference. Listed here rather than
validated against a set literal inline, so the refusal names the whole set an
operator may choose from.
"""

_MAX_SEED_OPENINGS_PROB: Final[float] = 0.5
"""Ceiling on :attr:`SelfPlayConfig.seed_openings_prob` (KOM-254 D2).

Half, not one: a seeded game is a training game like any other -- its length,
its termination and its (absent) draft enter every iteration metric -- so a run
whose majority starts from harvested plateaus is a curriculum rather than
self-play, and the erosion census the curriculum is read by would be measuring
its own seeds. The pre-registered flip-on value is far below it (0.04).
"""


@dataclass(frozen=True, slots=True)
class SelfPlayConfig:
    """Self-play worker fan-out and the Gumbel search budget it plays with.

    Args:
        workers: How many worker processes the run fans out to.
        process_batch_size: The per-process batch hint for the parts of the
            system that never spawn anything.
        gumbel_m_full: Root actions sampled by a full search.
        gumbel_n_full: Simulation budget of a full search.
        gumbel_m_fast: Root actions sampled by a fast search.
        gumbel_n_fast: Simulation budget of a fast search.
        prob_fast: Probability a move draws the fast search (design D6/F1).
        draw_score: Search-time contempt. Never reaches a target (design D1).
        games_in_flight: How many games one worker process plays **at once**
            (KOM-040 b1). ``1`` is the default and is today's behaviour to the
            line: one game, one :class:`~komugi.selfplay.inference.SlotClient`,
            no threads at all. ``2`` hides the worker's measured 57% of wall
            spent blocked in ``SlotClient.submit`` -- while game A sleeps in
            the submit poll (``time.sleep`` releases the GIL) game B runs
            engine compute -- by giving each game its own thread, its own
            segment and its own client, so every doorbell invariant
            (one request in flight per slot, single writer, payload before
            counter) holds exactly as it does with one game. Only ``1`` or
            ``2``: three lanes per process is a measurement, not a guess, and
            nothing has measured it. Ignored (forced to one) by
            :func:`~komugi.selfplay.orchestrate.serial_debug`, whose whole
            claim is one game in one thread.
        book_path: The draft book this run's workers start ~
            ``book_fraction`` of their games from (KOM-014 §4.1), or ``""``
            for **no book at all** -- which is the default, and which is
            today's behaviour to the line: an empty path means
            :class:`~komugi.selfplay.worker.SelfPlayWorker` loads nothing, no
            game is booked, and every row is a searched ply exactly as before
            this field existed. In a campaign the path is not the caller's to
            choose: :func:`~komugi.campaign.loop.run_campaign` overrides it per
            iteration, the way it already owns ``run_seed`` and the shard root.
        book_fraction: Probability that one game starts from a book prefix,
            drawn per game on its own seed stream (KOM-014 §4.2). The
            pre-registered production value is ``0.10``; the **default here is
            0.0**, and the two are deliberately different numbers. §4.1's draft
            wrote ``0.10`` as the default while also refusing a positive
            fraction with an empty ``book_path``, and those two rules cannot
            both hold: every pre-KOM-014 construction passes neither field, so
            a 0.10 default would make each of them raise (acceptance #29 pins
            that they must not, and #28 pins that ``book_path="",
            book_fraction=0.1`` must). A fraction is a claim about a book, so
            it is the operator's to state together with one.
        fast_gumbel_scale: Multiplier on
            :attr:`komugi.search.gumbel.GumbelConfig.gumbel_scale` for a
            **fast** search (KOM-036 §2.3). ``1.0`` -- the default -- is off,
            and off is today's behaviour to the line: the composed scale is
            then exactly ``1.0`` and multiplying a float64 by exactly 1.0 is
            the identity (``gumbel.py:508-511``). The pre-registered experiment
            value is ``0.5``, which by KOM-036 §2.2 squares every pairwise odds
            ratio in the root prior -- "exploration damped" written down.
        draft_gumbel_scale: The same multiplier while **the mover is
            drafting** (KOM-036 §3.2), composed multiplicatively with
            ``fast_gumbel_scale`` because a scale is a temperature and
            temperatures compose on the log-odds scale. ``1.0`` is off; the
            pre-registered experiment value is ``2.0``, and a *fast draft* ply
            therefore lands at ``2.0 * 0.5 == 1.0`` -- intended, not
            accidental.

            Both scales are bounded to ``[1e-6, 1e6]`` rather than merely
            tested for finiteness and sign, and the bound is the whole point:
            finite-and-positive does **not** survive multiplication.
            ``1e-320 * 1e-10 == 0.0`` and ``1e200 * 1e200 == inf`` are both
            products of two individually-accepted factors, and either one makes
            :meth:`komugi.search.gumbel.GumbelConfig.__post_init__` raise on
            the first fast draft ply -- hours into a campaign. Inside the
            bound every accepted pair composes into ``[1e-12, 1e12]``, so the
            product is provably finite and positive. A successor widening this
            is giving up exactly that proof.
        repetition_rule: The fourfold-repetition key this run's games are
            refereed under (KOM-141). ``RepetitionRule.FULL`` -- the default --
            is Komugi's repaired 274-byte key (cells + side to move + draft
            flags + hands, ``docs/design/rules-audit.md`` F1-F4 / KOM-027), and
            is what every training row this project has ever produced was played
            under. ``RepetitionRule.PLACEMENT`` is gungi.js@3.4.11's cells-only
            key; it exists so a **referee** can apply the site's rule, and
            **self-play should not use it** -- it re-introduces the aliasing the
            audit documents, so a run configured with it is training on a rule
            Komugi deliberately rejected. It is settable here only because the
            worker must be able to say which rule it played under rather than
            assume; nothing in this repo sets it to anything but the default.
        min_placements: The draft-mask curriculum's threshold (v1's
            ``draftMinPlacements``, "DRAFTMIN"). ``0`` -- the default -- is
            **off**, and off is today's behaviour to the line: the root mask is
            then the engine's own ``valid_moves`` object, unchanged and
            un-copied. A positive ``N`` withholds 終 (draft-finish) from the
            **root** legal set until the side to move has placed ``N`` pieces
            in its draft, which is exactly
            :func:`komugi.engine.game.draft_root_mask`'s contract.

            ``DESIGN.html`` pre-registers the knob as a contingency -- "Draft
            mask curriculum retained as a contingency ... The mask tooling
            carries over, off by default" -- and campaign 2 is the contingency
            arriving: from an empty advanced-mode board the median side placed
            four pieces and ended its draft, which is v1's draft attractor
            reproduced. v1's mitigation was this mask run as a curriculum,
            ``10 -> 7 -> 4``, the operator relaunching at each ramp; nothing
            here schedules the ramp, because a threshold is a property of one
            generation and the campaign relaunches between them.

            **Root-only**, and that is the whole design: the mask is a *view*
            of the position the search is asked about, never a rule. Interior
            nodes, ``Game.valid_moves`` and ``win_state`` are untouched, so no
            training row, no adjudication and no engine invariant depends on
            it -- a run that sets this plays the same game under the same
            rules, and merely never *chooses* an early 終 at a root.

            A **no-op** wherever the draft is not live: intro and beginner
            start from laid-out boards with both draft bits clear, so
            ``draft_root_mask`` returns the input mask unchanged there for any
            value of this field.

        mlh_utility_lambda: Strength of the **moves-left conversion pressure**
            in search (KOM-159 D5/D7): the ``lambda`` of
            ``u = v + s*d + lambda * g(|v|) * sign(v) * f(m)``, the term that
            makes the side that is winning prefer the line that ends sooner and
            the side that is losing prefer the line that lasts. ``0.0`` -- the
            default -- is **off**, and off is today's behaviour to the line: the
            term is skipped entirely rather than added as a zero (design review
            F9), so no search that does not ask for it pays for it or can be
            changed by it. The pre-registered value is ``0.05``, taken at a
            pre-registered iteration after the head's calibration read passes;
            ``0.10`` is the single fallback arm.
        mlh_gate_tau0: Where the gate opens. Below ``|w - l| = tau0`` the term
            is exactly zero, so the pressure acts only on positions the value
            head already reads as decided; ``0.20`` is pre-registered.
        mlh_gate_tau1: Where the gate is fully open --
            ``g(|v|) = |v| * clip((|v| - tau0) / (tau1 - tau0), 0, 1)``, so the
            ramp between the two is linear and there is no discontinuity in
            utility between two siblings that straddle a hard threshold.
            ``0.40`` is pre-registered, and ``0 <= tau0 < tau1 <= 1``.
        mlh_slope_plies: How many plies of difference from the root's own
            moves-left prediction saturate the term:
            ``f(m) = clip((m_root - m) / mlh_slope_plies, -1, 1)``. ``32`` is
            pre-registered -- coarser than Lc0's ~13-ply saturation because
            this head's error budget is 40 plies and these games are longer, so
            a 32-ply difference between two winning lines is a plan rather than
            head noise. At least 1.

            **All four are inert here.** Nothing in this module and nothing in
            the trainer reads them; they are consumed at the *leaf* seam of
            self-play search, which is a separate change. They live on this
            config because they describe how a game is played, exactly as
            ``draw_score`` and the Gumbel budgets do, and because a campaign
            that turns the utility on must be able to say so in the same object
            the workers are handed.

        proof_mode: Which proof layer this run's searches run (KOM-221).
            ``off`` -- the default -- is today's pure Gumbel sequential-halving
            search to the line: the survivor is the decision and the completed-Q
            over the full legal action set is the target.
        discovery_mode: Which discovery layer runs before sequential halving
            (KOM-220). ``off`` is the default and costs nothing. Discovery only
            means anything when the proof layer is there to consume the terminal
            results it backs up, so ``root_marshal_v1`` requires
            ``proof_mode == "interval_v1"`` -- the pair is validated here, at
            construction, rather than at the first move of a launched run.

            The pair names the algorithm: :attr:`search_identity` derives the
            provenance identity from these two fields and from nothing else, so
            a recorded config always names exactly one search. There is no
            separately settable decision or target knob, by design.

        cap_adjudication: How a game stopped by the turn cap is labelled
            (KOM-245). ``off`` -- the default -- is today's behaviour to the
            line: a capped game is a draw and its rows carry the draw target.
            ``material`` labels it by the **final material difference** instead,
            as a soft target ramped by the two knobs below -- the cap draw is
            ~10% of games and the side that is materially ahead at it is not,
            by the counterfactual census, in a drawn position.
        cap_adjudication_p_max: The ceiling of that ramp: the win probability a
            fully-decided material lead is labelled with. ``0.8`` is
            pre-registered -- deliberately short of 1.0, because a material
            lead at the cap is evidence about the position and not a result.
            In ``(0, 1]``.
        cap_adjudication_d_full: Where the ramp reaches ``p_max``: a material
            difference of ``d_full`` or more is labelled ``p_max``, and a
            smaller lead is labelled proportionally. ``8`` is pre-registered.
            In ``[1, 25]`` -- the bound is a hand, as for ``min_placements``:
            with both marshals alive ``|d| <= 24`` at the cap, so a threshold
            above the hand is a ramp that can never be reached.
        min_placements_max: The **high** edge of the draft-mask threshold when
            it is drawn per side rather than fixed (KOM-246), with
            ``min_placements`` as the low edge: each side draws
            ``N ~ Uniform{min_placements .. min_placements_max}`` at the start
            of a game. ``0`` -- the default -- means *same as*
            ``min_placements``, which is today's behaviour to the line: one
            threshold, the one this config already names. Equal edges
            (``min_placements_max == min_placements``) mean the same thing and
            are accepted for the operator who prefers to write it out.

            A low edge of ``0`` is the mask switched OFF rather than a
            threshold, so pairing it with a non-zero high edge asks for a range
            that runs a share of the seats unmasked. That pair constructs --
            KOM-249 pinned this constructor to the hand bound alone -- and is
            refused by :func:`komugi.selfplay.records.draft_thresholds` at the
            first game that would draw from it.

            ``cap_adjudication*`` is still **inert here**, as KOM-159's four
            were when they landed: nothing in this module and no worker reads it
            yet -- KOM-245 is the change that consumes it. It lives on this
            config because it describes how a game is played and what its rows
            mean, and because a campaign that turns it on must be able to say so
            in the same object the workers are handed.
        seed_openings_path: The harvested plateau positions this run's workers
            start ``seed_openings_prob`` of their games from (KOM-254 D2), or
            ``""`` for **no seed openings at all** -- which is the default, and
            which is today's behaviour to the line: an empty path means
            :class:`~komugi.selfplay.worker.SelfPlayWorker` loads nothing, draws
            nothing (KOM-246's "no new code on the default path" doctrine) and
            plays exactly the games it played before this field existed. The
            file's schema and its guards are
            :func:`~komugi.selfplay.seed_openings.load_seed_openings`'s; it is
            loaded ONCE per worker, and an unloadable one fails the worker
            loudly rather than quietly seeding 0% while reporting the
            configured share.
        seed_openings_prob: Probability that one game starts from a harvested
            prefix, drawn per game on its own seed stream
            (:data:`~komugi.selfplay.seed_openings.SEED_OPENINGS_DOMAIN`). The
            default is ``0.0`` and the mechanism therefore lands **inert**; the
            flip-on value is the launcher's to state together with a path, for
            the reason ``book_fraction`` documents about itself.

            Bounded above by ``0.5`` rather than ``1.0``: seeded games are
            still training games, so their lengths, terminations and (absent)
            drafts enter every iteration metric, and a run more than half of
            which starts from harvested plateaus is a curriculum rather than
            self-play. A seeded game is never also booked -- the seed draw
            happens first and a seeded game skips the book consult entirely.

    Raises:
        ValueError: If ``games_in_flight`` is neither 1 nor 2, if
            ``book_fraction`` is outside ``[0, 1]``, if a positive
            ``book_fraction`` is paired with an empty ``book_path``, if either
            Gumbel scale falls outside ``[1e-6, 1e6]``, if ``min_placements``
            falls outside ``[0, 25]``, if any moves-left utility parameter is
            out of range, if a positive ``mlh_utility_lambda`` is paired
            with a non-zero ``draw_score``, if ``proof_mode`` /
            ``discovery_mode`` name a search nobody can run (unknown mode, or
            discovery without the proof layer it feeds), if
            ``cap_adjudication`` is not ``off``/``material``, if
            ``cap_adjudication_p_max`` falls outside ``(0, 1]``, if
            ``cap_adjudication_d_full`` falls outside ``[1, 25]``, if a
            non-zero ``min_placements_max`` is not in
            ``[min_placements, 25]``, if a non-zero ``min_placements_max``
            is paired with ``min_placements == 0`` -- a range that includes
            the mask's own off switch, refused here rather than at the first
            game that would draw from it -- if ``seed_openings_prob`` falls
            outside ``[0, 0.5]``, or if a positive ``seed_openings_prob`` is
            paired with an empty ``seed_openings_path``.
    """

    workers: int
    process_batch_size: int
    gumbel_m_full: int = 32
    gumbel_n_full: int = 256
    gumbel_m_fast: int = 8
    gumbel_n_fast: int = 32
    prob_fast: float = 0.75
    draw_score: float = 0.20
    games_in_flight: int = 1
    book_path: str = ""
    book_fraction: float = 0.0
    fast_gumbel_scale: float = 1.0
    draft_gumbel_scale: float = 1.0
    repetition_rule: RepetitionRule = RepetitionRule.FULL
    # Appended rather than slotted in beside the other search knobs: the fields
    # of a dataclass are positional parameters, so inserting one in the middle
    # would silently rebind every positional call site past it.
    min_placements: int = 0
    # KOM-159's four, appended for that same reason -- and appended TOGETHER,
    # because they are one lever: a lambda without its gate and its slope is
    # not a configuration anybody can read back off a run.
    mlh_utility_lambda: float = 0.0
    mlh_gate_tau0: float = 0.20
    mlh_gate_tau1: float = 0.40
    mlh_slope_plies: int = 32
    # KOM-223's two, appended for that same reason -- and appended together,
    # because the two modes are one lever: they name a search algorithm only as
    # a pair (see `search_identity`).
    proof_mode: str = "off"
    discovery_mode: str = "off"
    # KOM-249's four, appended for that same reason: KOM-245's cap-adjudication
    # lever (a mode and the two knobs that shape its ramp -- a mode without them
    # is not a rule anybody can read back off a run) and KOM-246's high edge for
    # the draft threshold, landed together so the two follow-on changes can be
    # written against one field order rather than race each other's tails.
    cap_adjudication: str = "off"
    cap_adjudication_p_max: float = 0.8
    cap_adjudication_d_full: int = 8
    min_placements_max: int = 0
    # KOM-254's two, appended for that same reason -- and appended together,
    # because they are one lever: a share without a file is a run that seeds
    # nothing while reporting that it does, and the config refuses that pair
    # below exactly as it refuses `book_fraction` without a `book_path`.
    seed_openings_path: str = ""
    seed_openings_prob: float = 0.0

    def __post_init__(self) -> None:
        """Reject a lane count nobody has measured, and a book nobody has."""
        if self.games_in_flight not in (1, 2):
            raise ValueError(
                f"games_in_flight must be 1 or 2, got {self.games_in_flight!r}: "
                f"one game per worker is the default, two is the interleaved "
                f"lane of KOM-040 b1, and more lanes than that would need the "
                f"measurement that sized this knob in the first place"
            )
        if not 0.0 <= self.book_fraction <= 1.0:
            # `not <=` rather than `<`/`>` so a NaN fraction is refused here
            # rather than discovered as a run that booked nothing.
            raise ValueError(
                f"book_fraction must be in [0, 1], got {self.book_fraction!r}"
            )
        if self.book_fraction > 0.0 and not self.book_path:
            raise ValueError(
                f"book_fraction={self.book_fraction!r} with an empty book_path: "
                f"a run configured to book that share of its games and handed no "
                f"book would quietly play a bookless generation and report "
                f"success (KOM-014 §4.1). Give a book_path, or set "
                f"book_fraction=0.0"
            )
        if not 0 <= self.min_placements <= _MAX_MIN_PLACEMENTS:
            raise ValueError(
                f"min_placements must be in [0, {_MAX_MIN_PLACEMENTS}], got "
                f"{self.min_placements!r}: the bound is a hand, and a threshold "
                f"no side can reach would withhold 終 for the whole draft -- "
                f"the mask's own last-piece fallback is a safety net, not a "
                f"licence to configure a draft that can never end"
            )
        for field, scale in (
            ("fast_gumbel_scale", self.fast_gumbel_scale),
            ("draft_gumbel_scale", self.draft_gumbel_scale),
        ):
            # `not <=` rather than `<`/`>` so a NaN scale is refused here as
            # well. The bound, not merely finiteness and sign: see the field
            # docstrings -- two individually-accepted factors can compose to
            # 0.0 or inf, and GumbelConfig would then raise on the first fast
            # draft ply of a campaign that had already been running for hours.
            if not _MIN_GUMBEL_SCALE <= scale <= _MAX_GUMBEL_SCALE:
                raise ValueError(
                    f"{field} must be in [{_MIN_GUMBEL_SCALE}, "
                    f"{_MAX_GUMBEL_SCALE}], got {scale!r}: the bound is what "
                    f"makes the composed draft x fast product provably finite "
                    f"and > 0, which finite-and-positive on each factor alone "
                    f"does not (1e-320 * 1e-10 == 0.0, 1e200 * 1e200 == inf)"
                )
        # `not <=` throughout, so a NaN is refused here rather than discovered
        # as a search whose every leaf utility is NaN.
        if not 0.0 <= self.mlh_utility_lambda <= 1.0:
            raise ValueError(
                f"mlh_utility_lambda must be in [0, 1], got "
                f"{self.mlh_utility_lambda!r}"
            )
        if not 0.0 <= self.mlh_gate_tau0 < self.mlh_gate_tau1 <= 1.0:
            raise ValueError(
                f"the moves-left gate needs 0 <= tau0 < tau1 <= 1, got "
                f"tau0={self.mlh_gate_tau0!r}, tau1={self.mlh_gate_tau1!r}: "
                f"the ramp's denominator is tau1 - tau0, so an inverted or "
                f"empty band is a division by zero at the first gated leaf "
                f"rather than a narrower gate"
            )
        if self.mlh_slope_plies < 1:
            raise ValueError(
                f"mlh_slope_plies must be >= 1, got {self.mlh_slope_plies!r}: "
                f"it is the denominator of the moves-left difference"
            )
        if self.mlh_utility_lambda > 0.0 and self.draw_score != 0.0:
            # Design review F13. The conversion term is ODD under the frame
            # swap (w <-> l), which is what lets negamax re-express it as the
            # same preference at every depth; the contempt term `s * d` is
            # EVEN, so it oscillates with depth parity (kom-021 D1 F3b). The
            # two have never been measured together, and `lambda > 0` is
            # pre-registered at contempt 0 only -- so a run that asked for both
            # is refused here rather than producing a policy target nobody can
            # attribute to either lever.
            raise ValueError(
                f"mlh_utility_lambda={self.mlh_utility_lambda!r} with "
                f"draw_score={self.draw_score!r}: the moves-left conversion "
                f"term is pre-registered at contempt 0 only (KOM-159 review "
                f"F13). Set draw_score=0.0, or leave mlh_utility_lambda at 0.0"
            )
        if self.cap_adjudication not in _CAP_ADJUDICATION_MODES:
            raise ValueError(
                f"cap_adjudication must be one of "
                f"{list(_CAP_ADJUDICATION_MODES)}, got "
                f"{self.cap_adjudication!r}: the mode names how a capped game "
                f"is labelled, and an unknown name is a run whose rows nobody "
                f"can attribute to a rule"
            )
        # `not <` / `not <=` so a NaN is refused here rather than discovered as
        # a shard full of NaN targets.
        if not 0.0 < self.cap_adjudication_p_max <= 1.0:
            raise ValueError(
                f"cap_adjudication_p_max must be in (0, 1], got "
                f"{self.cap_adjudication_p_max!r}: it is the win probability a "
                f"fully-decided material lead is labelled with, and 0 would be "
                f"a ramp that labels the leader as the loser"
            )
        if not 1 <= self.cap_adjudication_d_full <= _MAX_MIN_PLACEMENTS:
            raise ValueError(
                f"cap_adjudication_d_full must be in [1, "
                f"{_MAX_MIN_PLACEMENTS}], got {self.cap_adjudication_d_full!r}: "
                f"it is the ramp's denominator, and the bound is a hand -- with "
                f"both marshals alive the material difference at the cap is at "
                f"most 24, so a larger threshold is a ramp no game can reach"
            )
        # The hand bound and the direction of the range first, so a high edge
        # nobody can play is reported as the edge it is; the off-switch rule
        # below is about a range that is well-formed and still wrong.
        if self.min_placements_max != 0 and not (
            self.min_placements <= self.min_placements_max <= _MAX_MIN_PLACEMENTS
        ):
            raise ValueError(
                f"min_placements_max must be 0 (same as min_placements) or in "
                f"[{self.min_placements}, {_MAX_MIN_PLACEMENTS}], got "
                f"{self.min_placements_max!r} with "
                f"min_placements={self.min_placements!r}: it is the HIGH edge "
                f"of the per-side draw, so an edge below the low one is an "
                f"empty range and one above a hand is a threshold no draft can "
                f"reach"
            )
        # A low edge of 0 with a non-zero high edge is a range that includes
        # the mask's own off switch (design review F7).
        if self.min_placements == 0 and self.min_placements_max != 0:
            raise ValueError(
                f"min_placements={self.min_placements!r} with "
                f"min_placements_max={self.min_placements_max!r}: 0 is the "
                f"draft mask switched OFF, not a threshold, so a drawn range "
                f"whose low edge is 0 would run a share of the seats unmasked "
                f"-- a silently mixed population, which is the one thing a "
                f"lever meant to SPREAD the population must not be. The low "
                f"edge of a drawn range must be at least 1"
            )
        # `not <=` rather than `<`/`>` so a NaN share is refused here rather
        # than discovered as a run that seeded nothing.
        if not 0.0 <= self.seed_openings_prob <= _MAX_SEED_OPENINGS_PROB:
            raise ValueError(
                f"seed_openings_prob must be in [0, {_MAX_SEED_OPENINGS_PROB}], "
                f"got {self.seed_openings_prob!r}: seeded games are training "
                f"games, so their lengths, terminations and absent drafts enter "
                f"every iteration metric -- a run more than half of which starts "
                f"from harvested plateaus is a curriculum, not self-play"
            )
        if self.seed_openings_prob > 0.0 and not self.seed_openings_path:
            raise ValueError(
                f"seed_openings_prob={self.seed_openings_prob!r} with an empty "
                f"seed_openings_path: a run configured to seed that share of its "
                f"games and handed no file would quietly play an unseeded "
                f"generation and report success (KOM-254 D2, the book's rule and "
                f"its reason). Give a seed_openings_path, or set "
                f"seed_openings_prob=0.0"
            )
        # Last, and by the search's own validator rather than a copy of its
        # rules: an unknown mode, or a discovery layer with no proof layer to
        # consume what it backs up, is refused where the run is configured
        # rather than at the first move of a launched campaign.
        from komugi.search.modes import validate_search_modes

        validate_search_modes(self.proof_mode, self.discovery_mode)

    @property
    def search_identity(self) -> SearchAlgorithmIdentity:
        """Return the provenance identity of the search these modes select.

        **Derived, never a knob**: the decision rule and the training target are
        a function of ``(proof_mode, discovery_mode)`` and of nothing else, so a
        manifest that records this pair always names exactly one algorithm.
        """
        from komugi.search.modes import search_algorithm_identity

        return search_algorithm_identity(self.proof_mode, self.discovery_mode)

    def __getstate__(self) -> tuple[object, ...]:
        """Return every field value, in field order, as a plain tuple.

        The orchestrator pickles this object into each spawned worker, and a
        worker respawned after a code update imports NEW code while its parent
        still holds -- and sends -- the payload the OLD code built. A positional
        tuple plus the length-dispatching :meth:`__setstate__` below is what
        makes that crossing legible: see there for the contract.
        """
        return _config_getstate(self)

    def __setstate__(self, state: object) -> None:
        """Restore from a current (27-field) or legacy (25-, 21- or 19-field) payload.

        ``_LEGACY_STATE_FIELDS`` is the field count as of KOM-159, before the
        two search modes were appended; such a payload predates the modes, so it
        is restored as ``off``/``off``, which is exactly the search it was
        produced by. ``_KOM223_STATE_FIELDS`` is the count after them and before
        KOM-249's four, which are restored at their defaults for the same
        reason: a payload that predates cap adjudication and the drawn draft
        threshold was produced by a run that had neither. Any OTHER length is
        refused rather than padded: it is a payload this build cannot interpret,
        and a silently-defaulted field would be a run whose recorded
        configuration is not the one it played.

        Raises:
            ValueError: If *state* is not a tuple, or is a tuple of any length
                other than 27, 25, 21 or 19, or if the restored modes are not a
                pair the search can run (the same validation ``__post_init__``
                applies).
        """
        _config_setstate(self, state)


_LEGACY_STATE_FIELDS: Final[int] = 19
"""Field count of :class:`SelfPlayConfig` before the search modes (KOM-159).

Frozen literals, not ``len(fields) - 2``: the legacy payload's shape is history
and cannot change, so a future appended field must appear here as a new accepted
length rather than shift what "legacy" means.
"""

_KOM223_STATE_FIELDS: Final[int] = 21
"""Field count as of KOM-223: the search modes appended, KOM-249's four not yet.

A second frozen literal, by the rule above: when KOM-249 appended cap
adjudication and the draft threshold's high edge, the KOM-223-era payload became
history in its turn, and a parent still running that code can pickle one into a
worker respawned on this one.
"""

_KOM249_STATE_FIELDS: Final[int] = 25
"""Field count as of KOM-249: cap adjudication and the drawn draft threshold in,
KOM-254's seed-openings pair not yet.

The third frozen literal, by the same rule: a parent running KOM-249-era code
pickles a 25-field payload into a worker respawned on this code, and that
payload was produced by a run with no seed openings -- restored as
``seed_openings_path=""``, ``seed_openings_prob=0.0``, the configuration it
actually played.
"""

_CURRENT_STATE_FIELDS: Final[int] = 27


def _config_getstate(config: SelfPlayConfig) -> tuple[object, ...]:
    """Return every field of *config*, in field order, as a plain tuple."""
    return tuple(getattr(config, f.name) for f in dataclasses.fields(config))


def _config_setstate(config: SelfPlayConfig, state: object) -> None:
    """Restore *config* from a current or legacy positional payload."""
    # The default dataclass hook on the PRE-KOM-223 class sends a 19-element LIST
    # (dataclasses._dataclass_getstate); a parent started on that code can pickle it
    # into a worker respawned on this code, so a list is the legacy shape too.
    if isinstance(state, list):
        state = tuple(state)
    if not isinstance(state, tuple):
        raise ValueError(
            f"SelfPlayConfig.__setstate__ expects a tuple of field values, "
            f"got {type(state).__name__}"
        )
    # Each legacy arm pads exactly the fields appended after its era, at their
    # defaults -- which is not a guess but the configuration that payload was
    # produced by: a parent that predates a field ran without it.
    if len(state) == _LEGACY_STATE_FIELDS:
        values: tuple[object, ...] = (
            *state, "off", "off", "off", 0.8, 8, 0, "", 0.0,
        )
    elif len(state) == _KOM223_STATE_FIELDS:
        values = (*state, "off", 0.8, 8, 0, "", 0.0)
    elif len(state) == _KOM249_STATE_FIELDS:
        values = (*state, "", 0.0)
    elif len(state) == _CURRENT_STATE_FIELDS:
        values = state
    else:
        raise ValueError(
            f"SelfPlayConfig.__setstate__ expects {_CURRENT_STATE_FIELDS} "
            f"field values (or the legacy {_KOM249_STATE_FIELDS}, "
            f"{_KOM223_STATE_FIELDS} or {_LEGACY_STATE_FIELDS}), got "
            f"{len(state)}"
        )
    names = tuple(f.name for f in dataclasses.fields(config))
    # object.__setattr__ because the class is frozen (and slotted, so there is
    # no __dict__ to update instead); strict=True because a field count that has
    # drifted from the two constants above is a pickle contract that no longer
    # holds, and it must fail here rather than restore a half-populated config.
    for name, value in zip(names, values, strict=True):
        object.__setattr__(config, name, value)
    # Exactly the validation a constructed config gets -- arriving over a pipe
    # is not an exemption.
    config.__post_init__()


# The two hooks live in the class body (so they are documented where they are
# used, and so a caller can type-check `cfg.__setstate__`), and are re-attached
# here from the module-level implementations above: `@dataclass(slots=True)`
# RECREATES the class object, and a frozen slotted dataclass installs pickle
# hooks of its own -- guarded against clobbering a user's on current CPython,
# unguarded on earlier ones. This line makes the transport contract independent
# of that interpreter detail rather than of a patch version nobody pinned. It is
# a no-op whenever the class-body definitions survived.
setattr(SelfPlayConfig, "__getstate__", _config_getstate)  # noqa: B010
setattr(SelfPlayConfig, "__setstate__", _config_setstate)  # noqa: B010


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """Replay-window and optimiser settings for a training run."""

    window: int = 5
    batch_size: int = 512
    lr: float = 0.01
