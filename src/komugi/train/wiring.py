"""The one place :class:`~komugi.net.model.NetOutput` meets the loss assembler.

Three vocabularies have to line up before a single gradient is correct: the
network's output field names, the sample schema's target field names
(:data:`komugi.selfplay.records.SCHEMA`), and the assembler's key constants
(:mod:`komugi.train.losses`). They *nearly* agree, and the near-misses are
exactly the kind that fail silently:

* ``NetOutput.moves_left`` versus the assembler's ``MLH_PRED == "mlh"`` -- a
  plain rename, so handing the assembler ``out._asdict()`` would raise a
  ``KeyError`` for the moves-left head and nothing at all for the other three.
* ``NetOutput.soft_policy_logits`` versus the assembler's soft-policy term,
  which until the rewire this module accompanies read ``policy_logits``. That
  one did *not* raise: it trained the main policy head against the sharpened
  auxiliary target while ``SoftPolicyHead`` received no gradient at all, at
  full forward cost. See ``tests/train/test_netoutput_wiring.py``, which was a
  strict-xfail tripwire until the rewire landed and is now the regression test.

So the translation lives here, once, and the trainer never spells a key.

Surprise weights from the stored scalar
---------------------------------------
:func:`komugi.train.losses.policy_surprise_weight` computes the KL itself from
an ``(A,)`` recorded prior -- a vector design D4/F2 deliberately does **not**
store, because four bytes of ``policy_kl`` per row buy the same weighting at
~1/9000th of the size. The wave-9b review recorded the consequence: the
promised path from that stored scalar to ``assemble(..., sample_weight=)``
"exists nowhere and no test pins it" (KOM-020, 2026-08-23 follow-up).
:func:`surprise_weights_from_kl` is that path, and it is the *only* one the
trainer uses; ``policy_surprise_weight`` stays for callers that do hold a
prior vector (a fresh search result in memory, the golden tests).

The fast-row weight
-------------------
:func:`fast_row_weights` is the other schema-to-loss translation this module
owns: the ``search_mode`` column becomes a per-row **multiplier** on the policy
and soft-policy heads' sample weights, which is the entire mechanism of
KOM-036's pre-registered down-weight-versus-discard A/B. It lives here rather
than in :mod:`komugi.train.losses` for the same reason everything else here
does -- the assembler knows heads, and must not learn what a self-play row is.

The cap-row weight
------------------
:func:`cap_row_weights` is the third, and the same shape as the second: the
``termination`` column becomes a per-row multiplier on the **WDL** head's
sample weights, which is the entire mechanism of KOM-160's campaign 4D. The two
functions are deliberately not one parametrised helper -- they key on different
columns, weight different heads, and their refusals mean different things -- but
they share their arithmetic, their device rule and their refuse-an-unknown-code
discipline, and a change to one should be read against the other.

The moves-left row weight
-------------------------
:func:`mlh_row_weights` is the fourth, and the first to key on *three* columns
rather than one: ``ply``, ``mlh_target`` and ``termination`` together
reconstruct "this row's game ended at the turn cap", which is the predicate
KOM-159 D4 censors the moves-left head on. It keys :data:`HEAD_MLH
<komugi.train.losses.HEAD_MLH>` alone, exactly as :func:`cap_row_weights` keys
``HEAD_WDL`` alone, and it shares that function's device rule and its
refuse-an-unknown-code discipline -- a change to one should be read against the
other.

The recipe
----------
:data:`P0A_RECIPE` is the loss configuration the first Komugi training run
uses, pinned field by field by a test so that no run can inherit
:class:`~komugi.train.losses.LossConfig`'s defaults by accident and nobody has
to reconstruct months later which weights a curve was produced under.

:data:`P0B_RECIPE` is the production recipe from P0b on: the same five P0a
weights plus the four staged heads at their OFF defaults, stated rather than
inherited. It is a **new name** rather than an edit to the old one for exactly
the reason above -- the P0a curves exist, and the constant that produced them
must keep producing them.

:data:`MLH_RECIPE` is campaign 6's: :data:`P0B_RECIPE` field for field with the
moves-left head **on at weight 1.0** (KOM-159 D1/D3). A third name for the same
reason there is a second.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import TYPE_CHECKING, Final

import torch

from komugi.modes import MODE_FAST, MODE_FULL
from komugi.terminations import TERMINATION_DECISIVE, TERMINATION_DRAW
from komugi.train.losses import (
    DRAFT_OUTCOME_LOGITS,
    DRAFT_OUTCOME_TARGET,
    FINAL_CONTROL_TARGET,
    MLH_PRED,
    MLH_TARGET,
    OWNERSHIP_LOGITS,
    POLICY_LOGITS,
    POLICY_TARGET,
    SHORT_TERM_VALUE_PRED,
    SHORT_TERM_VALUE_TARGET,
    SOFT_POLICY_LOGITS,
    UNCERTAINTY_PRED,
    WDL_LOGITS,
    WDL_TARGET,
    LossConfig,
)

if TYPE_CHECKING:  # pragma: no cover -- import for typing only
    # Imported under TYPE_CHECKING so that :mod:`komugi.train` keeps its
    # runtime independence from :mod:`komugi.net` (see the package docstring):
    # this module reads four attributes off whatever it is handed, and the
    # annotation is what makes a wrong object a type error rather than an
    # AttributeError at step 1.
    from komugi.net import NetOutput

TARGET_KEYS: Final[tuple[str, ...]] = (
    POLICY_TARGET,
    WDL_TARGET,
    MLH_TARGET,
    FINAL_CONTROL_TARGET,
    SHORT_TERM_VALUE_TARGET,
    DRAFT_OUTCOME_TARGET,
)
"""Sample-schema fields the assembler consumes, in schema order.

Every name here is spelled identically in
:data:`komugi.selfplay.records.SCHEMA` and in :mod:`komugi.train.losses`, so
the mapping is the identity -- which is a fact worth pinning rather than
assuming, since the identity is what makes the *other* schema fields
(``policy_kl``, ``root_value``, the provenance ints) obviously not targets.
``prior_logits`` is absent from both this tuple and the schema on purpose; see
:func:`surprise_weights_from_kl`.

``final_control`` predates the staged heads on the wire (KOM-010) and becomes
a *target* only now that the ownership head reads it; ``short_term_value`` and
``draft_outcome`` arrive with them. There is deliberately no uncertainty entry:
that head's target is derived inside the assembler from ``wdl_target`` and the
net's own ``wdl_logits``, so it costs no column.
"""


def outputs_from_net(out: NetOutput) -> dict[str, torch.Tensor]:
    """Return `out`'s eight predictions under the assembler's output keys.

    Kills the two key mismatches named in the module docstring:
    ``soft_policy_logits`` reaches the soft-policy term (it used to read the
    main head's logits, silently), and ``moves_left`` reaches ``"mlh"`` (it
    used to reach nothing at all). The four staged keys are spelled exactly as
    their ``NetOutput`` fields are, so ``moves_left`` remains the only rename
    this function performs.

    Every head is included whether or not the config in force enables it: the
    assembler ignores the outputs of a disabled head, and a mapping whose
    contents depended on the loss config would make "which heads were fed" a
    second place the staging is decided. That rule is exactly what the four
    staged heads need -- they are forward-evaluated from the first P0b net and
    their losses are off.
    """
    return {
        POLICY_LOGITS: out.policy_logits,
        SOFT_POLICY_LOGITS: out.soft_policy_logits,
        WDL_LOGITS: out.wdl_logits,
        MLH_PRED: out.moves_left,
        UNCERTAINTY_PRED: out.uncertainty,
        OWNERSHIP_LOGITS: out.ownership_logits,
        SHORT_TERM_VALUE_PRED: out.short_term_value,
        DRAFT_OUTCOME_LOGITS: out.draft_outcome_logits,
    }


def targets_from_batch(batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Return the assembler's targets from a stacked sample `batch`.

    `batch` is a batch of :data:`komugi.selfplay.records.SCHEMA` rows as
    :func:`komugi.train.shards.stack_samples` produces it, so it also carries
    the observation, the provenance ints, ``root_value`` and ``policy_kl``.
    Only the three fields the assembler consumes are passed through -- the rest
    are not targets, and handing them over would make a future typo in a head
    name find a plausible tensor to train on.

    A field the batch does not carry is skipped rather than faked: which
    targets are *required* is a function of the loss config, and the assembler
    already raises a ``KeyError`` naming the head that wanted a missing one.
    """
    return {key: batch[key] for key in TARGET_KEYS if key in batch}


def surprise_weights_from_kl(
    policy_kl: torch.Tensor,
    *,
    alpha: float,
    max_weight: float,
) -> torch.Tensor:
    """Return per-sample surprise weights from the *stored* ``policy_kl``.

    ``clip(1 + alpha * clamp(policy_kl, min=0), 1, max_weight)`` over a
    ``(batch,)`` vector of ``KL(policy_target || masked-softmax(root prior))``
    values recorded at self-play time (schema field ``policy_kl``). The result
    goes to :meth:`komugi.train.losses.LossAssembler.__call__` as
    ``sample_weight=``, which is the whole of the wiring the wave-9b review
    found missing.

    Same arithmetic and same clip as
    :func:`komugi.train.losses.policy_surprise_weight`, from the scalar instead
    of from a prior vector the shards do not carry. The lower clip means
    surprise can only ever *up*-weight; the upper one stops a single sample
    whose prior put ~0 mass where search went from owning a batch. The
    ``clamp(min=0)`` is defensive about the data rather than the arithmetic: a
    KL is non-negative, so a negative value on disk is a corrupt or
    misconstructed row, and folding it to 0 keeps it from being read as
    "search agreed *more* than perfectly".

    Args:
        policy_kl: ``(batch,)`` recorded KL, one per sample. Any float dtype;
            the weights come back in it.
        alpha: Surprise strength. ``0`` returns exact ones -- the off switch,
            and the reason this is safe to call unconditionally.
        max_weight: The clip's upper bound; must be at least the lower bound
            of 1, since a smaller ceiling would make the clip empty.

    Raises:
        ValueError: If `alpha` is negative, `max_weight` is below 1, or
            `policy_kl` is not 1-D. A wrong-rank input would broadcast against
            the per-sample losses and reweight the wrong axis silently, which
            is precisely the failure the assembler's own length check exists
            to stop.
    """
    if alpha < 0.0:
        raise ValueError(f"alpha must be >= 0, got {alpha}")
    if max_weight < 1.0:
        raise ValueError(
            f"max_weight must be >= 1 (the clip's lower bound), got {max_weight}"
        )
    if policy_kl.ndim != 1:
        raise ValueError(
            f"policy_kl must be 1-D over the batch, got shape {tuple(policy_kl.shape)}"
        )
    if alpha == 0.0:
        return torch.ones_like(policy_kl)
    return (1.0 + alpha * policy_kl.clamp_min(0.0)).clamp(min=1.0, max=max_weight)


def fast_row_weights(
    search_mode: torch.Tensor, fast_weight: float
) -> torch.Tensor:
    """Return the per-row policy multiplier keyed on the ``search_mode`` column.

    ``1.0`` at :data:`~komugi.modes.MODE_FULL`, `fast_weight` at
    :data:`~komugi.modes.MODE_FAST` -- the whole mechanism of KOM-036's
    pre-registered P0b A/B, whose two arms are this one float at ``0.125``
    (down-weight, the Gumbel-native choice, derived as ``gumbel_n_fast /
    gumbel_n_full = 32 / 256``) and at ``0.0`` (PCR-style discard). ``1.0`` is
    arm C, today's behaviour, and it runs the *same* code path: the vector is
    then exactly ones and ``weights * 1.0`` is the identity, so all three arms
    differ in one number and in nothing else.

    The result is a **multiplier** for
    :meth:`komugi.train.losses.LossAssembler.__call__`'s
    ``head_sample_weight=``, applied to the policy and soft-policy heads only
    (§1.4: the soft-policy target is *derived from the same*
    ``policy_target``, at eight times the policy head's weight, so a discard
    arm that left it alone would be a no-op wearing a label). WDL and MLH keep
    the unmodified base, because a fast move still advances the game and its
    outcome-derived targets are exactly as good as a full move's.

    The mode codes come from :mod:`komugi.modes`, a leaf at the package root
    that imports nothing. **Not** from :mod:`komugi.selfplay.worker`, which
    closes an import cycle that raises on ``import komugi.selfplay``; and not
    from a hypothetical ``komugi.selfplay.modes`` either, because Python draws
    its import edge at *package* granularity and that spelling would execute
    the whole self-play stack from the trainer. Not restated here as ``0``/``1``
    either: a second source of truth for the schema's vocabulary is what the
    leaf exists to prevent.

    Args:
        search_mode: The batch's ``(batch,)`` integer ``search_mode`` column,
            spelled as :data:`komugi.selfplay.records.SCHEMA` writes it.
        fast_weight: The arm. Any finite non-negative float; the assembler
            validates the composed vector.

    Returns:
        A ``(batch,)`` float32 multiplier on the same device as `search_mode`.

    Raises:
        ValueError: If `search_mode` is not 1-D, or carries a value that is
            neither mode. An unknown code is refused rather than silently
            taking the full-row weight: a third search mode is a schema change
            whose stratum this function has no arm for, and reading it as
            "full" would train it at a weight nobody chose.
    """
    if search_mode.ndim != 1:
        raise ValueError(
            f"search_mode must be 1-D over the batch, got shape "
            f"{tuple(search_mode.shape)}"
        )
    # Branchless, and the membership check folded into the SAME reduction as
    # the value it guards: a Python branch on tensor values is a host sync, and
    # `losses.py` deliberately budgets exactly one of those per training step.
    # Stacking the two out-of-range predicates into one `.item()`-bearing
    # tensor keeps the refusal free rather than doubling that budget.
    out_of_range = torch.stack(
        (
            (search_mode < MODE_FULL).any(),
            (search_mode > MODE_FAST).any(),
        )
    ).any()
    if bool(out_of_range.item()):
        raise ValueError(
            f"search_mode carries a value outside {{{MODE_FULL}, "
            f"{MODE_FAST}}} (full, fast); the fast-row weight has no arm for "
            f"a third search mode and must not read it as a full row"
        )
    return torch.where(
        search_mode == MODE_FAST,
        torch.full((), float(fast_weight), device=search_mode.device),
        torch.ones((), device=search_mode.device),
    )


def cap_row_weights(termination: torch.Tensor, weight: float) -> torch.Tensor:
    """Return the per-row WDL multiplier keyed on the ``termination`` column.

    ``1.0`` at :data:`~komugi.terminations.TERMINATION_DECISIVE`, `weight` at
    :data:`~komugi.terminations.TERMINATION_DRAW` -- the whole mechanism of
    KOM-160's campaign 4D, whose pre-registered arm is ``0.25`` and whose
    fallback is ``0.6`` (``155 / 256``, the row-count parity a cap game would
    have if it were as long as a decisive one). ``1.0`` is today's behaviour and
    runs the *same* code path: the vector is then exactly ones and
    ``weights * 1.0`` is the identity, so the arms differ in one number and in
    nothing else -- :func:`fast_row_weights`'s argument, for the same reason.

    **A draw, not a cap.** ``termination`` deliberately does not record which
    draw rule fired, so this weights every adjudicated draw and not only the
    engine's cap. The exact cap predicate *is* reconstructable per row
    (``ply + mlh_target / MLH_SCALE >= TURN_CAP_PLIES``, which is what
    :mod:`komugi.campaign.game_stats` counts), but it would couple the trainer
    to :data:`komugi.selfplay.records.MLH_SCALE` -- behind the same import cycle
    the leaf exists to avoid -- for a distinction that has been empty in three
    campaigns: stalemate, fourfold repetition and insufficient material have
    fired zero times, and the live tripwire is the journal's
    ``draws - turn_cap_games`` rather than an arithmetic the loss re-derives.

    The result is a **multiplier** for
    :meth:`komugi.train.losses.LossAssembler.__call__`'s ``head_sample_weight=``,
    applied to :data:`~komugi.train.losses.HEAD_WDL` **and to nothing else**
    (design review F10). ``draft_outcome`` is a copy of ``wdl_target`` on draft
    rows and the uncertainty target is the WDL residual; both are staged OFF
    today, and both would resolve their own ``composed.get(head, weights)`` --
    so each staged bring-up makes an explicit decision about cap rows rather
    than inheriting this one. The moves-left head is the clearest case: its
    target on a cap row is exactly the plies-to-cap it is meant to learn.

    The codes come from :mod:`komugi.terminations`, a leaf at the package root
    that imports nothing -- **not** from :mod:`komugi.selfplay.worker`, which
    re-exports them but cannot be imported from here without closing the cycle
    that module's own comment describes. Not restated as ``0``/``1`` either: a
    second source of truth for the schema's vocabulary is what the leaf exists
    to prevent.

    Args:
        termination: The batch's ``(batch,)`` integer ``termination`` column,
            spelled as :data:`komugi.selfplay.records.SCHEMA` writes it.
        weight: The arm. Any finite non-negative float; the assembler validates
            the composed vector. Note that ``0.0`` is reachable and makes an
            all-draw batch a zero-mass WDL head, which the assembler omits.

    Returns:
        A ``(batch,)`` float32 multiplier on the same device as `termination`.

    Raises:
        ValueError: If `termination` is not 1-D, or carries a value that is
            neither decisive nor draw. :data:`~komugi.terminations.TERMINATION_RESIGN`
            is refused rather than read as decisive: a resigned row's weight is
            a decision nobody has made, and taking it as ``1.0`` would train it
            at a weight nobody chose. A fourth code is a vocabulary widening
            (fingerprint-invisible), and this refusal is what makes it loud.
    """
    if termination.ndim != 1:
        raise ValueError(
            f"termination must be 1-D over the batch, got shape "
            f"{tuple(termination.shape)}"
        )
    # One host sync for both out-of-range predicates, exactly as
    # `fast_row_weights` does it and for the same budget reason.
    out_of_range = torch.stack(
        (
            (termination < TERMINATION_DECISIVE).any(),
            (termination > TERMINATION_DRAW).any(),
        )
    ).any()
    if bool(out_of_range.item()):
        raise ValueError(
            f"termination carries a value outside {{{TERMINATION_DECISIVE}, "
            f"{TERMINATION_DRAW}}} (decisive, draw); the cap-row value weight "
            f"has no arm for a resigned or unknown termination and must not "
            f"read it as a decisive row"
        )
    return torch.where(
        termination == TERMINATION_DRAW,
        torch.full((), float(weight), device=termination.device),
        torch.ones((), device=termination.device),
    )


def mlh_row_weights(
    ply: torch.Tensor,
    mlh_target: torch.Tensor,
    termination: torch.Tensor,
    cap_plies: int,
    weight: float,
) -> torch.Tensor:
    """Return the per-row **moves-left** multiplier that censors turn-cap rows.

    `weight` on a row whose game reached the engine's turn cap -- a draw
    (:data:`~komugi.terminations.TERMINATION_DRAW`) whose reconstructed length
    ``round(ply + mlh_target * cap_plies)`` is at least `cap_plies` -- and
    ``1.0`` on every other row. KOM-159 D4, and the whole of the moves-left
    head's treatment of the stalls campaign 5 ended in.

    **Why a cap row is not a label.** A capped game's row carries
    ``mlh_target = (cap_plies - ply) / cap_plies``: the plies until the
    *engine's* adjudication, which is what actually happened at the board
    (:data:`komugi.selfplay.records.MLH_SCALE`) and is a perfectly honest
    description of the environment. It is not an observation of the quantity
    the head is being trained to predict. "How far is the result" has, for a
    stall, the answer "at least ``cap_plies - ply``, unknown" -- a
    right-censored observation. Training the MSE on it teaches "a late position
    ends at the cap" and compresses exactly the sibling *differences* at plies
    100-300 that the search-side utility reads. So the pre-registered arm is
    ``0.0`` (drop the row from this head, keep it in every other) and ``1.0``
    is today's behaviour on the same code path, for the reason
    :func:`cap_row_weights` gives: the arms differ in one number and in nothing
    else.

    **Repetition draws keep weight 1.** ``termination`` records that a game was
    drawn, not which rule drew it, so the cap is reconstructed from the row's
    own arithmetic rather than assumed. A fourfold repetition at ply 300 is a
    real ending and its moves-left target is a real observation; only the rows
    whose total reaches the cap are censored.

    **The recorded alternative, not chosen.** A one-sided Tobit-style hinge on
    the censored rows -- ``relu(target - pred) ** 2``, "the game lasted *at
    least* this long" -- keeps the information a weight of 0 throws away. It is
    the arm to try if the censored head under-predicts on long decisive games
    (design review R7 / F7: ``mlh_bias_dec`` below -20 plies at the iteration-14
    read). It is not the launch arm because it is a second loss form on one
    head, and the bring-up's job is to measure the plain one first.

    **Why the head's loss stays MSE at all** (design review F7). The MLH loss's
    *share* of ``loss_total`` is ~0.05%, not the "< 5%" the design first said:
    the class-mean squared error on decisive rows is 0.0174 at iteration 14 and
    0.0316 at 30, against a total of 43-49. The share is not the argument. The
    **gradient** is: ``d/dpred (pred - target)**2 = 2 * (pred - target)`` is
    about 0.2-0.3 per sample, against ``p - y`` of about 0.3-0.7 per WDL logit,
    so at weight 1.0 the head pushes on the trunk within 2-3x of the value
    head -- which is the scale ``w_mlh = 1.0`` was chosen for. A Huber loss
    would cap the leverage of the rare very long rows and a log target would
    compress the tail; neither is taken, because the utility reads *differences
    between siblings* at 50-150 plies and a log target makes those differences
    depend on the root's own prediction, which is precisely the quantity the
    centred form subtracts out.

    Args:
        ply: The batch's ``(batch,)`` integer ``ply`` column.
        mlh_target: The batch's ``(batch,)`` float ``mlh_target`` column, on
            the recorded ``MLH_SCALE`` scale (``1.0`` is a full cap).
        termination: The batch's ``(batch,)`` integer ``termination`` column.
        cap_plies: The engine's turn cap the shards were played under, i.e.
            :meth:`komugi.engine.game.Game.max_turns`. Passed in as an ``int``
            rather than read from :mod:`komugi.selfplay.records`, which would
            couple the trainer to the self-play package behind the import cycle
            :func:`cap_row_weights` describes. Must be at least 1.
        weight: The arm. ``0.0`` censors (KOM-159 D4); ``1.0`` is an all-ones
            vector and therefore the pre-KOM-159 arithmetic.

    Returns:
        A ``(batch,)`` float32 multiplier on the same device as `ply`.

    Raises:
        ValueError: If any column is not 1-D, if the three columns disagree
            about the batch size, if `cap_plies` is below 1, or if
            `termination` carries a value that is neither decisive nor draw.
            :data:`~komugi.terminations.TERMINATION_RESIGN` is refused rather
            than read as decisive, exactly as :func:`cap_row_weights` refuses
            it: a resigned game's moves-left target has a definition nobody has
            written down (``records.MLH_SCALE``'s "until adjudication" is the
            open question), and taking it as an uncensored row would train it
            at a weight nobody chose.
    """
    for name, column in (
        ("ply", ply),
        ("mlh_target", mlh_target),
        ("termination", termination),
    ):
        if column.ndim != 1:
            raise ValueError(
                f"{name} must be 1-D over the batch, got shape "
                f"{tuple(column.shape)}"
            )
    if not ply.numel() == mlh_target.numel() == termination.numel():
        raise ValueError(
            f"ply, mlh_target and termination must describe the same batch, "
            f"got {ply.numel()}, {mlh_target.numel()} and "
            f"{termination.numel()} entries"
        )
    if cap_plies < 1:
        raise ValueError(f"cap_plies must be >= 1, got {cap_plies}")
    # One host sync for both out-of-range predicates, exactly as
    # `cap_row_weights` does it and for the same budget reason.
    out_of_range = torch.stack(
        (
            (termination < TERMINATION_DECISIVE).any(),
            (termination > TERMINATION_DRAW).any(),
        )
    ).any()
    if bool(out_of_range.item()):
        raise ValueError(
            f"termination carries a value outside {{{TERMINATION_DECISIVE}, "
            f"{TERMINATION_DRAW}}} (decisive, draw); the moves-left cap-row "
            f"weight has no arm for a resigned or unknown termination and must "
            f"not read it as an uncensored row"
        )
    # float64 for the reconstruction alone: `mlh_target` arrives as float32 and
    # `ply + mlh_target * cap_plies` has to land on the integer total the
    # recorder derived it from, so the `round` is exact rather than one ULP
    # short of the cap on the very rows this function exists to find.
    total = torch.round(
        ply.to(torch.float64) + mlh_target.to(torch.float64) * float(cap_plies)
    )
    censored = (termination == TERMINATION_DRAW) & (total >= float(cap_plies))
    return torch.where(
        censored,
        torch.full((), float(weight), device=ply.device),
        torch.ones((), device=ply.device),
    )


P0A_RECIPE: Final[LossConfig] = LossConfig(
    policy_enabled=True,
    policy_weight=1.0,
    soft_policy_enabled=True,
    soft_policy_weight=8.0,
    soft_policy_exponent=0.25,
    wdl_enabled=True,
    wdl_weight=1.0,
    mlh_enabled=False,
    mlh_weight=0.0,
    surprise_alpha=0.0,
    surprise_max_weight=5.0,
)
"""The loss configuration the first Komugi training run uses.

Not :class:`~komugi.train.losses.LossConfig`'s defaults, and deliberately not
reachable by omission: a run states this name or states its own config.

* **policy at 1.0** -- the objective, and the unit the other weights are read
  against.
* **soft policy ON at 8.0** -- design §2.3-2.4 puts the sharpened auxiliary
  target at 8x weight and calls it P0a *core*, not an extra. At Gungi's ~229
  branching the failure mode it prevents (the policy collapsing onto its own
  top move and search never seeing the alternatives again) is the one that
  ends a run quietly. Reaching ``SoftPolicyHead`` at all is what the rewire
  this module accompanies fixed; 8x is what makes it matter.
* **WDL at 1.0** -- three-way value, on from the first step.
* **moves-left OFF** -- staged (§2.4), and blocked on a decision this recipe
  does not get to make: MLH is plain MSE on whatever scale the targets arrive
  in, and while :data:`komugi.selfplay.records.MLH_SCALE` now pins that scale
  at ``1/256``, the head still lands behind its own matched-compute gate. Off
  rather than enabled-at-weight-0 because a disabled head costs nothing to
  report and P0a has no curve to compare it against yet.
* **surprise alpha 0.0** -- the weighting exists and is wired
  (:func:`surprise_weights_from_kl`), and is switched on as a deliberate,
  measured act rather than inherited. ``surprise_max_weight`` is carried at
  its default 5.0 so that switching alpha on is a one-field change.

Note which ``surprise_alpha`` is in force. :class:`~komugi.train.step.StepConfig`
carries its own, and it is the one that decides the weights actually applied:
above zero the trainer computes them here and passes ``sample_weight=``, which
bypasses the assembler's copy entirely. This one governs the assembler when it
is asked to weight a batch by itself -- an evaluation pass, a test -- and it is
0.0 for the same reason.

Superseded in production by :data:`P0B_RECIPE`, and **not edited** when that
landed: the P0a curves exist, and the constant that produced them must keep
producing them.
"""

P0B_RECIPE: Final[LossConfig] = LossConfig(
    policy_enabled=True,
    policy_weight=1.0,
    soft_policy_enabled=True,
    soft_policy_weight=8.0,
    soft_policy_exponent=0.25,
    wdl_enabled=True,
    wdl_weight=1.0,
    mlh_enabled=False,
    mlh_weight=0.0,
    uncertainty_enabled=False,
    uncertainty_weight=0.0,
    ownership_enabled=False,
    ownership_weight=0.0,
    short_term_value_enabled=False,
    short_term_value_weight=0.0,
    draft_outcome_enabled=False,
    draft_outcome_weight=0.0,
    surprise_alpha=0.0,
    surprise_max_weight=5.0,
)
"""The loss configuration the P0b production campaign uses.

:data:`P0A_RECIPE` field for field on all eleven P0a fields, plus the four
staged heads at their OFF defaults -- **stated, not inherited**, for the same
reason every other field here is stated: a run names this constant or names its
own config, and a weight that arrived by omission is a weight nobody chose.

A **new** constant rather than an edit to :data:`P0A_RECIPE`, for the reason
that constant's own docstring gives: *"nobody has to reconstruct months later
which weights a curve was produced under"*. The P0a curves exist; the name that
produced them keeps producing them.

Every staged head is off at weight 0 because the net freeze (KOM-005b D1) puts
all eight heads in the first P0b net while the *loss* stages: adding a head
after the campaign's first kept checkpoint would make every earlier checkpoint
unloadable, so the four parameter sets land now and each head's weight moves
only when its own pre-registered matched-compute gate says so, one head at a
time, in ``DESIGN.html`` §2.4's normative order. ``enabled=False`` rather than
``enabled=True, weight=0.0``: the enabled-at-zero state is each head's bring-up
*step*, taken deliberately when its ladder opens, and there is no curve to
compare against before then.
"""

MLH_RECIPE: Final[LossConfig] = LossConfig(
    policy_enabled=True,
    policy_weight=1.0,
    soft_policy_enabled=True,
    soft_policy_weight=8.0,
    soft_policy_exponent=0.25,
    wdl_enabled=True,
    wdl_weight=1.0,
    mlh_enabled=True,
    mlh_weight=1.0,
    uncertainty_enabled=False,
    uncertainty_weight=0.0,
    ownership_enabled=False,
    ownership_weight=0.0,
    short_term_value_enabled=False,
    short_term_value_weight=0.0,
    draft_outcome_enabled=False,
    draft_outcome_weight=0.0,
    surprise_alpha=0.0,
    surprise_max_weight=5.0,
)
"""The loss configuration campaign 6 uses: P0b, plus moves-left at weight 1.0.

:data:`P0B_RECIPE` field for field on all nineteen fields except two --
``mlh_enabled`` is ``True`` and ``mlh_weight`` is ``1.0`` -- and a **new**
constant rather than an edit, which is the rule :data:`P0A_RECIPE`'s docstring
states and :data:`P0B_RECIPE` follows: campaigns 1 through 5 ran under
``P0B_RECIPE``, and the name that produced those curves keeps producing them.
The weight-0 bring-up step the staging ladder mandates
(``docs/design/kom-005b-staged-heads.md``) is taken in **CI and at tiny scale**
for this head rather than as a campaign iteration: ``tests/train/
test_staged_losses.py`` pins that ``mlh_enabled=True, mlh_weight=0.0`` is
bit-identical to :data:`P0B_RECIPE`, which is the whole content of that step,
and campaign 6 then runs at 1.0 from iteration 0 (KOM-159 D1).

**Why 1.0** (design review F7). Not because the head is a small share of the
total -- it is ~0.05% of it, the class-mean squared error on decisive rows
being 0.0174 at iteration 14 against a ``loss_total`` of 43-49 -- but because
of the gradient it puts on the trunk: ``2 * (pred - target)`` is about 0.2-0.3
per sample against the WDL head's ``p - y`` of 0.3-0.7 per logit, i.e. the same
order. ``4.0`` is the single pre-registered fallback arm, taken only if the
head's curve falls (A1) while its calibration does not (A2).

**The loss form is unchanged** -- plain MSE on the recorded
:data:`komugi.selfplay.records.MLH_SCALE` scale, so no shard and no fingerprint
moves. See :func:`mlh_row_weights` for why MSE is kept over a Huber or a log
target, and for the censoring of turn-cap rows that this recipe is launched
with (KOM-159 D4): the recipe turns the head *on*, the row weight decides which
rows it learns from, and the two are separate knobs on purpose.
"""

MLH4_RECIPE: Final[LossConfig] = dataclasses.replace(MLH_RECIPE, mlh_weight=4.0)
"""The KOM-159 fallback arm: :data:`MLH_RECIPE` field for field with the
moves-left head at weight 4.0 (design §6, chosen 2026-09-17 after the weight-1.0
head measured near-constant: window MAE 90 plies, sibling rank accuracy 0.54).
Every other weight is :data:`MLH_RECIPE`'s.
"""
