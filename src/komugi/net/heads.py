"""The eight heads: the P0a core, plus the four staged heads landed OFF.

This module is the readout half of the network. Each head takes the trunk's
``(N, channels, 9, 9)`` feature stack and turns it into one prediction; none
of them knows how those features were computed, which is the seam that keeps a
transformer trunk a drop-in replacement (see :mod:`komugi.net.trunk`).

.. _StagingRule:

Why the staged heads land now, and land off
-------------------------------------------
``docs/DESIGN.html`` §2.4 is explicit that the staged heads -- uncertainty,
final-control ownership, short-term value, draft-outcome -- are *not* P0a:
each one arrives behind its own matched-compute ladder-Elo gate, because a
single confounded gate cannot attribute blame across eight head groups. What
the P0a core is remains exactly what it was: "WDL + moves-left + soft-policy
only", plus the main policy head those three accompany.

**But staging is a loss-config act, never an architecture act** (KOM-005b D1).
:class:`~komugi.net.model.KomugiNet` constructs its heads unconditionally and
its submodule names are part of the ``state_dict``, and every consumer loads
strictly (``komugi.train.ema``: "``load_state_dict`` is strict at every
consumer"). A head added *after* the production campaign's first kept
checkpoint therefore makes every earlier checkpoint unloadable by the current
code. Landing the four staged heads one at a time, as their gates pass, would
invalidate every checkpoint of the campaign four times.

So all eight parameter sets exist from the first P0b net, all four staged
*losses* land at ``enabled=False, weight=0.0``
(:data:`komugi.train.wiring.P0B_RECIPE`), and what stages is the weight. This
paragraph is the one place a reader learns that rule; it must not be deleted,
only updated. The cost of OFF is one forward pass through 51,210 extra
parameters at ``channels=128`` and nothing else: construction does not touch
the global RNG (see `Device and RNG`_), each head's BatchNorm statistics live
inside that head, and a disabled head receives no gradient and therefore no
optimiser state. That OFF is *bit-identical* training, not merely close, is
pinned by execution in ``tests/train/behavior/test_golden_step.py`` and
``tests/train/behavior/test_staged_off_identity.py``.

Every head emits **raw** numbers: logits, not probabilities, and a raw
moves-left scalar. Nothing here softmaxes, masks illegal actions, or scales a
prediction into a utility. Those are the trainer's and the search's jobs
respectively, and doing them here would only mean undoing them there.

.. _PolicyOrdering:

The policy plane ordering -- the load-bearing part
--------------------------------------------------
The policy head is a 1x1 convolution ``channels -> 461`` producing
``(N, 461, 9, 9)``, flattened to ``(N, 37341)``. The flatten is **plane-major**
and its result is not merely *a* labelling of the action space: flat index
``i`` **is** the engine's action id ``i``, exactly, with no permutation table
anywhere. That identity is what lets the search read
:func:`komugi.engine.action_encoding.legal_cython_action_ids` and index the
logit vector with it directly.

It holds because two independent orderings agree, and both were checked
against the engine rather than assumed:

**Plane-major over squares.** Every block of the action encoding is laid out
as ``block_start + plane * 81 + to_sq`` (see
:mod:`komugi.engine.action_encoding`), and the block starts are the running
sums of the per-block plane counts, so the whole space is one flat
``plane * 81 + sq`` with 461 planes::

    A0  planes   0-80    from_sq                  (board move, no stack)
    A1  planes  81-404   81 + from_sq * 4 + sel   (board move, stack)
    B   planes 405-432   405 + piece_idx * 2 + bsel  (arata)
    C   planes 433-460   433 + piece_idx * 2 + bsel  (arata + draft-finish)

A ``(N, 461, 9, 9)`` tensor flattened row-major is ``plane * 81 + <spatial>``,
which is the same shape of expression -- provided ``<spatial>`` is ``sq``.

**Row-major over the board.** It is. The engine indexes squares over
``komugi.engine.utils.SQUARES``, whose order is ``rank`` 1..9 outer and
``file`` 9..1 inner, so

    ``sq = (rank - 1) * 9 + (9 - file)``

-- which is precisely ``action_encoding._RF_TO_SQ = r * 9 + (8 - f)`` for
0-based ``r``, ``f``. The observation encoder writes the same square to the
same place: ``observation.py`` fills ``obs[plane, r, j]`` with ``j = 8 - f``.
So the tensor's spatial position ``(row, col)`` is ``(rank - 1, 9 - file)``,
its row-major flat index ``row * 9 + col`` is ``sq``, and square index 0 is
the top-left cell -- rank 1, file 9.

Worked example, the one the unit suite asserts mechanically::

    action id 19553 = 241 * 81 + 32

    plane 241 -> A1 (81 <= 241 < 405), block-local plane 241 - 81 = 160
              -> 160 = from_sq 40 * 4 + sel 0   (a plain tsuke, converting
                                                 nothing)
    from_sq 40 -> row 4, col 4 -> rank 5, file 5 -> square "5-5"
    square 32 -> row 3, col 5 -> rank 4, file 4 -> square "4-4"

    so logits[:, 19553] is conv_output[:, 241, 3, 5]: "stack the piece on
    5-5 onto the tower at 4-4, converting nothing".

And at a block boundary, ``id 6561`` -- the first A1 id -- is plane 81,
square 0, i.e. ``conv_output[:, 81, 0, 0]``.

Get this wrong and nothing crashes: every policy target is silently permuted
and the net learns a scrambled game. Hence the worked example above is pinned
by :mod:`tests.net.unit.test_heads_unit` through
:func:`_flatten_action_planes` itself, not through a re-derivation of it.

Head architectures
------------------
The two policy heads are identical in shape and independent in parameters::

    h = Mish(BatchNorm2d(x))            # (N, C, 9, 9)
    h = Conv1x1(C -> 461)(h)            # (N, 461, 9, 9), bias
    out = h.reshape(N, -1)              # (N, 37341)

The two scalar heads share a shape too, differing only in output width::

    h = Mish(BatchNorm2d(x))            # (N, C, 9, 9)
    h = cat(mean(h), max(h)) over the board          # (N, 2C)
    h = Mish(Linear(2C -> C // 2)(h))                # (N, C // 2)
    out = Linear(C // 2 -> 3 or 1)(h)

The pooling is the same mean-and-max pair the trunk's ``_GlobalPoolingBias``
uses, and for the same reason: the mean says how much of a feature is on the
board at all, the maximum says whether there is one square where it is
extreme, and a value judgement needs both. Unlike the trunk's, this pooling is
not a bias added back to the squares -- it *is* the head's entire view, since
a game result is a property of the position and not of any one square.

Initialisation
--------------
The conventions are the trunk's, imported from it rather than restated, with
one deliberate departure at the output layers.

The one interior layer -- each scalar head's ``2C -> C // 2`` pooling linear
-- uses the trunk's fixed-variance draw, ``Normal(0, gain / sqrt(fan_in))``.
Its input is a pooled statistic of a Mish activation rather than a
unit-second-moment vector, so its gain corrects for both:
``1 / sqrt(_MISH_SECOND_MOMENT * _POOLED_SECOND_MOMENT)``, about 0.94.

Output layers -- the policy convolution and the two final linears -- use
:data:`_READOUT_GAIN`, a deliberately small 0.2, with **zero bias**. This is
not variance preservation and is not trying to be. A policy conv initialised
for unit output variance would hand a freshly built net a confident opinion
about 37341 actions that it then has to spend gradient steps unlearning; at
gain 0.2 the initial logit spread is around ``0.2 * sqrt(_MISH_SECOND_MOMENT)``,
roughly 0.13, so the initial policy is near-uniform and the first updates go
into learning rather than into demolition. The same argument gives WDL a
near-uniform initial (win, draw, loss) and moves-left an initial prediction of
approximately zero -- from which, being a raw regression, it learns its own
scale.

Device and RNG
--------------
As in the trunk: nothing here names a device, and every parameter is drawn
from a local :class:`torch.Generator` inside
:func:`komugi.net.trunk._undisturbed_global_rng`, so constructing a head is
both reproducible on its own and invisible to whatever seeding the
surrounding process is doing.

The eight heads of one :class:`HeadsConfig` must not be *identical* as well as
reproducible -- to each other, or to the trunk, which consumes ``config.seed``
raw -- so each derives its own stream: a head with salt ``k`` (salts run 1..8)
seeds its generator with ``config.seed + k * _SEED_STRIDE``. See
:data:`_SEED_STRIDE`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

# The trunk's initialisation machinery, imported rather than copied. These
# names are private to the *package*, not to the trunk module: the heads are
# initialised on exactly the same terms as the trunk, and a second copy of
# "the gain that undoes Mish's attenuation" would be a second thing to keep in
# step with the first.
from komugi.net.trunk import (
    _MISH_SECOND_MOMENT,
    _POOLED_SECOND_MOMENT,
    _fill_parameter,
    _init_fixed_variance,
    _undisturbed_global_rng,
)

# Squares on a Gungi board, and hence the stride between consecutive policy
# planes in the flat action space. Not a configuration knob: the action
# encoding's "block_start + plane * 81 + to_sq" is 81 by construction.
_BOARD_SQUARES = 81

# Gain for the three output layers -- the policy convolution and the final
# linear of each scalar head. Small on purpose; see `Initialisation`_ in the
# module docstring.
_READOUT_GAIN = 0.2

# Second moment of the (mean, max) pooled vector that the scalar heads' first
# linear reads: the trunk's pooled-vs-features ratio, applied to a Mish
# activation rather than to a unit-second-moment feature.
_POOLED_INPUT_SECOND_MOMENT = _MISH_SECOND_MOMENT * _POOLED_SECOND_MOMENT

_POOL_LINEAR_GAIN = 1.0 / math.sqrt(_POOLED_INPUT_SECOND_MOMENT)

# Spacing between the eight heads' initialisation streams. A head with salt k
# under a config seeded s draws from seed `s + k * _SEED_STRIDE`, so the eight
# heads of one config never share parameters, and two configs collide only if
# their seeds differ by an exact multiple of a million -- a number chosen to
# be far above any seed a sweep would plausibly enumerate. Salts start at 1,
# not 0: the trunk consumes `config.seed` raw, so a salt of 0 would hand the
# policy conv the same standard-normal prefix as the trunk's stem whenever
# the two configs share a seed (as they do by default).
_SEED_STRIDE = 1_000_003

_POLICY_SALT = 1
_SOFT_POLICY_SALT = 2
_WDL_SALT = 3
_MLH_SALT = 4
# Salts 5..8 are the staged heads (KOM-005b §1.0). They extend the run rather
# than reusing a P0a salt, so the four heads that already exist keep exactly
# the parameter draw they had -- which is what makes OFF bit-identical.
_UNCERTAINTY_SALT = 5
_OWNERSHIP_SALT = 6
_SHORT_TERM_VALUE_SALT = 7
_DRAFT_OUTCOME_SALT = 8

_OWNERSHIP_CLASSES = 3
"""Per-square control classes: ``0`` the row's mover, ``1`` the opponent, ``2``
empty -- the class order of :data:`komugi.selfplay.records.SCHEMA`'s
``final_control`` column, which is what this head is scored against. (The
column's fourth value, ``3`` = unlabelled, is a *loss* concept: it is masked to
an all-zero target row and is never a class this head predicts.)"""


@dataclass(frozen=True)
class HeadsConfig:
    """The shape of the four P0a core heads.

    Args:
        channels: Trunk width ``C`` the heads read. Must match the trunk's own
            ``channels`` -- :class:`~komugi.net.model.NetConfig` is where that
            agreement is enforced -- and must be even, because the scalar
            heads work at ``C // 2``.
        num_actions: Size of the flat action space. Defaults to the engine's
            ``ACTION_SIZE``; no default is *read* from the engine, so that a
            change over there surfaces as a loud validation failure here
            rather than as a silently reshaped head.
        policy_planes: Number of 1x1-convolution output planes. Must satisfy
            ``num_actions == policy_planes * 81``: the two are one number
            expressed twice, and the constructor refuses any pair that is not.
        seed: Seed for parameter initialisation. Construction is a pure
            function of the config -- equal configs give bit-identical
            ``state_dict``s -- and neither reads nor advances the global torch
            RNG.
    """

    channels: int
    num_actions: int = 37341
    policy_planes: int = 461
    seed: int = 0

    def __post_init__(self) -> None:
        """Reject configurations that cannot describe the heads."""
        if self.channels < 1:
            raise ValueError(f"channels must be >= 1, got {self.channels!r}")
        if self.channels % 2 != 0:
            raise ValueError(
                f"channels must be even -- the scalar heads work at "
                f"channels // 2 -- got {self.channels!r}"
            )
        if self.policy_planes < 1:
            raise ValueError(
                f"policy_planes must be >= 1, got {self.policy_planes!r}"
            )
        if self.num_actions != self.policy_planes * _BOARD_SQUARES:
            raise ValueError(
                f"num_actions must be policy_planes * {_BOARD_SQUARES} -- the "
                f"policy head is a plane-major readout over board squares -- "
                f"got num_actions={self.num_actions!r} with "
                f"policy_planes={self.policy_planes!r} "
                f"(expected {self.policy_planes * _BOARD_SQUARES})"
            )
        if self.seed < 0:
            raise ValueError(f"seed must be >= 0, got {self.seed!r}")

    @property
    def value_hidden(self) -> int:
        """Hidden width ``C // 2`` of the WDL and moves-left heads."""
        return self.channels // 2


def _head_generator(seed: int, salt: int) -> torch.Generator:
    """Return the local generator for the `salt`-th head of a `seed`-ed config.

    The salts are small integers; the stride between streams is
    :data:`_SEED_STRIDE`.
    """
    generator = torch.Generator()
    generator.manual_seed(seed + salt * _SEED_STRIDE)
    return generator


def _flatten_action_planes(planes: torch.Tensor) -> torch.Tensor:
    """Flatten ``(N, P, 9, 9)`` plane logits into ``(N, P * 81)`` action logits.

    **This one line is the policy plane ordering**, and every claim in
    `PolicyOrdering`_ is a claim about it: ``reshape`` walks the trailing
    dimensions fastest, so the result is ``plane * 81 + row * 9 + col``, which
    is ``plane * 81 + sq`` and hence the engine's action id. There is
    deliberately no lookup table to get out of step with the engine, and no
    second implementation of this expression -- the tests call this function.
    """
    return planes.reshape(planes.shape[0], -1)


def _pool_board(features: torch.Tensor) -> torch.Tensor:
    """Reduce ``(N, C, H, W)`` to the ``(N, 2C)`` (mean, max) board summary.

    Channel-wise mean over the board concatenated with channel-wise maximum,
    in that order -- the same statistics, computed the same way, as the
    trunk's :class:`~komugi.net.trunk._GlobalPoolingBias`.
    """
    flat = features.flatten(2)
    return torch.cat((flat.mean(dim=2), flat.amax(dim=2)), dim=1)


class _PlaneConvHead(nn.Module):
    """Shared implementation of the two policy heads.

    ``BatchNorm2d -> Mish -> Conv1x1(C -> policy_planes) -> flatten``. The
    convolution carries a bias (zeroed at init): it is the output layer, so
    there is no normalisation after it to absorb one, and a per-action offset
    is exactly the kind of freedom a policy wants.

    Subclasses supply nothing but their initialisation stream. They are
    separate classes rather than one class used twice because the network
    holds one of each and their parameters must never be confused -- see
    :class:`PolicyHead` and :class:`SoftPolicyHead`.

    Args:
        config: The head shape. Stored unchanged as :attr:`config`.
        salt: Which initialisation stream of `config`'s seed to draw from.
    """

    config: HeadsConfig
    norm: nn.BatchNorm2d
    act: nn.Mish
    conv: nn.Conv2d

    def __init__(self, config: HeadsConfig, *, salt: int) -> None:
        super().__init__()
        self.config = config

        with _undisturbed_global_rng():
            generator = _head_generator(config.seed, salt)

            self.norm = nn.BatchNorm2d(config.channels)
            self.act = nn.Mish()
            self.conv = nn.Conv2d(
                config.channels, config.policy_planes, kernel_size=1, bias=True
            )

            _init_fixed_variance(
                self.conv, gain=_READOUT_GAIN, generator=generator
            )
            _fill_parameter(self.norm, "weight", 1.0)
            _fill_parameter(self.norm, "bias", 0.0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return raw ``(N, num_actions)`` logits for ``(N, C, 9, 9)`` features.

        Raw: unmasked and un-softmaxed. Masking illegal actions is the
        search's job and it needs the logits to do it with.
        """
        h: torch.Tensor = self.act(self.norm(features))
        return _flatten_action_planes(self.conv(h))


class PolicyHead(_PlaneConvHead):
    """The main policy head: one logit per action id.

    ``forward`` maps ``(N, channels, 9, 9)`` to ``(N, num_actions)`` raw
    logits, whose index *is* the engine's action id -- see `PolicyOrdering`_
    in the module docstring for why, and for the worked example.

    Args:
        config: The head shape.
    """

    def __init__(self, config: HeadsConfig) -> None:
        super().__init__(config, salt=_POLICY_SALT)


class SoftPolicyHead(_PlaneConvHead):
    """KataGo's auxiliary soft-policy head: same shape, own parameters.

    Identical in construction and in output contract to :class:`PolicyHead`,
    and sharing no parameter with it -- a different initialisation stream, and
    a separate module in the network. The point of the head (§2.3) is that it
    is trained against a *sharpened* target, ``target ** 0.25`` renormalised,
    at 8x weight: that target keeps mass on the moves search examined but did
    not choose, which is what stops a policy collapsing onto its own top move
    at Gungi's branching factor of ~229.

    Note for whoever wires the trainer up: ``komugi.train.losses`` currently
    computes its ``soft_policy`` term from the ``policy_logits`` key, i.e.
    from the *main* head's output. With this head in the network that key
    should be pointed at :attr:`~komugi.net.model.NetOutput.soft_policy_logits`
    instead, or the auxiliary target trains the main head and this head
    receives no gradient at all.

    Args:
        config: The head shape.
    """

    def __init__(self, config: HeadsConfig) -> None:
        super().__init__(config, salt=_SOFT_POLICY_SALT)


class _PooledScalarHead(nn.Module):
    """Shared implementation of the WDL and moves-left heads.

    ``BatchNorm2d -> Mish -> (mean, max) board pooling -> Linear(2C -> C // 2)
    -> Mish -> Linear(C // 2 -> outputs)``.

    Args:
        config: The head shape.
        outputs: Width of the final linear.
        salt: Which initialisation stream of `config`'s seed to draw from.
    """

    config: HeadsConfig
    norm: nn.BatchNorm2d
    act: nn.Mish
    pool_linear: nn.Linear
    out_linear: nn.Linear

    def __init__(self, config: HeadsConfig, *, outputs: int, salt: int) -> None:
        super().__init__()
        self.config = config

        with _undisturbed_global_rng():
            generator = _head_generator(config.seed, salt)

            self.norm = nn.BatchNorm2d(config.channels)
            # Mish is stateless, so one instance serves both activation sites.
            self.act = nn.Mish()
            self.pool_linear = nn.Linear(
                2 * config.channels, config.value_hidden
            )
            self.out_linear = nn.Linear(config.value_hidden, outputs)

            _init_fixed_variance(
                self.pool_linear, gain=_POOL_LINEAR_GAIN, generator=generator
            )
            _init_fixed_variance(
                self.out_linear, gain=_READOUT_GAIN, generator=generator
            )
            _fill_parameter(self.norm, "weight", 1.0)
            _fill_parameter(self.norm, "bias", 0.0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return the ``(N, outputs)`` readout for ``(N, C, 9, 9)`` features."""
        h: torch.Tensor = self.act(self.norm(features))
        h = self.act(self.pool_linear(_pool_board(h)))
        out: torch.Tensor = self.out_linear(h)
        return out


class WDLHead(_PooledScalarHead):
    """Three-way value: win, draw and loss logits for the side to move.

    ``forward`` maps ``(N, channels, 9, 9)`` to ``(N, 3)`` raw logits in the
    order **(win, draw, loss)** -- index 0 win, index 1 draw, index 2 loss,
    matching the three-way cross-entropy in ``komugi.train.losses``. Raw: the
    softmax belongs to whoever consumes them, and the loss wants the logits.

    Draw mass is a first-class output rather than a byproduct of a scalar
    value (§2.4): contempt search needs to know *how much* draw there is in a
    position to decide whether to fight, and a single scalar in [-1, 1] cannot
    distinguish "dead drawn" from "sharp and balanced".

    The perspective is the side to move, which is the perspective the
    observation encoding already takes (``observation.py`` plane 114).

    Args:
        config: The head shape.
    """

    def __init__(self, config: HeadsConfig) -> None:
        super().__init__(config, outputs=3, salt=_WDL_SALT)


class MLHHead(_PooledScalarHead):
    """Moves-left: how many more plies this game has in it.

    ``forward`` maps ``(N, channels, 9, 9)`` to ``(N,)`` -- squeezed, so a
    prediction is a scalar per sample and not a length-1 vector.

    **The output is on the raw moves-left scale.** Nothing here normalises,
    clamps or exponentiates it, and the head is initialised to predict
    approximately zero and to learn its own scale from the targets. That is a
    contract the trainer has to meet from its side: ``komugi.train.losses``
    regresses this head with plain MSE on whatever scale the targets arrive
    in, so an unscaled target of "83 plies remaining" produces a head loss
    three orders of magnitude above the cross-entropies it is summed with at
    weight 1. See ``docs/reviews/2026-08-22-kom-020-trainer-review.md``, F12
    ("MLH is MSE on the raw moves-left scale ... the trainer must scale
    targets upstream or accept a head loss that dwarfs the others"). The scale
    is fixed **upstream**, in the trainer, and deliberately not here, so that
    a single place decides it.

    One further obligation, from §2.4 and not settled by this module: the
    moves-left target is truncation-affected, so what "moves left" means on a
    resigned or adjudicated game must be defined before this head is trained
    on one. Unlike Go, our games will not run to a natural end position.

    Args:
        config: The head shape.
    """

    def __init__(self, config: HeadsConfig) -> None:
        super().__init__(config, outputs=1, salt=_MLH_SALT)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return the raw ``(N,)`` moves-left prediction."""
        return super().forward(features).squeeze(-1)


class UncertaintyHead(_PooledScalarHead):
    """Staged head 1: how wrong the value head is about to be.

    ``forward`` maps ``(N, channels, 9, 9)`` to ``(N,)`` -- squeezed, exactly
    as :class:`MLHHead` is, and for the same reason: a prediction is a scalar
    per sample and not a length-1 vector.

    **The output is raw**, and it is deliberately *not* the predicted error:
    :mod:`komugi.train.losses` applies ``softplus`` before scoring it, because
    a predicted squared error must be non-negative and a head initialised to
    predict approximately zero then starts at ``softplus(0) = 0.693``, inside
    the target's ``[0, 4]`` range rather than on its boundary. An ``exp`` was
    rejected: it makes the gradient of a badly-initialised head unbounded, and
    this head is switched on mid-run on a trunk that has never seen its loss.

    The target is the WDL head's own squared residual, ``(v̂ - z)**2`` with
    ``v̂ = p[0] - p[2]`` and ``z = wdl_target[0] - wdl_target[2]``, **detached**
    (KOM-005b §1.1.2). The detach is the invariant: without it the WDL head
    acquires a second objective -- *be predictable* -- which it can serve by
    being confidently wrong in a consistent direction.

    ``DESIGN.html`` §2.4 carries the correction that bounds what this head may
    be used for: it predicts short-term difficulty on distribution and
    structurally *cannot* flag out-of-distribution positions, because it is
    trained on the same distribution that starves the value head there.

    Args:
        config: The head shape.
    """

    def __init__(self, config: HeadsConfig) -> None:
        super().__init__(config, outputs=1, salt=_UNCERTAINTY_SALT)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return the raw ``(N,)`` uncertainty readout (pre-``softplus``)."""
        return super().forward(features).squeeze(-1)


class OwnershipHead(nn.Module):
    """Staged head 2: who controls each square when the game ends.

    ``BatchNorm2d -> Mish -> Conv1x1(C -> 3)``, then a reshape to
    ``(N, 81, 3)``. It is the :class:`_PlaneConvHead` prefix without the
    action flatten, written out here rather than hoisted behind a "flatten or
    not" flag: that flag would put the policy plane ordering -- the one thing
    in this module that must never be conditional -- behind a boolean.

    ``forward`` maps ``(N, channels, 9, 9)`` to ``(N, 81, 3)`` raw logits.
    Class order is the ``final_control`` column's own: **0 = the row's mover
    controls the square, 1 = the opponent does, 2 = empty** (see
    :data:`komugi.selfplay.records.SCHEMA`).

    **The square ordering is the policy head's, and that is not a
    coincidence.** ``reshape`` on the trailing two axes gives ``row * 9 + col``,
    which `PolicyOrdering`_ establishes *is* the engine's ``sq_index``
    ``s = (rank - 1) * 9 + (9 - file)``; ``final_control`` is indexed
    ``s = r * 9 + (8 - f)`` "like a flattened observation plane". Same
    expression, so no permutation table exists anywhere and none is needed.
    Get it wrong and nothing crashes -- the head learns a transposed board --
    which is why the correspondence is pinned square by square, all 81 of
    them, rather than by shape.

    ``(N, 81, 3)`` rather than ``(N, 3, 9, 9)`` because the loss reduces over
    the **class** axis and the assembler's cross-entropy sums over the last
    one. The transpose is one view-level op per batch and it keeps the head's
    output shaped like what consumes it.

    Args:
        config: The head shape. Stored unchanged as :attr:`config`.
    """

    config: HeadsConfig
    norm: nn.BatchNorm2d
    act: nn.Mish
    conv: nn.Conv2d

    def __init__(self, config: HeadsConfig) -> None:
        super().__init__()
        self.config = config

        with _undisturbed_global_rng():
            generator = _head_generator(config.seed, _OWNERSHIP_SALT)

            self.norm = nn.BatchNorm2d(config.channels)
            self.act = nn.Mish()
            self.conv = nn.Conv2d(
                config.channels, _OWNERSHIP_CLASSES, kernel_size=1, bias=True
            )

            _init_fixed_variance(
                self.conv, gain=_READOUT_GAIN, generator=generator
            )
            _fill_parameter(self.norm, "weight", 1.0)
            _fill_parameter(self.norm, "bias", 0.0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return raw ``(N, 81, 3)`` per-square control logits."""
        h: torch.Tensor = self.act(self.norm(features))
        planes: torch.Tensor = self.conv(h)
        flat = planes.reshape(
            planes.shape[0], _OWNERSHIP_CLASSES, _BOARD_SQUARES
        )
        return flat.transpose(1, 2)


class ShortTermValueHead(_PooledScalarHead):
    """Staged head 3: the value ``h`` plies from now, at three horizons.

    ``forward`` maps ``(N, channels, 9, 9)`` to ``(N, 3)`` raw values. **The
    three outputs are horizons, not classes** -- this head shares
    :class:`WDLHead`'s shape and shares nothing else -- and their order is
    :data:`komugi.selfplay.records.SHORT_TERM_HORIZONS`, ``(4, 16, 64)`` plies.
    The width is 3 because that tuple is 3: a horizon count is a property of
    the data generation, so :class:`HeadsConfig` gains no field for it.

    The target is data, not another head's output, so -- unlike
    :class:`UncertaintyHead` -- there is **no detach anywhere**. The two heads
    must not be implemented by copying one onto the other.

    Args:
        config: The head shape.
    """

    def __init__(self, config: HeadsConfig) -> None:
        super().__init__(config, outputs=3, salt=_SHORT_TERM_VALUE_SALT)


class DraftOutcomeHead(_PooledScalarHead):
    """Staged head 4: the game's result, read out of a draft position.

    ``forward`` maps ``(N, channels, 9, 9)`` to ``(N, 3)`` raw logits in the
    **same (win, draw, loss) order** as :class:`WDLHead`, because it is scored
    against the same kind of target and a second convention would be a bug
    waiting for a reader.

    What makes it a different head from WDL, given the same shape and the same
    label, is that it is trained *only on draft-phase rows*: a non-draft row
    carries an all-zero target and so contributes exactly zero, with no mask
    and no per-head sample weight (KOM-005b D7). ``DESIGN.html`` §2.4 states
    both its purpose and its limit -- "predicts the net's own evolving habits
    -- a non-stationary target: its purpose is representation shaping, its loss
    is excluded from every gate".

    Args:
        config: The head shape.
    """

    def __init__(self, config: HeadsConfig) -> None:
        super().__init__(config, outputs=3, salt=_DRAFT_OUTCOME_SALT)
