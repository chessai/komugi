"""The four staged heads: construction, shapes, seeds and parameter cost.

KOM-005b's D1 is that all four staged heads -- uncertainty, final-control
ownership, short-term value, draft-outcome -- are **constructed and
forward-evaluated from the first P0b net**, with only their *losses* staged.
That decision is forced by the net freeze: ``load_state_dict`` is strict at
every consumer, so a head added after the production campaign's first kept
checkpoint makes every earlier checkpoint unloadable by the current code.

What that buys has to be paid for in three properties this module pins:

* **construction is still a pure function of the config**, and still leaves the
  global torch RNG exactly as it found it -- four more heads must not perturb
  the initialisation of the four that already exist;
* **the eight initialisation streams are distinct**, which is not free: two
  pairs of heads here are the *same class at the same shape*
  (uncertainty/moves-left, short-term-value/draft-outcome), so a salt mistake
  makes them bit-identical rather than merely similar;
* **the parameter cost is the number the design committed to**, so "under 1% of
  the policy readout" stays a measurement rather than an estimate.

The loss-side behaviour of these heads lives in
``tests/train/test_staged_losses.py``; their square-index correspondence in
``tests/net/heads_behavior/test_staged_head_wiring.py``.
"""

from __future__ import annotations

from typing import Final

import torch

from komugi.net import (
    DraftOutcomeHead,
    HeadsConfig,
    KomugiNet,
    MLHHead,
    NetConfig,
    OwnershipHead,
    ShortTermValueHead,
    TrunkConfig,
    UncertaintyHead,
)

_BATCH: Final[int] = 3
_CHANNELS: Final[int] = 16
_PRODUCTION_CHANNELS: Final[int] = 128

_HEADS: Final[HeadsConfig] = HeadsConfig(
    channels=_CHANNELS, num_actions=81, policy_planes=1, seed=17
)
_NET: Final[NetConfig] = NetConfig(
    trunk=TrunkConfig(
        in_planes=4, channels=_CHANNELS, blocks=2, gpool_every=3, seed=17
    ),
    heads=_HEADS,
)

_STAGED_PARAMETERS: Final[dict[str, int]] = {
    "uncertainty": 16_769,
    "ownership": 643,
    "short_term_value": 16_899,
    "draft_outcome": 16_899,
}
"""Parameter count per staged head at the production width ``channels=128``.

Derived by hand from the shapes, not read off the code under test:

* uncertainty -- ``BatchNorm2d(128)`` 256 + ``Linear(256, 64)`` 16,448 +
  ``Linear(64, 1)`` 65 = **16,769**
* ownership -- ``BatchNorm2d(128)`` 256 + ``Conv2d(128, 3, 1)`` with bias 387
  = **643**
* short-term value -- 256 + 16,448 + ``Linear(64, 3)`` 195 = **16,899**
* draft outcome -- the same shape as short-term value = **16,899**

BatchNorm's running statistics are buffers, not parameters, and so are
correctly absent from every one of those sums.
"""

_TOTAL_STAGED_PARAMETERS: Final[int] = 51_210


def _features() -> torch.Tensor:
    """A ``(N, C, 9, 9)`` feature stack from a LOCAL generator."""
    generator = torch.Generator().manual_seed(4242)
    return torch.randn(
        _BATCH, _CHANNELS, 9, 9, generator=generator, dtype=torch.float32
    )


def _production_heads() -> HeadsConfig:
    """The head shape at the production trunk width."""
    return HeadsConfig(channels=_PRODUCTION_CHANNELS, seed=0)


# --------------------------------------------------------------------------
# Shapes
# --------------------------------------------------------------------------


def test_the_uncertainty_head_emits_a_bare_scalar_per_sample() -> None:
    """``(N,)`` and not ``(N, 1)``: the squeeze is the moves-left contract.

    The loss subtracts this from an ``(N,)`` residual, so a trailing singleton
    would broadcast into an N-by-N matrix and train something that is not a
    per-sample error at all.
    """
    out = UncertaintyHead(_HEADS)(_features())

    assert out.shape == (_BATCH,)
    assert out.dim() == 1
    assert out.dtype == torch.float32
    assert bool(torch.isfinite(out).all())


def test_the_ownership_head_emits_one_three_way_readout_per_square() -> None:
    """``(N, 81, 3)``: squares on the middle axis, classes last.

    Classes last because the loss reduces over the class axis and the
    assembler's cross-entropy sums over the trailing one. ``(N, 3, 9, 9)``
    would be the same numbers under an axis order the loss cannot consume.
    """
    out = OwnershipHead(_HEADS)(_features())

    assert out.shape == (_BATCH, 81, 3)
    assert out.dtype == torch.float32
    assert bool(torch.isfinite(out).all())

    # Raw logits, not probabilities: the head does not pre-normalise, exactly
    # as every other head in this package refuses to.
    sums = out.softmax(dim=-1).sum(dim=-1)
    assert float((sums - 1.0).abs().max().item()) < 1e-5
    assert float((out.sum(dim=-1) - 1.0).abs().max().item()) > 1e-6


def test_the_short_term_value_and_draft_outcome_heads_emit_three_wide() -> None:
    """Both are ``(N, 3)`` -- and the 3 means different things in each.

    Short-term value's three are *horizons*; draft-outcome's three are the
    (win, draw, loss) classes ``wdl_logits`` uses. Same shape, so a mix-up
    between the two is shape-clean, which is why the wiring suite asserts
    tensor identity rather than equality.
    """
    features = _features()
    short_term = ShortTermValueHead(_HEADS)(features)
    draft = DraftOutcomeHead(_HEADS)(features)

    for out in (short_term, draft):
        assert out.shape == (_BATCH, 3)
        assert out.dtype == torch.float32
        assert bool(torch.isfinite(out).all())


# --------------------------------------------------------------------------
# Seeds
# --------------------------------------------------------------------------


def test_the_staged_heads_are_seeded_from_the_config_not_the_global_stream() -> None:
    """Two clauses, because the first alone passes on an empty diff.

    (a) ``torch.get_rng_state()`` is identical before and after building a
    whole net. Necessary, but it cannot distinguish "four staged heads,
    correctly written" from "no staged heads at all": both head bases already
    construct inside ``_undisturbed_global_rng``, so any conforming subclass
    inherits the guard and the restore hides everything.

    (b) is the clause that catches the real failure. Two nets built from the
    *same* config under two *different* ``torch.manual_seed`` values must have
    bit-identical ``state_dict``s. A head that constructs its ``nn.Linear`` or
    ``nn.Conv2d`` inside the guard and then forgets to overwrite the weights
    from ``_head_generator(config.seed, salt)`` draws from the caller's global
    stream -- and the guard's *restore* makes clause (a) pass anyway. The
    distinctness node below does not catch it either: two heads' discarded
    default draws differ, so "not equal" still holds.

    D1's whole checkpoint story rests on this property. A net whose parameters
    depend on the ambient seed is not reproducible from its config, and a
    campaign that cannot rebuild a net from its config cannot verify a
    checkpoint it loads.
    """
    torch.manual_seed(1234)
    before = torch.get_rng_state()
    first = KomugiNet(_NET)
    assert torch.equal(torch.get_rng_state(), before), (
        "constructing a net advanced the global RNG: some head is drawing "
        "from the ambient stream"
    )

    torch.manual_seed(99_999)
    second = KomugiNet(_NET)

    left = first.state_dict()
    right = second.state_dict()
    assert set(left) == set(right)
    differing = [
        name for name in left if not torch.equal(left[name], right[name])
    ]
    assert differing == [], (
        "these parameters changed with the ambient torch seed, so they are "
        f"not a pure function of config.seed: {differing}"
    )


def test_the_uncertainty_and_mlh_heads_do_not_share_a_parameter_draw() -> None:
    """Eight distinct init streams, checked where a salt slip is invisible.

    Uncertainty and moves-left are the *same class at the same shape*
    (``_PooledScalarHead`` at ``outputs=1``), and so are short-term value and
    draft-outcome (``outputs=3``). A head that reuses another's salt is
    therefore not "similar" -- it is bit-identical, parameter for parameter,
    and every shape assertion in this module still passes. The comparison of
    the generator-drawn tensors below is what makes that red.
    """
    for left, right, label in (
        (UncertaintyHead(_HEADS), MLHHead(_HEADS), "uncertainty vs mlh"),
        (
            ShortTermValueHead(_HEADS),
            DraftOutcomeHead(_HEADS),
            "short_term_value vs draft_outcome",
        ),
    ):
        first = left.state_dict()
        second = right.state_dict()
        assert set(first) == set(second), (
            f"{label}: the two heads are the same class, so their parameter "
            "names must match for this comparison to mean anything"
        )
        # Exactly the tensors the head's own generator draws, named rather
        # than filtered. ``_init_fixed_variance`` draws the two weights and
        # *zeroes* both biases; the norm's weight and bias are filled with
        # constants and its running statistics are buffers of accumulated
        # stats. All of those are identical across two correctly salted heads
        # by design, so a dtype/numel filter over the state dict makes this
        # node fail on a correct implementation rather than a broken one.
        drawn = ("pool_linear.weight", "out_linear.weight")
        for name in drawn:
            assert name in first, (
                f"{label}: {name} is not in the head's state dict -- the set "
                "of generator-drawn tensors has changed and this node is no "
                "longer comparing the draws it claims to"
            )
        for name in drawn:
            assert not torch.equal(first[name], second[name]), (
                f"{label}: {name} is bit-identical in both heads -- the two "
                "share an initialisation salt"
            )


def test_every_staged_head_draws_a_different_stream_from_every_other() -> None:
    """All four staged heads pairwise distinct, not just the confusable pairs.

    The pooled trio share a class, so their ``out_linear.weight`` shapes differ
    only in width; ownership is a convolution and is compared to itself under a
    different config seed instead.
    """
    net = KomugiNet(_NET)
    pooled = {
        "uncertainty": net.uncertainty.pool_linear.weight,
        "short_term_value": net.short_term_value.pool_linear.weight,
        "draft_outcome": net.draft_outcome.pool_linear.weight,
        "mlh": net.mlh.pool_linear.weight,
        "wdl": net.wdl.pool_linear.weight,
    }
    names = sorted(pooled)
    for index, name in enumerate(names):
        for other in names[index + 1 :]:
            assert not torch.equal(pooled[name], pooled[other]), (
                f"{name} and {other} share a pooling-linear draw"
            )

    other_seed = NetConfig(
        trunk=_NET.trunk,
        heads=HeadsConfig(
            channels=_CHANNELS, num_actions=81, policy_planes=1, seed=18
        ),
    )
    assert not torch.equal(
        net.ownership.conv.weight, KomugiNet(other_seed).ownership.conv.weight
    ), "the ownership head ignores config.seed"


# --------------------------------------------------------------------------
# What OFF costs
# --------------------------------------------------------------------------


def test_the_staged_heads_add_the_documented_parameter_count() -> None:
    """51,210 parameters at ``channels=128``, and the per-head split.

    The design's "under 1% of the policy readout's arithmetic" argument is
    what licenses landing four heads at once and leaving them off, so the
    number it rests on is asserted rather than assumed. Per head as well as in
    total, because a compensating error in two heads would pass a total-only
    check.
    """
    config = _production_heads()
    heads = {
        "uncertainty": UncertaintyHead(config),
        "ownership": OwnershipHead(config),
        "short_term_value": ShortTermValueHead(config),
        "draft_outcome": DraftOutcomeHead(config),
    }

    counts = {
        name: sum(parameter.numel() for parameter in head.parameters())
        for name, head in heads.items()
    }
    assert counts == _STAGED_PARAMETERS
    assert sum(counts.values()) == _TOTAL_STAGED_PARAMETERS


def test_the_staged_heads_are_a_small_fraction_of_one_policy_head() -> None:
    """The comparison the parameter budget is actually argued against.

    One policy head at ``channels=128`` is ``BatchNorm2d`` 256 +
    ``Conv2d(128, 461, 1)`` 59,469 = 59,725 parameters, so all four staged
    heads together are under a single policy readout. Stated as a strict
    inequality rather than a second literal: the claim is the ordering, and
    pinning the policy head's own count here would duplicate a number that
    module already owns.
    """
    config = _production_heads()
    policy_parameters = sum(
        parameter.numel()
        for parameter in KomugiNet(
            NetConfig(
                trunk=TrunkConfig(
                    in_planes=4,
                    channels=_PRODUCTION_CHANNELS,
                    blocks=1,
                    gpool_every=3,
                    seed=0,
                ),
                heads=config,
            )
        ).policy.parameters()
    )
    assert _TOTAL_STAGED_PARAMETERS < policy_parameters
