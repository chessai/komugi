"""The staged heads' wiring: the ownership square index, and field->module.

Two claims live here, both of them about *which tensor is which* rather than
about any tensor's value.

The ownership head's square index **is** the engine's square index.

This is the one staged head whose failure mode is silent. Get the square
ordering wrong and nothing crashes, no shape mismatches, no gradient
disappears: the head simply learns a transposed board, against a
``final_control`` column that is indexed the other way, and the resulting loss
curve falls exactly as convincingly as a correct one would.

The claim being pinned is an *identity*, not a convention:

``reshape`` on the trailing two axes of a ``(N, 3, 9, 9)`` convolution output
gives ``row * 9 + col``. ``komugi.net.heads``' policy-ordering section
establishes that ``row * 9 + col`` **is** the engine's ``sq_index``,
``s = (rank - 1) * 9 + (9 - file)``; ``komugi.selfplay.records`` states that
``final_control`` is indexed ``s = r * 9 + (8 - f)`` "like a flattened
observation plane". Same expression, so no permutation table exists anywhere
and none is needed -- and that is precisely why it must be asserted square by
square rather than by shape.

The technique is the policy suite's: substitute the head's *input* through a
``forward_pre_hook`` so the whole net is exercised (the head really is the one
wired to ``NetOutput.ownership_logits``) while the features reaching the head
are the test's own, one driven spatial cell at a time.

The second claim is the **field-to-submodule binding** (KOM-005b impl review
F1). ``KomugiNet.forward`` assigns eight keyword arguments from eight
submodule calls, and two of those pairs are shape-identical:
``short_term_value`` / ``draft_outcome_logits`` are both ``(N, 3)``, and
``uncertainty`` / ``moves_left`` are both ``(N,)``. Swapping either pair inside
``forward`` -- the whole mutation is two lines -- leaves every shape check,
every ``NetOutput`` field-identity check and every per-head parameter-stream
check green, because ``test_netoutput_wiring`` pins the *field* (which a swap
preserves) and the init-stream nodes read the *module* (which a swap does not
touch). Nothing else in the tree closes the gap between the two, so a
cross-wired head trains on the other head's target and only becomes observable
at the stage gate whose entire job is attribution.

The node that closes it is the starvation argument the P0a suite already makes
for ``policy`` vs ``soft_policy``: differentiate one field, and the *other*
head's own parameters must not move. It lives here rather than beside its P0a
twin because ``tests/net/heads_behavior/test_heads_sanity.py`` is outside this
change's file allowlist; the claim, the four pairs and the mutant it is red
against are the review's.
"""

from __future__ import annotations

from typing import Any, Final

import pytest
import torch

from komugi import net as komugi_net
from komugi.net import HeadsConfig, KomugiNet, NetConfig, TrunkConfig
from komugi.net.heads import _flatten_action_planes

_IN_PLANES: Final[int] = 4
_CHANNELS: Final[int] = 16
_BOARD: Final[int] = 9
_SQUARES: Final[int] = _BOARD * _BOARD
_SPIKE: Final[float] = 4.0
"""Driven cell magnitude. Well away from zero so that Mish -- which is exactly
0 at 0 and monotone away from it -- carries the disturbance through rather than
attenuating it into float noise."""


def _net() -> KomugiNet:
    """A small net in eval mode.

    ``eval()`` matters: in train mode BatchNorm normalises using *batch*
    statistics, which mixes every spatial cell into every other and would make
    "only square s moved" false for reasons that have nothing to do with the
    ordering. In eval mode the norm is a fixed affine map, so the whole head is
    pointwise over the board and locality is exact.
    """
    config = NetConfig(
        trunk=TrunkConfig(
            in_planes=_IN_PLANES, channels=_CHANNELS, blocks=2, gpool_every=3, seed=5
        ),
        heads=HeadsConfig(
            channels=_CHANNELS, num_actions=_SQUARES, policy_planes=1, seed=5
        ),
    )
    net = KomugiNet(config)
    net.eval()
    return net


def _ownership_for(net: KomugiNet, features: torch.Tensor) -> torch.Tensor:
    """Return ``NetOutput.ownership_logits`` with the head fed `features`.

    The hook replaces whatever the trunk produced, so the forward pass is the
    real one end to end -- including the fact that it is ``net.ownership``,
    and not some other submodule, whose output lands in the named field.
    """

    def substitute(
        module: torch.nn.Module, args: tuple[Any, ...]
    ) -> tuple[torch.Tensor]:
        """Feed the ownership head `features` whatever the trunk handed it."""
        return (features,)

    handle = net.ownership.register_forward_pre_hook(substitute)
    try:
        with torch.no_grad():
            out = net(torch.zeros(1, _IN_PLANES, _BOARD, _BOARD))
    finally:
        handle.remove()
    logits: torch.Tensor = out.ownership_logits
    return logits


def test_ownership_output_square_index_is_the_engine_square_index() -> None:
    """All 81 squares, one at a time. A transposed board must fail.

    For each ``(row, col)`` the features are zero everywhere but that one
    spatial cell; the response must move at flat index ``row * 9 + col`` and
    **nowhere else**. Both halves are load-bearing: "it moved at s" alone is
    satisfied by a head that moves everywhere, and "it moved nowhere else"
    alone is satisfied by a head that moves nowhere.

    A transposition -- ``col * 9 + row``, the classic way to get this wrong --
    is caught at every square except the 9 on the diagonal, so 72 of the 81
    iterations are individually decisive.
    """
    net = _net()
    baseline = _ownership_for(net, torch.zeros(1, _CHANNELS, _BOARD, _BOARD))
    assert baseline.shape == (1, _SQUARES, 3)

    for row in range(_BOARD):
        for col in range(_BOARD):
            features = torch.zeros(1, _CHANNELS, _BOARD, _BOARD)
            features[0, :, row, col] = _SPIKE
            moved = _ownership_for(net, features)

            delta = (moved - baseline).abs().sum(dim=-1)[0]
            square = row * _BOARD + col
            assert float(delta[square].item()) > 1e-6, (
                f"driving cell (row={row}, col={col}) did not move square "
                f"{square}: the ownership readout is not row-major over the "
                "board"
            )
            elsewhere = torch.cat((delta[:square], delta[square + 1 :]))
            assert float(elsewhere.max().item()) < 1e-6, (
                f"driving cell (row={row}, col={col}) moved a square other "
                f"than {square}; the largest stray response was "
                f"{float(elsewhere.max().item())}"
            )


def test_the_ownership_square_order_is_the_policy_heads_square_order() -> None:
    """The two readouts flatten the board the same way, by construction.

    ``_flatten_action_planes`` is the single expression the policy plane
    ordering is defined by, and the ownership head's reshape must agree with
    it -- a permutation table between the two is exactly what this design
    refuses to introduce. Asserted against that function rather than against a
    re-derivation of it, so the two cannot drift apart.
    """
    planes = torch.arange(
        1 * 3 * _BOARD * _BOARD, dtype=torch.float32
    ).reshape(1, 3, _BOARD, _BOARD)

    # What the ownership head does: reshape to (N, 3, 81), then transpose.
    ownership = planes.reshape(1, 3, _SQUARES).transpose(1, 2)
    # What the policy head does: one flat plane-major run.
    flattened = _flatten_action_planes(planes)

    for plane in range(3):
        for square in range(_SQUARES):
            assert float(ownership[0, square, plane].item()) == float(
                flattened[0, plane * _SQUARES + square].item()
            ), (
                "the ownership readout and the policy readout disagree about "
                f"square {square} of plane {plane}"
            )


def test_the_three_ownership_classes_are_independent_per_square() -> None:
    """The class axis is last, and it is the axis the loss reduces over.

    A head that put the classes on the square axis would give an ``(N, 81, 3)``
    tensor of the right shape whose "squares" were classes -- shape-clean, and
    caught here: perturbing one *class* of the readout must move that class at
    every square, not one square at every class.

    The perturbation is the conv's **output-channel bias**, which is the one
    knob that is exactly a class: adding ``d`` to ``conv.bias[k]`` shifts
    output plane ``k`` uniformly over the board and nothing else, with no
    nonlinearity after it, so the correct response is ``d`` at class ``k`` on
    all 81 squares and exactly 0 elsewhere.

    That is what makes this node non-vacuous, and the earlier form of it was
    not (impl review F10): asserting that a softmax over the last axis sums to
    one is a property of ``softmax``, true of *any* ``(1, 81, 3)`` tensor
    whatever the two axes mean. Under the reshape-without-transpose bug
    (mutant M12) plane ``k``'s bias lands on the contiguous run
    ``[81k, 81k + 81)`` of the flattened readout, i.e. on 27 *squares* across
    all three classes -- confined, but confined to the wrong axis, which the
    old assertion could not see and the two clauses below both catch.
    """
    net = _net()
    features = torch.zeros(1, _CHANNELS, _BOARD, _BOARD)
    features[0, 0, 4, 4] = _SPIKE
    baseline = _ownership_for(net, features)
    assert baseline.shape == (1, _SQUARES, 3)

    bias = net.ownership.conv.bias
    assert bias is not None, "the ownership conv is built with bias=True"
    original = bias.detach().clone()
    for klass in range(3):
        with torch.no_grad():
            bias.copy_(original)
            bias[klass] += _SPIKE
        delta = (_ownership_for(net, features) - baseline)[0]

        moved = delta[:, klass].abs()
        assert float(moved.min().item()) > 1e-6, (
            f"perturbing class {klass} left square "
            f"{int(moved.argmin().item())} of that class unmoved: the class "
            "axis is not the last axis"
        )
        others = torch.cat((delta[:, :klass], delta[:, klass + 1 :]), dim=-1)
        assert float(others.abs().max().item()) < 1e-6, (
            f"perturbing class {klass} moved another class; the largest stray "
            f"response was {float(others.abs().max().item())}"
        )

    with torch.no_grad():
        bias.copy_(original)


# --------------------------------------------------------------------------
# Field -> submodule: which head's output landed in which NetOutput field
# --------------------------------------------------------------------------

_STARVATION: Final[tuple[Any, ...]] = (
    pytest.param(
        "short_term_value",
        ("DraftOutcomeHead", "draft_outcome"),
        id="short-term-value-does-not-reach-draft-outcome",
    ),
    pytest.param(
        "draft_outcome_logits",
        ("ShortTermValueHead", "short_term_value"),
        id="draft-outcome-does-not-reach-short-term-value",
    ),
    pytest.param(
        "uncertainty",
        ("MLHHead", "mlh"),
        id="uncertainty-does-not-reach-mlh",
    ),
    pytest.param(
        "moves_left",
        ("UncertaintyHead", "uncertainty"),
        id="moves-left-does-not-reach-uncertainty",
    ),
)
"""The two shape-identical pairs, each direction. ``(N, 3)`` and ``(N,)``.

Both directions of each pair, deliberately: a swap of the two assignments
makes *both* directions red, but a one-sided mistake -- ``short_term_value``
assigned from ``self.draft_outcome`` while ``draft_outcome_logits`` keeps its
own head -- makes only one, and that is the shape the mistake takes when it is
a copy-paste rather than a transposition.
"""

_PRODUCER: Final[dict[str, str]] = {
    "short_term_value": "short_term_value",
    "draft_outcome_logits": "draft_outcome",
    "uncertainty": "uncertainty",
    "moves_left": "mlh",
}
"""``NetOutput`` field -> the ``KomugiNet`` attribute that must produce it.

The binding this module asserts, written down once. ``moves_left`` comes from
``net.mlh`` and ``draft_outcome_logits`` from ``net.draft_outcome``: the field
names and the submodule names differ for two of the four, which is part of why
a swap reads as plausible in ``forward``.
"""


def _starved_head(net: KomugiNet, class_name: str, child: str) -> torch.nn.Module:
    """The submodule that IS the named head, located by exact type.

    Exact type, not name: the four staged heads and ``MLHHead`` all subclass
    ``_PooledScalarHead``, so an ``isinstance`` walk would match five modules
    where the contract names one. The child-attribute assertion is the second
    half -- it pins that ``KomugiNet`` reaches this class through the attribute
    the design named, so a rename cannot make the lookup silently find a
    different module.
    """
    cls = getattr(komugi_net, class_name, None)
    assert cls is not None, f"komugi.net exports no {class_name}"
    exact = [module for module in net.modules() if type(module) is cls]
    assert len(exact) == 1, f"expected exactly one {class_name}, found {len(exact)}"
    assert getattr(net, child) is exact[0], (
        f"KomugiNet.{child} is not the net's {class_name}"
    )
    module: torch.nn.Module = exact[0]
    return module


@pytest.mark.parametrize(("field", "starved"), _STARVATION)
def test_a_single_heads_loss_starves_the_other_heads_parameters(
    field: str, starved: tuple[str, str]
) -> None:
    """A loss on one field must not reach the shape-identical head's weights.

    This is the field-to-module binding, and it is the *only* thing that
    distinguishes ``short_term_value=self.short_term_value(features)`` from
    ``short_term_value=self.draft_outcome(features)``. Differentiating the
    named field must move the head that is supposed to produce it and leave the
    other one's own parameters at exactly zero gradient. The trunk beneath them
    is shared and is expected to move either way -- only the sibling head's own
    weights are asserted untouched.

    Non-vacuity is asserted rather than assumed: a backward that reached
    *nothing* would satisfy the starvation clause trivially, so the head the
    field is supposed to come from is checked to have received a gradient in
    the same pass.
    """
    net = _net()
    net.train()
    net.zero_grad(set_to_none=True)

    generator = torch.Generator().manual_seed(20250906)
    features = torch.randn(
        4, _IN_PLANES, _BOARD, _BOARD, generator=generator, dtype=torch.float32
    )
    output = net(features)
    scalar: torch.Tensor = getattr(output, field).pow(2).mean()
    scalar.backward()  # type: ignore[no-untyped-call]

    starved_head = _starved_head(net, starved[0], starved[1])
    touched = [
        name
        for name, parameter in starved_head.named_parameters()
        if parameter.grad is not None
        and float(parameter.grad.abs().max().item()) > 0.0
    ]
    assert touched == [], (
        f"a loss on NetOutput.{field} moved {starved[0]}'s parameters "
        f"{touched}: the field is wired to the wrong submodule"
    )

    # Non-vacuity: the field's OWN head really did receive the gradient, so
    # the clause above is about the wiring and not about a dead backward.
    producer = getattr(net, _PRODUCER[field])
    moved = [
        name
        for name, parameter in producer.named_parameters()
        if parameter.grad is not None
        and float(parameter.grad.abs().max().item()) > 0.0
    ]
    assert moved != [], (
        f"a loss on NetOutput.{field} moved no parameter of "
        f"net.{_PRODUCER[field]}: nothing about the wiring was exercised"
    )
