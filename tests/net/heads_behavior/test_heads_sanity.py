"""Eight heads, eight outputs, eight live gradients (KOM-105, KOM-005b).

The shape contract in one place -- ``NetOutput(policy_logits,
soft_policy_logits, wdl_logits, moves_left, uncertainty, ownership_logits,
short_term_value, draft_outcome_logits)``, in that order, with widths
37341 / 37341 / 3 / scalar / scalar / 81x3 / 3 / 3 -- plus the two properties
that a net can violate while still training to a falling loss:

* **the heads are separate learners.**  A soft-policy head that shares
  parameters with the policy head (or, worse, *is* the policy head behind two
  names) makes the auxiliary target a second opinion on nothing: the whole
  point of the soft head is that it fits a different, higher-temperature
  target with its own capacity.  Pinned by perturbing one head's parameters
  and requiring the other's output to be bit-identical, and by requiring the
  two parameter sets to be literally disjoint tensors.
* **every parameter is reachable.**  The trunk suite (KOM-104) makes this
  claim about the trunk; a head is exactly as easy to build, register and then
  forget to wire, so the same walk is repeated over the whole net -- missing,
  non-finite and identically-zero gradients each get their own bucket, because
  ``grad is None`` and ``grad == 0`` are different bugs (out of the graph vs.
  detached from the loss).

Run in train mode where it matters (BatchNorm affine parameters take their
gradients through the normalised activations), eval mode where per-sample
determinism does.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

net = pytest.importorskip("komugi.net")

if not hasattr(net, "KomugiNet"):
    pytest.skip(
        "komugi.net exports no KomugiNet yet (KOM-005a)", allow_module_level=True
    )

_IN_PLANES = 8
_CHANNELS = 16
_BOARD = 9
_BATCH = 4

#: The NamedTuple's fields, in order.  Order is contract, not cosmetics: a
#: caller that unpacks ``policy, soft, wdl, mlh, ... = model(x)`` gets silently
#: wrong tensors if the head order ever changes.  The staged four are appended
#: in ``DESIGN.html`` §2.4's normative staging order.
_FIELDS = (
    "policy_logits",
    "soft_policy_logits",
    "wdl_logits",
    "moves_left",
    "uncertainty",
    "ownership_logits",
    "short_term_value",
    "draft_outcome_logits",
)


def _noise(shape: tuple[int, ...], *, seed: int) -> torch.Tensor:
    """Gaussian noise from a LOCAL generator -- the global RNG stays untouched."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randn(*shape, generator=generator)


def _build(*, train: bool = False) -> Any:
    config = net.NetConfig(
        trunk=net.TrunkConfig(
            in_planes=_IN_PLANES, channels=_CHANNELS, blocks=2, gpool_every=3, seed=0
        ),
        heads=net.HeadsConfig(channels=_CHANNELS),
    )
    model = net.KomugiNet(config)
    if train:
        model.train()
    else:
        model.eval()
    return model


def _forward(model: Any, *, batch: int = _BATCH, seed: int = 105) -> Any:
    with torch.no_grad():
        return model(_noise((batch, _IN_PLANES, _BOARD, _BOARD), seed=seed))


def _head(model: Any, class_name: str, keyword: str) -> Any:
    """The submodule that IS the named head.

    Located by exact type first -- the contract exports the four head classes,
    and exact type distinguishes ``SoftPolicyHead`` from ``PolicyHead`` even if
    one subclasses the other -- and by child name as a fallback, so a head
    composed rather than subclassed is still found.  ``exclude`` keeps the
    name fallback for ``PolicyHead`` from also matching ``soft_policy``.
    """
    exclude = "soft" if class_name == "PolicyHead" else None
    cls = getattr(net, class_name, None)
    assert cls is not None, f"komugi.net exports no {class_name}"
    exact: list[Any] = [module for module in model.modules() if type(module) is cls]
    if len(exact) == 1:
        return exact[0]
    named: list[Any] = [
        module
        for name, module in model.named_children()
        if keyword in name.lower() and (exclude is None or exclude not in name.lower())
    ]
    assert len(named) == 1, (
        f"cannot locate the {class_name} inside KomugiNet: {len(exact)} submodules "
        f"of that exact type and {len(named)} children whose name mentions "
        f"{keyword!r}"
    )
    return named[0]


def _parameter_ids(module: Any) -> set[int]:
    return {id(parameter) for parameter in module.parameters()}


# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------


def test_the_output_is_the_eight_named_fields_in_order() -> None:
    """``NetOutput`` is a NamedTuple, and its field order and arity are contract.

    Three assertions, and the third is the one that pins KOM-005b D18. Field
    equality and an eight-wide unpack are both satisfied by a NamedTuple whose
    four staged fields carry *defaults* -- and a default is exactly what D18
    forbids, because it lets a fake net omit a head, hand
    ``outputs_from_net`` a ``None``-shaped hole, and turn a loud ``TypeError``
    at the seam into a silently untrained head. ``_field_defaults == {}`` is
    what makes this node red against that.
    """
    output = _forward(_build())
    assert tuple(getattr(type(output), "_fields")) == _FIELDS

    (
        policy,
        soft,
        wdl,
        moves_left,
        uncertainty,
        ownership,
        short_term_value,
        draft_outcome,
    ) = output
    assert policy is output.policy_logits
    assert soft is output.soft_policy_logits
    assert wdl is output.wdl_logits
    assert moves_left is output.moves_left
    assert uncertainty is output.uncertainty
    assert ownership is output.ownership_logits
    assert short_term_value is output.short_term_value
    assert draft_outcome is output.draft_outcome_logits

    assert getattr(type(output), "_field_defaults") == {}, (
        "every NetOutput field must be required (KOM-005b D18): a default "
        "would let a fake net omit a head and train nothing through it"
    )


def test_the_head_shapes_are_the_contract() -> None:
    """37341 / 37341 / 3 / scalar-per-sample, and every one of them finite."""
    output = _forward(_build())
    assert output.policy_logits.shape == (_BATCH, 37341)
    assert output.soft_policy_logits.shape == (_BATCH, 37341)
    assert output.wdl_logits.shape == (_BATCH, 3)
    assert output.moves_left.shape == (_BATCH,), (
        "moves_left is a scalar per sample -- shape (N,), not (N, 1): a trailing "
        "singleton broadcasts against a (N,) target and silently trains an "
        "N-by-N loss matrix"
    )
    assert output.uncertainty.shape == (_BATCH,), (
        "uncertainty is a scalar per sample -- shape (N,), not (N, 1), for the "
        "same broadcasting reason moves_left is"
    )
    assert output.ownership_logits.shape == (_BATCH, 81, 3)
    assert output.short_term_value.shape == (_BATCH, 3)
    assert output.draft_outcome_logits.shape == (_BATCH, 3)
    for field in _FIELDS:
        tensor: torch.Tensor = getattr(output, field)
        assert bool(torch.isfinite(tensor).all()), f"{field} is not finite at init"


def test_the_wdl_logits_softmax_to_a_distribution() -> None:
    """Three raw logits per sample; the head does not pre-normalise them."""
    output = _forward(_build(), batch=3, seed=7)
    wdl: torch.Tensor = output.wdl_logits
    probabilities = torch.softmax(wdl, dim=1)

    assert probabilities.shape == (3, 3)
    assert bool(torch.isfinite(probabilities).all())
    row_sums = probabilities.sum(dim=1)
    assert float((row_sums - 1.0).abs().max().item()) < 1e-5
    assert float(probabilities.min().item()) > 0.0

    # Raw logits, not probabilities already: a head that softmaxed internally
    # would hand back rows that already sum to one.
    assert float((wdl.sum(dim=1) - 1.0).abs().max().item()) > 1e-6 or float(
        wdl.min().item()
    ) < 0.0


def test_the_moves_left_head_is_a_finite_scalar_per_sample() -> None:
    """No shape surprises and no NaN out of a freshly initialised MLH head."""
    output = _forward(_build(), batch=5, seed=11)
    moves_left: torch.Tensor = output.moves_left
    assert moves_left.shape == (5,)
    assert moves_left.dim() == 1
    assert bool(torch.isfinite(moves_left).all())


# ---------------------------------------------------------------------------
# The heads are separate learners
# ---------------------------------------------------------------------------


def test_the_policy_and_soft_policy_heads_own_disjoint_parameters() -> None:
    """Not one shared tensor between them -- no tying, no aliasing."""
    model = _build()
    policy = _head(model, "PolicyHead", "policy")
    soft = _head(model, "SoftPolicyHead", "soft")
    assert policy is not soft, "the two policy heads are the same module object"

    policy_ids = _parameter_ids(policy)
    soft_ids = _parameter_ids(soft)
    assert policy_ids, "the policy head has no parameters at all"
    assert soft_ids, "the soft-policy head has no parameters at all"
    assert policy_ids & soft_ids == set(), (
        "the policy and soft-policy heads share parameter tensors -- the "
        "auxiliary target would be fitted by the same weights it is meant to "
        "give a second opinion against"
    )


@pytest.mark.parametrize(
    ("perturbed", "kept"),
    [
        pytest.param(("SoftPolicyHead", "soft"), "policy_logits", id="perturb-soft"),
        pytest.param(("PolicyHead", "policy"), "soft_policy_logits", id="perturb-hard"),
    ],
)
def test_perturbing_one_policy_head_leaves_the_other_bit_identical(
    perturbed: tuple[str, str], kept: str
) -> None:
    """The independence claim, made behaviourally rather than structurally."""
    model = _build()
    inputs = _noise((2, _IN_PLANES, _BOARD, _BOARD), seed=23)
    with torch.no_grad():
        before = model(inputs)
    before_kept: torch.Tensor = getattr(before, kept).clone()
    moved = "soft_policy_logits" if kept == "policy_logits" else "policy_logits"
    before_moved: torch.Tensor = getattr(before, moved).clone()

    head = _head(model, perturbed[0], perturbed[1])
    with torch.no_grad():
        for parameter in head.parameters():
            parameter.add_(0.25)

    with torch.no_grad():
        after = model(inputs)

    assert bool(torch.equal(getattr(after, kept), before_kept)), (
        f"perturbing {perturbed[0]} moved {kept} -- the two policy heads are "
        "not independent"
    )
    assert not bool(torch.equal(getattr(after, moved), before_moved)), (
        f"perturbing every parameter of {perturbed[0]} did not move {moved}; "
        "the head is built but not wired into its own output"
    )
    for other in ("wdl_logits", "moves_left"):
        assert bool(torch.equal(getattr(after, other), getattr(before, other))), (
            f"perturbing {perturbed[0]} moved {other}"
        )


# ---------------------------------------------------------------------------
# Gradients
# ---------------------------------------------------------------------------


def _combined_loss(output: Any) -> torch.Tensor:
    """One scalar that every head contributes to, all eight weighted alike.

    Built by walking ``_FIELDS`` rather than by naming four of them: the whole
    subject of the walk below is that no parameter is left out of the graph,
    and a hand-written sum is exactly the thing that goes stale when a head is
    added. A staged head is off in the *loss recipe*, never absent from the
    net, so it must reach this scalar like any other.
    """
    total: torch.Tensor = torch.zeros(())
    for field in _FIELDS:
        tensor: torch.Tensor = getattr(output, field)
        total = total + tensor.pow(2).mean()
    return total


def test_every_parameter_of_the_whole_net_receives_a_live_gradient() -> None:
    """Trunk and all eight heads, from one combined scalar loss.

    Same three buckets as the trunk suite: ``None`` means the parameter never
    entered the graph, an all-zero gradient means it entered but was detached
    from the loss (a dropped branch, a ``.detach()``), and a non-finite one
    means the head is numerically broken at initialisation.
    """
    model = _build(train=True)
    model.zero_grad(set_to_none=True)
    output = model(_noise((_BATCH, _IN_PLANES, _BOARD, _BOARD), seed=29))
    loss = _combined_loss(output)
    loss.backward()  # type: ignore[no-untyped-call]  # torch stubs leave Tensor.backward untyped

    missing: list[str] = []
    dead: list[str] = []
    broken: list[str] = []
    total = 0
    for name, parameter in model.named_parameters():
        total += 1
        grad = parameter.grad
        if grad is None:
            missing.append(name)
            continue
        if not bool(torch.isfinite(grad).all()):
            broken.append(name)
        elif float(grad.abs().max().item()) == 0.0:
            dead.append(name)

    assert total > 0, "the net exposes no parameters at all"
    assert missing == [], f"parameters outside the autograd graph: {missing}"
    assert broken == [], f"parameters with NaN/Inf gradients: {broken}"
    assert dead == [], f"parameters with an all-zero gradient: {dead}"


@pytest.mark.parametrize(
    ("field", "starved"),
    [
        pytest.param("wdl_logits", ("PolicyHead", "policy"), id="wdl-loss"),
        pytest.param("policy_logits", ("WDLHead", "wdl"), id="policy-loss"),
    ],
)
def test_a_single_heads_loss_starves_the_other_heads_parameters(
    field: str, starved: tuple[str, str]
) -> None:
    """A loss on one head must not reach another head's own weights.

    The complement of the combined-loss walk above: it proves the heads are
    all connected, this proves they are not the SAME connection.  The trunk
    below them is shared and is expected to get gradient from either loss --
    only the other head's own parameters must stay untouched.
    """
    model = _build(train=True)
    model.zero_grad(set_to_none=True)
    output = model(_noise((_BATCH, _IN_PLANES, _BOARD, _BOARD), seed=31))
    scalar: torch.Tensor = getattr(output, field).pow(2).mean()
    scalar.backward()  # type: ignore[no-untyped-call]  # torch stubs leave Tensor.backward untyped

    head = _head(model, starved[0], starved[1])
    touched: list[str] = []
    for name, parameter in head.named_parameters():
        grad = parameter.grad
        if grad is not None and float(grad.abs().max().item()) > 0.0:
            touched.append(name)
    assert touched == [], (
        f"a loss on {field} moved {starved[0]}'s parameters {touched} -- the "
        "heads share weights"
    )
