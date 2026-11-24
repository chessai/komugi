"""Unit tests for the heads and the assembled net: their contract.

Shapes, dtypes, configuration validation, the promises construction makes
(pure function of the config, invisible to the global RNG), and -- the one
test here that is worth more than all the others -- that the policy head's
flatten puts action id ``i`` at index ``i``.

The behavioural properties (initialisation statistics, gradient flow, batch
independence, what the heads *do* to a real position) belong to the companion
oracle suite and are deliberately not duplicated here.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest
import torch

from komugi.engine.action_encoding import (
    A1_START,
    ACTION_SIZE,
    B_START,
    C_START,
    encode_move,
)
from komugi.net import (
    HeadsConfig,
    KomugiNet,
    MLHHead,
    NetConfig,
    NetOutput,
    PolicyHead,
    SoftPolicyHead,
    TrunkConfig,
    WDLHead,
)
from komugi.net.heads import _flatten_action_planes

# Narrow enough that a forward pass costs nothing, even and >= 2 so that the
# scalar heads have a hidden width; the action space stays full size, because
# it is the thing under test.
_HEADS = HeadsConfig(channels=16)

# A whole net at the same width, two blocks deep.
_NET = NetConfig(
    trunk=TrunkConfig(in_planes=7, channels=16, blocks=2, gpool_every=2),
    heads=_HEADS,
)

_BATCH = 2


def _features(seed: int = 3) -> torch.Tensor:
    """A ``(2, 16, 9, 9)`` stand-in for the trunk's output."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(_BATCH, _HEADS.channels, 9, 9, generator=generator)


def _state_dicts_equal(
    left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]
) -> bool:
    """Return whether two state dicts have the same keys and equal tensors."""
    if set(left) != set(right):
        return False
    return all(torch.equal(left[key], right[key]) for key in left)


# ---------------------------------------------------------------------------
# The policy plane ordering
# ---------------------------------------------------------------------------


def test_the_flatten_puts_the_worked_example_where_the_docstring_says() -> None:
    """Plane 241 at spatial ``(3, 5)`` is action id 19553, and nothing else is.

    This is the worked example from :mod:`komugi.net.heads`, asserted through
    the head's own flatten rather than through a re-derivation of it. Id 19553
    is a plain tsuke from 5-5 onto 4-4: block A1, block-local plane
    ``160 = from_sq 40 * 4 + sel 0``, destination square 32.

    The third assertion is the one that would catch a transposed board: square
    32 is rank 4, file 4, which is row ``4 - 1 = 3`` and column ``9 - 4 = 5``
    -- not row 5, column 3.
    """
    planes = torch.zeros(1, _HEADS.policy_planes, 9, 9)
    planes[0, 241, 3, 5] = 1.0

    flat = _flatten_action_planes(planes)

    assert flat.shape == (1, _HEADS.num_actions)
    assert float(flat[0, 19553]) == 1.0
    assert int(flat.nonzero()[0, 1]) == 19553
    # ... and the engine agrees that this is what 19553 means.
    assert encode_move({"type": "tsuke", "from": "5-5", "to": "4-4"}) == 19553


def test_the_flatten_is_plane_major_at_every_block_boundary() -> None:
    """``plane * 81 + sq`` holds at the first id of each encoding block.

    The block starts are the running sums of the per-block plane counts (81,
    324, 28, 28), so a boundary is exactly where an off-by-one in the plane
    layout would first show up as a whole block of scrambled targets.
    """
    boundaries = {
        0: (0, 0, 0),  # A0: from_sq 0, to_sq 0
        A1_START: (81, 0, 0),  # A1: first stacking plane
        B_START: (405, 0, 0),  # B: first arata plane
        C_START: (433, 0, 0),  # C: first arata + draft-finish plane
        ACTION_SIZE - 1: (460, 8, 8),  # the very last id
    }

    for action_id, (plane, row, col) in boundaries.items():
        planes = torch.zeros(1, _HEADS.policy_planes, 9, 9)
        planes[0, plane, row, col] = 1.0

        flat = _flatten_action_planes(planes)

        assert int(flat.nonzero()[0, 1]) == action_id, (
            f"plane {plane} at ({row}, {col}) should be action {action_id}"
        )


def test_every_square_of_one_plane_is_one_contiguous_id_run() -> None:
    """Plane ``p`` owns ids ``[p * 81, (p + 1) * 81)``, in row-major order."""
    plane = 137
    planes = torch.zeros(1, _HEADS.policy_planes, 9, 9)
    planes[0, plane] = torch.arange(81, dtype=torch.float32).reshape(9, 9)

    flat = _flatten_action_planes(planes)

    start = plane * 81
    assert torch.equal(
        flat[0, start : start + 81], torch.arange(81, dtype=torch.float32)
    )
    assert float(flat[0, :start].abs().sum()) == 0.0
    assert float(flat[0, start + 81 :].abs().sum()) == 0.0


def test_the_policy_head_forward_is_the_documented_composition() -> None:
    """``forward`` is norm -> act -> conv -> that flatten, with nothing extra.

    Built out of the head's *own* submodules, so what this pins is the absence
    of a permutation, transpose or reordering between the convolution and the
    logits -- the one place a plane-ordering bug could hide after the flatten
    itself has been checked.
    """
    head = PolicyHead(_HEADS)
    head.eval()
    x = _features()

    with torch.no_grad():
        expected = _flatten_action_planes(head.conv(head.act(head.norm(x))))
        assert torch.equal(head(x), expected)


# ---------------------------------------------------------------------------
# Shapes and dtypes
# ---------------------------------------------------------------------------


def test_the_two_policy_heads_emit_one_logit_per_action() -> None:
    """``(N, C, 9, 9)`` in, ``(N, num_actions)`` float32 out, for both."""
    x = _features()

    for head in (PolicyHead(_HEADS), SoftPolicyHead(_HEADS)):
        out = head(x)

        assert out.shape == (_BATCH, _HEADS.num_actions)
        assert out.dtype == torch.float32
        assert bool(torch.isfinite(out).all())


def test_the_wdl_head_emits_three_logits() -> None:
    """``(N, 3)`` float32: win, draw, loss -- raw, so not a distribution."""
    out = WDLHead(_HEADS)(_features())

    assert out.shape == (_BATCH, 3)
    assert out.dtype == torch.float32
    assert bool(torch.isfinite(out).all())


def test_the_moves_left_head_emits_a_bare_scalar_per_sample() -> None:
    """``(N,)`` and not ``(N, 1)``: the squeeze is part of the contract."""
    out = MLHHead(_HEADS)(_features())

    assert out.shape == (_BATCH,)
    assert out.dtype == torch.float32
    assert bool(torch.isfinite(out).all())


def test_the_net_returns_every_head_named() -> None:
    """One forward pass, eight predictions, unpackable by name and by position.

    The four staged heads are in this tuple because the net freeze puts them
    in the net from the first P0b checkpoint (KOM-005b D1); their *losses* are
    what stages. An omitted field here is an omitted parameter set in every
    checkpoint the campaign writes.
    """
    net = KomugiNet(_NET)
    generator = torch.Generator().manual_seed(9)
    x = torch.randn(_BATCH, _NET.trunk.in_planes, 9, 9, generator=generator)

    out = net(x)

    assert isinstance(out, NetOutput)
    assert out._fields == (
        "policy_logits",
        "soft_policy_logits",
        "wdl_logits",
        "moves_left",
        "uncertainty",
        "ownership_logits",
        "short_term_value",
        "draft_outcome_logits",
    )
    assert out.policy_logits.shape == (_BATCH, _HEADS.num_actions)
    assert out.soft_policy_logits.shape == (_BATCH, _HEADS.num_actions)
    assert out.wdl_logits.shape == (_BATCH, 3)
    assert out.moves_left.shape == (_BATCH,)
    assert out.uncertainty.shape == (_BATCH,)
    assert out.ownership_logits.shape == (_BATCH, 81, 3)
    assert out.short_term_value.shape == (_BATCH, 3)
    assert out.draft_outcome_logits.shape == (_BATCH, 3)
    assert all(
        tensor.dtype == torch.float32 and bool(torch.isfinite(tensor).all())
        for tensor in out
    )


def test_the_two_policy_heads_are_independent_parameters() -> None:
    """Same shape, same config, different weights -- and no shared storage."""
    net = KomugiNet(_NET)

    assert not _state_dicts_equal(
        net.policy.state_dict(), net.soft_policy.state_dict()
    )
    assert net.policy.conv.weight.data_ptr() != net.soft_policy.conv.weight.data_ptr()


def test_the_output_layers_start_unbiased() -> None:
    """Every readout bias is zero at init, as the initialisation section says."""
    net = KomugiNet(_NET)

    for bias in (
        net.policy.conv.bias,
        net.soft_policy.conv.bias,
        net.wdl.out_linear.bias,
        net.mlh.out_linear.bias,
    ):
        # torch types a convolution's bias as optional; every one of these was
        # built with bias=True, so a None here is itself the failure.
        assert bias is not None
        assert float(bias.detach().abs().max()) == 0.0


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_heads_config_defaults_are_the_engine_action_space() -> None:
    """37341 = 461 * 81, and 37341 is what the engine encodes into."""
    config = HeadsConfig(channels=128)

    assert config.num_actions == ACTION_SIZE
    assert config.num_actions == config.policy_planes * 81
    assert config.value_hidden == 64


def test_heads_config_rejects_shapes_that_cannot_describe_a_head() -> None:
    """Positive even width, positive planes, and planes that match the ids."""
    with pytest.raises(ValueError, match="channels"):
        HeadsConfig(channels=0)
    with pytest.raises(ValueError, match="even"):
        HeadsConfig(channels=15)
    with pytest.raises(ValueError, match="policy_planes"):
        HeadsConfig(channels=16, num_actions=0, policy_planes=0)
    with pytest.raises(ValueError, match="seed"):
        HeadsConfig(channels=16, seed=-1)


def test_heads_config_rejects_an_action_count_that_is_not_planes_times_81() -> None:
    """The two fields are one number written twice; disagreement is an error."""
    with pytest.raises(ValueError, match="num_actions"):
        HeadsConfig(channels=16, num_actions=37340, policy_planes=461)
    with pytest.raises(ValueError, match="num_actions"):
        HeadsConfig(channels=16, num_actions=37341, policy_planes=460)

    # The consistent smaller space is legal: nothing hardcodes 461.
    small = HeadsConfig(channels=16, num_actions=162, policy_planes=2)
    assert small.num_actions == 162


def test_net_config_rejects_heads_that_do_not_fit_the_trunk() -> None:
    """A width disagreement is caught at config time, not at forward time."""
    with pytest.raises(ValueError, match="channels"):
        NetConfig(
            trunk=TrunkConfig(in_planes=7, channels=16),
            heads=HeadsConfig(channels=32),
        )


def test_the_stored_configs_are_the_ones_passed_in() -> None:
    """A net and its heads keep their configs unmodified."""
    net = KomugiNet(_NET)

    assert net.config == _NET
    assert net.trunk.config == _NET.trunk
    assert net.policy.config == _HEADS
    assert net.mlh.config == _HEADS


# ---------------------------------------------------------------------------
# Determinism, RNG hygiene, checkpoints
# ---------------------------------------------------------------------------


def test_the_same_config_builds_the_same_net() -> None:
    """Construction is a pure function of the config."""
    assert _state_dicts_equal(
        KomugiNet(_NET).state_dict(), KomugiNet(_NET).state_dict()
    )


def test_a_different_head_seed_builds_different_heads() -> None:
    """The heads' seed is load-bearing, and reaches only the heads."""
    other = NetConfig(
        trunk=_NET.trunk,
        heads=HeadsConfig(channels=_HEADS.channels, seed=_HEADS.seed + 1),
    )

    left = KomugiNet(_NET).state_dict()
    right = KomugiNet(other).state_dict()

    assert not _state_dicts_equal(left, right)
    assert torch.equal(left["trunk.stem.weight"], right["trunk.stem.weight"])
    assert not torch.equal(left["policy.conv.weight"], right["policy.conv.weight"])


def test_construction_leaves_the_global_rng_untouched() -> None:
    """Building a net neither reads nor advances the global generator.

    Both halves matter: the caller's seeded stream is not perturbed, and the
    parameters do not depend on what that stream happened to be.
    """
    torch.manual_seed(1234)
    before = torch.get_rng_state()
    first = KomugiNet(_NET)
    after = torch.get_rng_state()

    assert torch.equal(before, after)

    torch.manual_seed(4321)
    second = KomugiNet(_NET)

    assert _state_dicts_equal(first.state_dict(), second.state_dict())


def test_state_dict_round_trips_through_save_and_load(tmp_path: Path) -> None:
    """Saved and reloaded parameters are bit-identical, and still predict."""
    original = KomugiNet(_NET)
    path = tmp_path / "net.pt"
    torch.save(original.state_dict(), path)

    restored = KomugiNet(
        NetConfig(
            trunk=_NET.trunk,
            heads=HeadsConfig(channels=_HEADS.channels, seed=_HEADS.seed + 7),
        )
    )
    assert not _state_dicts_equal(original.state_dict(), restored.state_dict())

    restored.load_state_dict(torch.load(path, weights_only=True))

    assert _state_dicts_equal(original.state_dict(), restored.state_dict())

    original.eval()
    restored.eval()
    generator = torch.Generator().manual_seed(5)
    x = torch.randn(_BATCH, _NET.trunk.in_planes, 9, 9, generator=generator)
    with torch.no_grad():
        left = original(x)
        right = restored(x)

    assert all(
        torch.equal(a, b) for a, b in zip(left, right, strict=True)
    )
    # An all-zero readout would satisfy the equality above while meaning the
    # heads had annihilated the trunk's signal.
    assert float(left.policy_logits.std()) > 0.0
