"""``NetConfig`` couples the trunk to the heads, or nothing does (KOM-105).

The trunk emits ``(N, channels, 9, 9)`` and every head consumes it.  That is
one number shared by two config objects, and the failure mode of an unchecked
duplicate is not a crash: with the wrong width a ``Conv2d`` raises deep inside
the first forward pass, mid-training-run, hours after the config was written.
So the mismatch is rejected at config time, next to the two integers that
disagree.

The same argument applies to ``num_actions`` and ``policy_planes``: the policy
head is a 461-plane convolution over 9x9 followed by a flatten, so
``num_actions == policy_planes * 81`` is not a convention but an arithmetic
identity of the layer itself.  A config that violates it describes a head that
cannot be built.

Finally the scale target: b18c256 plus heads must be a config change and
nothing else -- constructed and forwarded here once, batch 1, as the whole-net
counterpart of the trunk suite's ``test_b18c256_constructs_and_forwards``.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from action_ids import NUM_ACTIONS, POLICY_PLANES, SQUARES, observation_planes

net = pytest.importorskip("komugi.net")

if not hasattr(net, "KomugiNet"):
    pytest.skip(
        "komugi.net exports no KomugiNet yet (KOM-005a)", allow_module_level=True
    )

_BOARD = 9


def _trunk_config(*, channels: int, in_planes: int = 8, blocks: int = 1) -> Any:
    return net.TrunkConfig(
        in_planes=in_planes, channels=channels, blocks=blocks, gpool_every=3, seed=0
    )


# ---------------------------------------------------------------------------
# The defaults
# ---------------------------------------------------------------------------


def test_the_heads_config_defaults_are_the_engines_action_space() -> None:
    """Only ``channels`` is required; the rest already describe Gungi."""
    config = net.HeadsConfig(channels=32)
    assert (config.channels, config.num_actions, config.policy_planes, config.seed) == (
        32,
        NUM_ACTIONS,
        POLICY_PLANES,
        0,
    )
    assert config.policy_planes * SQUARES == config.num_actions
    assert net.HeadsConfig(channels=32) == config, "HeadsConfig is not a value object"


def test_a_matching_pair_composes_into_a_net_config() -> None:
    """Equal widths on both sides is the whole precondition."""
    config = net.NetConfig(
        trunk=_trunk_config(channels=16), heads=net.HeadsConfig(channels=16)
    )
    assert config.trunk.channels == config.heads.channels == 16
    assert config.heads.num_actions == NUM_ACTIONS


# ---------------------------------------------------------------------------
# The rejections
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("trunk_channels", "head_channels"),
    [
        pytest.param(32, 16, id="head-narrower"),
        pytest.param(16, 32, id="head-wider"),
        pytest.param(256, 128, id="scaled-trunk-stale-head"),
    ],
)
def test_a_channel_mismatch_is_rejected_at_config_time(
    trunk_channels: int, head_channels: int
) -> None:
    """Not at the first forward pass, and not by a broadcast that half-works."""
    with pytest.raises(ValueError):
        net.NetConfig(
            trunk=_trunk_config(channels=trunk_channels),
            heads=net.HeadsConfig(channels=head_channels),
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"num_actions": NUM_ACTIONS - 1}, id="one-short"),
        pytest.param({"num_actions": NUM_ACTIONS + 1}, id="one-long"),
        pytest.param({"num_actions": POLICY_PLANES * 9}, id="planes-times-9"),
        pytest.param({"policy_planes": POLICY_PLANES - 1}, id="stale-plane-count"),
        pytest.param({"policy_planes": 0}, id="no-planes"),
        pytest.param({"num_actions": 0}, id="no-actions"),
    ],
)
def test_num_actions_must_be_planes_times_81(kwargs: dict[str, int]) -> None:
    """``num_actions == policy_planes * 81`` or the head cannot be built.

    Both constructions sit inside the ``raises`` block on purpose: the identity
    belongs to ``HeadsConfig`` and is expected to be enforced there, but a net
    that instead checks it when the two configs are joined is equally sound.
    What may NOT happen is that neither does and the flatten silently produces
    a vector of the wrong width.
    """
    fields: dict[str, int] = {"channels": 16}
    fields.update(kwargs)
    with pytest.raises(ValueError):
        heads = net.HeadsConfig(**fields)
        net.NetConfig(trunk=_trunk_config(channels=16), heads=heads)


@pytest.mark.parametrize(
    "channels",
    [pytest.param(0, id="zero"), pytest.param(-16, id="negative")],
)
def test_a_head_needs_a_positive_input_width(channels: int) -> None:
    """The head reads the trunk's channels; zero of them is not a net."""
    with pytest.raises(ValueError):
        net.HeadsConfig(channels=channels)


# ---------------------------------------------------------------------------
# The config actually drives the head
# ---------------------------------------------------------------------------


def test_the_policy_width_follows_the_config_rather_than_the_constant() -> None:
    """A 4-plane head emits 324 logits -- 37341 is a default, not a literal.

    The one cheap way to tell "reads its config" from "happens to agree with
    its config today": build a head whose action space is not Gungi's and
    watch the width move.  If 37341 is hardcoded anywhere in the head, this is
    where it shows.
    """
    config = net.HeadsConfig(channels=8, policy_planes=4, num_actions=4 * SQUARES)
    head = net.PolicyHead(config)
    head.eval()
    with torch.no_grad():
        logits: torch.Tensor = head(torch.zeros(2, 8, _BOARD, _BOARD))
    assert logits.shape == (2, 4 * SQUARES)


# ---------------------------------------------------------------------------
# The scale target
# ---------------------------------------------------------------------------


def test_b18c256_with_heads_constructs_and_forwards_batch_one() -> None:
    """The v2.1 shape, end to end, on the encoder's real plane count."""
    config = net.NetConfig(
        trunk=net.TrunkConfig(
            in_planes=observation_planes(),
            channels=256,
            blocks=18,
            gpool_every=3,
            seed=0,
        ),
        heads=net.HeadsConfig(channels=256),
    )
    assert (config.trunk.channels, config.trunk.blocks) == (256, 18)

    model = net.KomugiNet(config)
    model.eval()
    with torch.no_grad():
        output = model(torch.zeros(1, observation_planes(), _BOARD, _BOARD))

    assert output.policy_logits.shape == (1, NUM_ACTIONS)
    assert output.soft_policy_logits.shape == (1, NUM_ACTIONS)
    assert output.wdl_logits.shape == (1, 3)
    assert output.moves_left.shape == (1,)
    for name in ("policy_logits", "soft_policy_logits", "wdl_logits", "moves_left"):
        tensor: torch.Tensor = getattr(output, name)
        assert bool(torch.isfinite(tensor).all()), f"b18c256 {name} is not finite"
