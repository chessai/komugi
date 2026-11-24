"""The engine's legal mask and the net's policy vector are one vector (KOM-105).

End to end, through the real objects: a fresh :class:`komugi.engine.game.Game`,
its own legal mask, its own 128-plane observation, and one forward pass of a
small :class:`KomugiNet`.  Nothing here is synthetic -- that is the point.  The
mask and the policy vector are produced by two subsystems that share nothing at
runtime but a number, ``ACTION_SIZE``, and the moment they disagree about it
every masked softmax in MCTS is silently reading the wrong entries or crashing
on a shape.

The width is asserted from BOTH ends: the mask is as wide as the policy logits,
*and* ``cymovegen.fill_legal_mask`` is handed a buffer sized from the net's own
output.  A future ``ACTION_SIZE`` drift therefore fails here rather than in a
training run, whichever side moved first.

Runtime: one real observation, one forward of a 1-block / 16-channel net,
shared across the module by a module-scoped fixture.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt
import pytest
import torch

import action_ids
from action_ids import BOARD, NUM_ACTIONS
from komugi.engine import cymovegen
from komugi.engine.cyboard import board_from_fen
from komugi.engine.fen import ADVANCED_POSITION
from komugi.engine.game import Game
from komugi.engine.observation import observation_size

net = pytest.importorskip("komugi.net")

if not hasattr(net, "KomugiNet"):
    pytest.skip(
        "komugi.net exports no KomugiNet yet (KOM-005a)", allow_module_level=True
    )

_CHANNELS = 16


class Probe(NamedTuple):
    """One real position, pushed through one real net."""

    observation: npt.NDArray[np.float32]
    mask: npt.NDArray[np.uint8]
    output: Any
    logits: torch.Tensor


def _build_net(in_planes: int) -> Any:
    """A deliberately tiny net over the encoder's real plane count."""
    config = net.NetConfig(
        trunk=net.TrunkConfig(
            in_planes=in_planes, channels=_CHANNELS, blocks=1, gpool_every=3, seed=0
        ),
        heads=net.HeadsConfig(channels=_CHANNELS),
    )
    model = net.KomugiNet(config)
    model.eval()
    return model


@pytest.fixture(scope="module")
def probe() -> Probe:
    """The suite's single real forward pass."""
    game = Game()
    observation: npt.NDArray[np.float32] = np.asarray(
        game.observation(), dtype=np.float32
    )
    mask: npt.NDArray[np.uint8] = np.asarray(game.valid_moves(), dtype=np.uint8)

    model = _build_net(int(observation.shape[0]))
    inputs = torch.from_numpy(observation).unsqueeze(0)
    with torch.no_grad():
        output = model(inputs)
    logits: torch.Tensor = output.policy_logits
    return Probe(observation=observation, mask=mask, output=output, logits=logits)


def _legal_ids(mask: npt.NDArray[np.uint8]) -> list[int]:
    return [int(index) for index in np.nonzero(mask)[0]]


# ---------------------------------------------------------------------------
# Widths
# ---------------------------------------------------------------------------


def test_the_observation_is_the_encoders_and_the_net_eats_it_whole(
    probe: Probe,
) -> None:
    """``(128, 9, 9)`` in, one sample's worth of every head out."""
    assert probe.observation.shape == observation_size() == (128, BOARD, BOARD)
    assert probe.logits.shape == (1, NUM_ACTIONS)
    assert bool(torch.isfinite(probe.logits).all()), "the policy logits are not finite"


def test_the_mask_and_the_policy_vector_have_the_same_width(probe: Probe) -> None:
    """``len(mask) == ACTION_SIZE == num_actions == policy width``.

    Four numbers that must be one number.  Written out separately rather than
    chained through a single constant so a failure says which pair drifted.
    """
    assert probe.mask.shape == (NUM_ACTIONS,)
    assert Game.action_size() == NUM_ACTIONS
    assert int(probe.logits.shape[1]) == NUM_ACTIONS
    assert int(probe.logits.shape[1]) == int(probe.mask.shape[0])


def test_the_nets_config_declares_the_engines_action_size() -> None:
    """The head's ``num_actions`` is not merely compatible, it is the same number."""
    heads = net.HeadsConfig(channels=_CHANNELS)
    assert int(heads.num_actions) == Game.action_size() == NUM_ACTIONS
    assert int(heads.policy_planes) * 81 == int(heads.num_actions)


def test_fill_legal_mask_accepts_a_buffer_sized_from_the_policy_head(
    probe: Probe,
) -> None:
    """The engine's own mask filler, handed a buffer as wide as the net's output.

    This is the drift test from the engine's side: the buffer is sized by
    ``policy_logits.shape[1]`` and nothing else, so a policy head that grew or
    shrank away from ``ACTION_SIZE`` produces a wrong-sized buffer here and the
    Cython filler refuses it (or writes out of the block bounds it was
    compiled with, which its own bounds checking catches).
    """
    width = int(probe.logits.shape[1])
    board = board_from_fen(ADVANCED_POSITION)
    mask: npt.NDArray[np.uint8] = np.zeros(width, dtype=np.uint8)
    cymovegen.fill_legal_mask(board, mask)
    assert int(mask.sum()) > 0, "no legal move at the advanced start position"
    assert int(np.nonzero(mask)[0].max()) < width


# ---------------------------------------------------------------------------
# The masked softmax
# ---------------------------------------------------------------------------


def test_the_position_has_legal_moves_and_they_are_addressable(probe: Probe) -> None:
    """Every set bit indexes a real logit, and splits into a real (plane, square)."""
    legal = _legal_ids(probe.mask)
    assert legal, "the start position has no legal move -- nothing to mask"
    assert max(legal) < NUM_ACTIONS
    for action_id in legal:
        plane, square = action_ids.id_to_plane_and_square(action_id)
        assert action_ids.plane_and_square_to_id(plane, square) == action_id
        assert 0 <= plane < 461 and 0 <= square < 81


def test_masking_the_illegal_ids_leaves_a_valid_distribution(probe: Probe) -> None:
    """``softmax(logits.masked_fill(~legal, -inf))`` sums to 1 with zero leakage.

    The exact-zero assertion on the illegal entries is the one that matters:
    ``-inf`` before a softmax is the only masking that leaves no probability at
    all on an illegal move, and an off-by-one in either vector shows up as mass
    on a move the engine never generated.
    """
    legal = torch.from_numpy(probe.mask.astype(np.bool_))
    assert legal.shape == (NUM_ACTIONS,)

    masked = probe.logits.masked_fill(~legal.unsqueeze(0), float("-inf"))
    probabilities = torch.softmax(masked, dim=1)

    assert bool(torch.isfinite(probabilities).all()), "masked softmax produced NaN/Inf"
    assert float(probabilities.sum().item()) == pytest.approx(1.0, abs=1e-5)
    assert float(probabilities[0][~legal].abs().max().item()) == 0.0, (
        "probability mass landed on an illegal action id"
    )
    assert float(probabilities[0][legal].min().item()) > 0.0, (
        "a legal action id got exactly zero probability"
    )


def test_the_gathered_legal_logits_softmax_to_one(probe: Probe) -> None:
    """The other common form: gather the legal ids, then softmax the subvector."""
    legal = _legal_ids(probe.mask)
    index = torch.tensor(legal, dtype=torch.long)
    gathered = probe.logits[0].index_select(0, index)
    assert gathered.shape == (len(legal),)

    probabilities = torch.softmax(gathered, dim=0)
    assert float(probabilities.sum().item()) == pytest.approx(1.0, abs=1e-5)
    assert float(probabilities.min().item()) > 0.0
    assert bool(torch.isfinite(probabilities).all())


def test_the_soft_policy_masks_the_same_way(probe: Probe) -> None:
    """The auxiliary head shares the action space, so it shares the mask."""
    soft: torch.Tensor = probe.output.soft_policy_logits
    assert soft.shape == probe.logits.shape
    assert bool(torch.isfinite(soft).all())

    legal = torch.from_numpy(probe.mask.astype(np.bool_))
    probabilities = torch.softmax(
        soft.masked_fill(~legal.unsqueeze(0), float("-inf")), dim=1
    )
    assert float(probabilities.sum().item()) == pytest.approx(1.0, abs=1e-5)
    assert float(probabilities[0][~legal].abs().max().item()) == 0.0
