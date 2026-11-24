"""komugi.net.onnx_export -- ONNX export of the serving net, and
self-generated parity positions for the release tooling (KOM-1186 U3d).

This module holds the net-shape constants, the exporter itself, and a
self-contained source of parity positions. It imports only ``komugi``,
``torch``, ``onnx`` and ``numpy``.

:func:`export_net` uses the TorchScript exporter (``dynamo=False``), opset
17, with input ``obs`` float32 ``(N, IN_PLANES, 9, 9)`` and outputs
``policy``, ``wdl``, ``moves_left`` -- the three
:class:`~komugi.net.model.NetOutput` fields the release tooling reads, at
their torch shapes. :func:`parity_positions` plays seeded uniformly random
legal moves from the Advanced start (the format a released model plays) and
encodes the positions it visits exactly as self-play feeds the network, so
the release tooling's export-parity self-check needs no recorded games.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import onnx
import torch

from komugi.campaign import factory
from komugi.engine import observation
from komugi.engine.cyboard import board_from_fen
from komugi.engine.fen import ADVANCED_POSITION
from komugi.engine.game import Game, RepetitionRule, draft_root_mask
from komugi.net import HeadsConfig, KomugiNet, NetConfig, TrunkConfig

#: ONNX opset the export is pinned to.
OPSET_VERSION: Final[int] = 17
#: Output names, in the order ``torch.onnx.export`` is handed them -- also the
#: order they appear in the exported graph.
OUTPUT_NAMES: Final[tuple[str, str, str]] = ("policy", "wdl", "moves_left")
#: The single input's name.
INPUT_NAME: Final[str] = "obs"
#: ONNX ``metadata_props`` key the exported plane-map version is stamped under.
PLANE_MAP_VERSION_METADATA_KEY: Final[str] = "komugi.plane_map_version"
#: The ``batch`` value that requests a dynamic batch axis.
DYNAMIC_BATCH: Final[str] = "dynamic"
#: Concrete batch size a dynamic-batch export is traced at.
_DYNAMIC_TRACE_BATCH: Final[int] = 1


def _static_batch(batch: int | str) -> int | None:
    """Return `batch` as a static size, or ``None`` for :data:`DYNAMIC_BATCH`.

    Any other string is a :class:`ValueError`.
    """
    if isinstance(batch, str):
        if batch != DYNAMIC_BATCH:
            raise ValueError(
                f"batch must be an int or {DYNAMIC_BATCH!r}, got {batch!r}"
            )
        return None
    return batch


class _ExportWrapper(torch.nn.Module):
    """Wraps a :class:`KomugiNet` so ``forward`` returns exactly the three
    exported heads, in :data:`OUTPUT_NAMES` order -- the net's own
    :class:`~komugi.net.model.NetOutput` carries five more fields that this
    export does not touch.
    """

    def __init__(self, net: torch.nn.Module) -> None:
        super().__init__()
        self.net = net

    def forward(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.net(obs)
        return out.policy_logits, out.wdl_logits, out.moves_left


def fresh_small_net(seed: int) -> torch.nn.Module:
    """Return a freshly initialised b2c16 net seeded by `seed`, in eval mode.

    :data:`komugi.campaign.factory.SMALL_CONFIG`'s shape, with the seed forced
    to `seed` rather than SMALL_CONFIG's own fixed ``0``. ``TrunkConfig`` and
    ``HeadsConfig`` construction is a pure function of the config -- it
    neither reads nor advances the global torch RNG -- so this, and not
    ``torch.manual_seed``, is the one way to vary a fresh small net's weights.
    """
    small = factory.SMALL_CONFIG
    config = NetConfig(
        trunk=TrunkConfig(
            in_planes=factory.IN_PLANES,
            channels=small.trunk.channels,
            blocks=small.trunk.blocks,
            gpool_every=small.trunk.gpool_every,
            seed=seed,
        ),
        heads=HeadsConfig(
            channels=small.heads.channels,
            num_actions=factory.NUM_ACTIONS,
            policy_planes=small.heads.policy_planes,
            seed=seed,
        ),
    )
    net = KomugiNet(config)
    net.eval()
    return net


def export_net(net: torch.nn.Module, out: Path, batch: int | str = 1) -> None:
    """Export `net` to `out`: ONNX opset 17, TorchScript exporter.

    `batch` is either a static batch size (the default, 1, is byte-for-byte
    the original static-batch-1 export) or the literal ``'dynamic'``, which
    exports a dynamic batch axis on `obs` and every output (traced at
    :data:`_DYNAMIC_TRACE_BATCH`).

    After the TorchScript export, the file is reloaded and
    ``metadata_props[komugi.plane_map_version]`` is stamped with
    :data:`komugi.engine.observation.PLANE_MAP_VERSION` and re-saved -- the
    graph and initializers themselves are untouched by that roundtrip.
    """
    static = _static_batch(batch)
    example_batch = _DYNAMIC_TRACE_BATCH if static is None else static
    example = torch.zeros(
        (example_batch, factory.IN_PLANES, 9, 9), dtype=torch.float32
    )
    wrapper = _ExportWrapper(net)
    wrapper.eval()
    dynamic_axes = (
        {name: {0: "batch"} for name in (INPUT_NAME, *OUTPUT_NAMES)}
        if static is None
        else None
    )
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (example,),
            str(out),
            input_names=[INPUT_NAME],
            output_names=list(OUTPUT_NAMES),
            opset_version=OPSET_VERSION,
            dynamo=False,
            dynamic_axes=dynamic_axes,
        )
    model = onnx.load(str(out))
    onnx.helper.set_model_props(
        model,
        {PLANE_MAP_VERSION_METADATA_KEY: str(observation.PLANE_MAP_VERSION)},
    )
    onnx.save(model, str(out))


def output_shapes(batch: int | str = 1) -> dict[str, list[int]]:
    """Return the torch output shapes for `batch`, keyed by output name.

    `batch` is a static size (default 1, matching the original N=1 shapes
    exactly) or ``'dynamic'``, reported as ``-1``.
    """
    static = _static_batch(batch)
    b = -1 if static is None else static
    return {
        "policy": [b, factory.NUM_ACTIONS],
        "wdl": [b, 3],
        "moves_left": [b],
    }


@dataclass(frozen=True)
class ParityPosition:
    """One position :func:`parity_positions` visited: `phase` is ``"draft"``
    while either side still held drafting rights when the position arose,
    else ``"play"``; `observation` is ``(IN_PLANES, 9, 9)`` float32, exactly
    :meth:`komugi.engine.game.Game.observation`'s own return."""

    phase: str
    observation: np.ndarray


#: DRAFTMIN threshold :func:`parity_positions` applies to its own random
#: walk (via :func:`komugi.engine.game.draft_root_mask`). Self-play's own
#: default threshold is 0, under which a uniformly random mover ends the
#: draft on its very first ply as often as not (one legal move among dozens);
#: holding it open past a few placements is what makes "both phases" a
#: reliable property of the walk rather than a matter of seed luck.
_DRAFT_MIN_PLACEMENTS: Final[int] = 6


def parity_positions(count: int, seed: int) -> list[ParityPosition]:
    """Return `count` positions from a seeded random playout, for the
    release tooling's export-parity self-check.

    Plays uniformly random legal moves (:class:`random.Random` seeded by
    `seed`) from the Advanced start -- the format a released model plays --
    recording the position BEFORE each move (so the Advanced start itself is
    the first draft sample) until `count` positions split evenly between the
    draft and the play phase have been collected, or the game ends first (a
    :class:`ValueError`). Every move is drawn from
    :meth:`Game.valid_moves`, withheld from ending the draft early by
    :func:`~komugi.engine.game.draft_root_mask` at :data:`_DRAFT_MIN_PLACEMENTS`
    (the walk's own choice, not self-play's default of 0), so both phases
    are reliably reached. Every recorded position is encoded with
    :meth:`Game.observation`, exactly as the network is fed in self-play, so
    two calls with the same `seed` return identical positions and different
    seeds diverge as soon as their move choices do.
    """
    if count < 2:
        raise ValueError(f"parity_positions needs count >= 2, got {count}")
    needed_draft = count // 2
    needed_play = count - needed_draft

    rng = random.Random(seed)
    # FULL is the engine default; named explicitly so every game construction
    # site in the repo names its repetition rule.
    game = Game(board=board_from_fen(ADVANCED_POSITION), repetition_rule=RepetitionRule.FULL)
    draft_rows: list[np.ndarray] = []
    play_rows: list[np.ndarray] = []

    while len(draft_rows) < needed_draft or len(play_rows) < needed_play:
        obs = np.ascontiguousarray(game.observation(), dtype=np.float32)
        in_draft = game.in_draft(0) or game.in_draft(1)
        bucket, limit = (draft_rows, needed_draft) if in_draft else (play_rows, needed_play)
        if len(bucket) < limit:
            bucket.append(obs)

        # `_board` because `Game` publishes no board accessor -- the same
        # underscore read `komugi.selfplay.adapter.SearchGameState.legal_mask`
        # makes of the same attribute; `draft_root_mask` never mutates it.
        mask = draft_root_mask(
            game._board, game.valid_moves(), _DRAFT_MIN_PLACEMENTS  # noqa: SLF001
        )
        legal = np.flatnonzero(mask)
        if legal.size == 0:
            break
        action = int(rng.choice(legal.tolist()))
        game.play_action(action)

    if len(draft_rows) < needed_draft or len(play_rows) < needed_play:
        raise ValueError(
            f"parity_positions(count={count}, seed={seed}) ran out of game "
            f"before collecting {needed_draft} draft and {needed_play} play "
            f"positions (got {len(draft_rows)} draft, {len(play_rows)} play)"
        )
    return [ParityPosition("draft", obs) for obs in draft_rows] + [
        ParityPosition("play", obs) for obs in play_rows
    ]
