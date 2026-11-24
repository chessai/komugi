#!/usr/bin/env python3
"""Does material lead predict the outcome in komugi's own self-play? (KOM-243)

Against an external reference bot komugi wins short tactical games with EVEN OR
NEGATIVE material and loses or draws the long ones while bleeding about a piece
every ten plies, with its value head still reading +0.9 at an eight-piece
deficit (ticket KOM-243, read from B0+B1).  The falsifiable hypothesis that
reading suggests: in self-play both seats play the same material-indifferent
style, so **material lead does not predict the outcome in the training
distribution** and the value head never had a reason to learn it.

This script measures exactly that, on self-play shards and nothing else.  For
every row it decodes both sides' total material straight out of the observation
planes, pairs it with that row's own recorded outcome (``wdl_target``) and its
search value (``root_value``), and tabulates

  * ``P(win) / P(draw) / P(loss)`` and mean ``root_value`` per (material-lead
    band x ply band) and per lead band overall,
  * the Spearman correlation between material lead and outcome (+1/0/-1) per
    ply band,
  * the fraction of decisive games whose eventual winner was the side with the
    larger material lead at ply 100,
  * and, with ``--ckpt``, the mean value a checkpoint's own value head reports
    per lead band -- so "what the distribution says" and "what the head
    believes" can be read off the same table.

Which planes the material comes from
------------------------------------
The plane map is ``komugi.engine.observation``'s, ``PLANE_MAP_VERSION = 3``, and
the decode reads two blocks of it and no others:

* **Board pieces -- planes 0-83.**  ``plane = piece_index * 6 + (tier - 1) * 2 +
  color_index``, one-hot at the square the piece stands on.  A piece appears in
  exactly one plane whatever tier of whatever tower it sits in, so summing the
  84 planes over all 81 squares counts every piece on the board once, and the
  colour is the plane's PARITY: even planes are white's, odd planes are black's
  (``color_index``: white 0, black 1, and it is the least significant term of
  the plane formula).  Towers therefore need no special handling -- "board
  pieces across tower levels" is the plain sum.
* **Hand pieces -- planes 84-111.**  ``plane = 84 + piece_index * 2 +
  color_index``, a CONSTANT plane whose value is ``count / MAX_COUNT[type]``.
  The count is recovered by multiplying square (0, 0) of the plane by that
  type's ``MAX_COUNT`` and rounding; the planes are checked to be constant
  before they are read, so a non-constant one is a corrupted observation rather
  than a silently mis-decoded hand.
* **Side to move -- plane 114** (0.0 white, 1.0 black), which says whose numbers
  are the "mover's".  The shard's own ``to_play`` column is read too and the
  disagreements are counted into the report (``to_play_mismatch``); the decode
  itself uses the plane, so the material and the frame it is expressed in come
  from the same tensor.

Planes 112/113 (drafting rights), 115-120 (clocks, repetition, last move) and
121-127 (reserved) are deliberately untouched: none of them carries a piece.

A side can never hold more than ``PIECES_PER_SIDE`` pieces (the sum of
``observation.MAX_COUNT``, i.e. every piece the advanced initial hand contains),
so both decoded totals are asserted into ``[0, PIECES_PER_SIDE]`` and a row
outside it raises rather than being tabulated.

Frames and signs, stated once
-----------------------------
``material lead = mover - opponent``.  ``wdl_target`` is ``(win, draw, loss)``
in the ROW'S OWN MOVER's frame (``selfplay/records.py``), so the outcome
``+1/0/-1`` and ``root_value`` are already in the same frame the lead is, and
nothing here flips anything: every quantity in this report belongs to the player
who moved in that row.  With ``--ckpt`` the head value is ``w - l`` of the net's
mover-relative ``wdl``, the same frame again.

Usage
-----
::

    python scripts/analysis/material_vs_outcome.py \\
        --shards <root>/shards/000060x --out results/material_vs_outcome.json

    # ... and what the iteration's own net believes at the same material:
    ... --ckpt <root>/ema/checkpoint-000018000-<digest>.pt --device cuda

    # custom bands: ATTACHED, because the edges start with a minus sign and
    # argparse reads a detached "-20,-10,..." as an option rather than a value.
    ... --bands=-20,-10,-5,-2,2,5,10,20 --ply-bands=0,40,80,120,160,240,512

Reading it
----------
If P(win) is flat across the lead bands, self-play does not reward material and
the head had nothing to learn from -- the ticket's hypothesis survives.  If
P(win) climbs with the lead while the head's mean value does not, the
distribution rewards material and the head is the thing that failed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Protocol

import numpy as np
import numpy.typing as npt

# The repo's mypy config checks `src/komugi` from source; a standalone run over
# this script resolves `komugi` as an installed distribution instead, which ships
# no py.typed marker -- hence the per-import waivers.
from komugi.campaign.factory import (  # type: ignore[import-untyped]
    IN_PLANES,
    NUM_ACTIONS,
    new_net,
)
from komugi.engine.observation import (  # type: ignore[import-untyped]
    MAX_COUNT,
    PIECE_TYPES,
    PLANE_MAP_VERSION,
    observation_size,
)
from komugi.selfplay.inference import (  # type: ignore[import-untyped]
    InferenceConfig,
    InProcessEvaluator,
    NetEval,
)
from komugi.selfplay.worker import (  # type: ignore[import-untyped]
    TERMINATION_DECISIVE,
)
from komugi.train.checkpoint import (  # type: ignore[import-untyped]
    load_checkpoint_file,
    require_matching_plane_count,
)
from komugi.train.shards import (  # type: ignore[import-untyped]
    SHARD_GLOB,
    ShardWarning,
    read_columns,
)

# ---------------------------------------------------------------------------
# the plane map this decode reads (see the module docstring)
# ---------------------------------------------------------------------------

OCCUPANCY_PLANES: Final[int] = 84
"""Planes 0-83: piece_type(14) x tier(3) x color(2), one-hot per board piece."""

HAND_PLANE_START: Final[int] = OCCUPANCY_PLANES
"""Plane 84 -- the first hand plane."""

HAND_PLANE_STOP: Final[int] = 112
"""One past plane 111 -- the last hand plane."""

SIDE_TO_MOVE_PLANE: Final[int] = 114
"""0.0 = white to move, 1.0 = black to move."""

MAX_COUNT_BY_INDEX: Final[npt.NDArray[np.float64]] = np.asarray(
    [MAX_COUNT[piece] for piece in PIECE_TYPES], dtype=np.float64
)
"""``MAX_COUNT`` in piece-index order -- the hand planes' denominators."""

PIECES_PER_SIDE: Final[int] = int(sum(MAX_COUNT.values()))
"""Every piece one side can own at once (25): the advanced initial hand.

Board pieces and hand pieces are the same pieces at different places, so this
bounds the board total, the hand total and their sum alike, and a decode that
exceeds it is a corrupted observation rather than a large lead.
"""

WHITE: Final[int] = 0
BLACK: Final[int] = 1

WIN: Final[int] = 0
DRAW: Final[int] = 1
LOSS: Final[int] = 2
"""``wdl_target``'s columns, mover-relative (``selfplay/records.py``)."""

OUTCOME_OF_LABEL: Final[npt.NDArray[np.int64]] = np.asarray([1, 0, -1], dtype=np.int64)
"""``wdl_target`` argmax -> the +1/0/-1 the Spearman correlation is taken on."""

# ---------------------------------------------------------------------------
# defaults
# ---------------------------------------------------------------------------

DEFAULT_LEAD_BANDS: Final[tuple[int, ...]] = (-20, -10, -5, -2, 2, 5, 10, 20)
"""Lead-band EDGES: nine bands, the middle one ``-2 <= lead < 2`` ("level")."""

DEFAULT_PLY_BANDS: Final[tuple[int, ...]] = (0, 40, 80, 120, 160, 240, 512)
"""Ply-band edges.  512 is the engine's turn cap, so the last band is the
capped tail."""

LEAD_PLY: Final[int] = 100
"""The ply the "did the material leader win?" question is asked at.

100 because that is where the ticket's reference-bot reading is taken (lead -5 to -15
at ply 100 with the head at +0.9), so the two numbers are comparable.
"""

FORWARD_BATCH: Final[int] = 256
"""Positions per ``--ckpt`` forward pass."""

BASE_COLUMNS: Final[tuple[str, ...]] = (
    "game_id",
    "ply",
    "to_play",
    "root_value",
    "termination",
    "observation",
    "wdl_target",
)
"""Every column this script needs; all seven are required of a shard."""

_DAMAGE: Final[tuple[type[BaseException], ...]] = (
    OSError,
    ValueError,
    KeyError,
    EOFError,
)
"""What "this shard is unreadable" looks like, so one rotted shard skips rather
than abandoning the other 999 (``cap_game_values.read_games``'s policy)."""


class PlaneMapMismatch(ValueError):
    """A shard was written under a different plane map than this process reads.

    Fatal rather than skippable: every position such a shard carries is encoded
    in a language this decode does not speak, and decoding it anyway would
    silently produce piece counts off a different plane layout.
    """


# ---------------------------------------------------------------------------
# the decode
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Material:
    """Per-row decoded material, all arrays of length ``n``.

    Attributes:
        board: ``(n, 2)`` board pieces, column 0 white and column 1 black.
        hand: ``(n, 2)`` hand pieces, same columns.
        to_play: ``(n,)`` the mover read off plane 114.
    """

    board: npt.NDArray[np.int64]
    hand: npt.NDArray[np.int64]
    to_play: npt.NDArray[np.int64]

    @property
    def total(self) -> npt.NDArray[np.int64]:
        """``(n, 2)`` board + hand pieces per colour."""
        return self.board + self.hand

    @property
    def mover(self) -> npt.NDArray[np.int64]:
        """``(n,)`` the moving side's total material."""
        return np.where(self.to_play == WHITE, self.total[:, WHITE], self.total[:, BLACK])

    @property
    def opponent(self) -> npt.NDArray[np.int64]:
        """``(n,)`` the other side's total material."""
        return np.where(self.to_play == WHITE, self.total[:, BLACK], self.total[:, WHITE])

    @property
    def lead(self) -> npt.NDArray[np.int64]:
        """``(n,)`` mover minus opponent."""
        return self.mover - self.opponent


def decode_material(observations: npt.NDArray[Any]) -> Material:
    """Decode both sides' material out of a batch of observation planes.

    The one place this script reads the plane map; everything downstream reads
    this function's output.  See the module docstring for which planes carry
    what and why the colour is a plane parity.

    Args:
        observations: ``(n, 128, 9, 9)`` float32 observations, as the shard
            stores them.

    Returns:
        The decoded :class:`Material`.

    Raises:
        ValueError: If the shape is not the encoder's, if a hand plane is not
            constant over the board (the encoder writes them constant, so a
            varying one is damage rather than a hand), or if a decoded total
            leaves ``[0, PIECES_PER_SIDE]``.
    """
    planes = np.asarray(observations, dtype=np.float32)
    expected = observation_size()
    if planes.ndim != 4 or planes.shape[1:] != expected:
        raise ValueError(
            f"observations must be (n, {expected[0]}, {expected[1]}, {expected[2]}), "
            f"got {planes.shape!r}"
        )

    # --- board: planes 0-83, colour = plane parity ---------------------------
    occupancy = planes[:, :OCCUPANCY_PLANES]
    board = np.stack(
        (
            occupancy[:, WHITE::2].sum(axis=(1, 2, 3)),
            occupancy[:, BLACK::2].sum(axis=(1, 2, 3)),
        ),
        axis=1,
    )

    # --- hand: planes 84-111, constant, value = count / MAX_COUNT ------------
    hand_planes = planes[:, HAND_PLANE_START:HAND_PLANE_STOP]
    spread = hand_planes.max(axis=(2, 3)) - hand_planes.min(axis=(2, 3))
    if not np.all(spread <= 1e-6):
        raise ValueError(
            "a hand plane is not constant over the board; planes "
            f"{HAND_PLANE_START}-{HAND_PLANE_STOP - 1} are written constant by the "
            f"encoder, so this observation is not this plane map's"
        )
    fractions = hand_planes[:, :, 0, 0].astype(np.float64)
    counts = np.rint(fractions * np.repeat(MAX_COUNT_BY_INDEX, 2)[None, :])
    hand = np.stack(
        (counts[:, WHITE::2].sum(axis=1), counts[:, BLACK::2].sum(axis=1)), axis=1
    )

    totals = np.rint(board + hand).astype(np.int64)
    if totals.size and (totals.min() < 0 or totals.max() > PIECES_PER_SIDE):
        raise ValueError(
            f"decoded material {int(totals.min())}..{int(totals.max())} leaves "
            f"[0, {PIECES_PER_SIDE}]: one side cannot hold more pieces than the "
            f"advanced initial hand, so these planes are not this plane map's"
        )

    to_play = np.rint(planes[:, SIDE_TO_MOVE_PLANE, 0, 0]).astype(np.int64)
    return Material(
        board=np.rint(board).astype(np.int64),
        hand=np.rint(hand).astype(np.int64),
        to_play=to_play,
    )


# ---------------------------------------------------------------------------
# the forward pass (--ckpt)
# ---------------------------------------------------------------------------


class Evaluator(Protocol):
    """The one method the forward pass uses of a batch evaluator."""

    def evaluate(
        self, obs: npt.NDArray[np.float32], legal: npt.NDArray[np.bool_]
    ) -> NetEval:
        """Evaluate a batch of positions."""


def load_evaluator(path: str | os.PathLike[str], device: str = "cpu") -> Evaluator:
    """Load `path` and return an evaluator serving exactly those weights.

    ``cap_game_values.load_evaluator``'s idiom, for its reasons: verify the
    file, build the net this campaign's ``KOMUGI_CAMPAIGN_NET`` names, check the
    plane count on the built net BEFORE ``load_state_dict`` so a foreign plane
    map dies by name, place it, and ``eval()`` it -- a net left in training mode
    would update its BatchNorm statistics from the batches pushed through it and
    the second half of this report would come from a different net than the
    first.
    """
    checkpoint = load_checkpoint_file(path)
    if checkpoint is None:
        raise FileNotFoundError(
            f"{path} names no checkpoint that verifies; refusing to report what "
            f"freshly initialised weights believe under a checkpoint's name"
        )
    net = new_net()
    require_matching_plane_count(checkpoint, net.config.trunk.in_planes)
    net.load_state_dict(checkpoint.model_state)
    net.to(device)
    net.eval()
    evaluator: Evaluator = InProcessEvaluator(
        net, InferenceConfig(in_planes=IN_PLANES, num_actions=NUM_ACTIONS)
    )
    return evaluator


def head_values(
    evaluator: Evaluator,
    observations: npt.NDArray[Any],
    *,
    batch: int = FORWARD_BATCH,
) -> npt.NDArray[np.float64]:
    """Return the value head's ``w - l`` for `observations`, mover-relative.

    ``w - l`` rather than a utility: the seam carries raw win/draw/loss
    probabilities and this reader applies no contempt, so the number is
    comparable to the shard's ``root_value`` without either being pushed through
    a ``draw_score`` the other did not see.  The legality mask is all-ones for
    ``cap_game_values.forward_wdl``'s reason -- the value head does not read it,
    and re-deriving the true mask would cost a move generation per position for
    a number no output depends on.
    """
    stacked = np.asarray(observations, dtype=np.float32)
    if stacked.shape[0] == 0:
        return np.zeros((0,), dtype=np.float64)
    step = max(batch, 1)
    out: list[npt.NDArray[np.float64]] = []
    for start in range(0, stacked.shape[0], step):
        chunk = stacked[start : start + step]
        legal = np.ones((chunk.shape[0], NUM_ACTIONS), dtype=np.bool_)
        wdl = np.asarray(evaluator.evaluate(chunk, legal).wdl, dtype=np.float64)
        out.append(wdl[:, WIN] - wdl[:, LOSS])
    return np.concatenate(out)


# ---------------------------------------------------------------------------
# reading the shards
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Rows:
    """Every row that survived the read, as parallel arrays of length ``n``.

    One flat table rather than a game tree: every question this script asks --
    the bands, the correlations, even the ply-100 read -- is a group-by over
    rows, and grouping into ``GameRecord`` objects first would cost a Python
    object per row of a multi-million-row iteration for nothing.
    """

    game_id: npt.NDArray[np.int64]
    ply: npt.NDArray[np.int64]
    to_play: npt.NDArray[np.int64]
    lead: npt.NDArray[np.int64]
    mover_material: npt.NDArray[np.int64]
    opponent_material: npt.NDArray[np.int64]
    outcome: npt.NDArray[np.int64]
    root_value: npt.NDArray[np.float64]
    termination: npt.NDArray[np.int64]
    head_value: npt.NDArray[np.float64] | None
    to_play_mismatch: int
    shards: int

    def __len__(self) -> int:
        return int(self.game_id.size)


def _empty_rows(*, head: bool) -> Rows:
    """An empty table -- what an empty (or entirely unreadable) tree reads as."""
    ints = np.zeros((0,), dtype=np.int64)
    floats = np.zeros((0,), dtype=np.float64)
    return Rows(
        game_id=ints,
        ply=ints.copy(),
        to_play=ints.copy(),
        lead=ints.copy(),
        mover_material=ints.copy(),
        opponent_material=ints.copy(),
        outcome=ints.copy(),
        root_value=floats,
        termination=ints.copy(),
        head_value=floats.copy() if head else None,
        to_play_mismatch=0,
        shards=0,
    )


def _plane_map_guard(path: Path, names: Sequence[str]) -> None:
    """Raise if `path` was written under another plane map.

    A shard that declares no ``plane_map_version`` predates the column and is
    read as this process's map, which is what every other reader in the repo
    does with it.
    """
    if "plane_map_version" not in names:
        return
    _, count, columns = read_columns(path, ("plane_map_version",))
    if count == 0:
        return
    versions = np.unique(columns["plane_map_version"])
    foreign = [int(v) for v in versions if int(v) != PLANE_MAP_VERSION]
    if foreign:
        raise PlaneMapMismatch(
            f"{path} was written under plane map(s) {foreign}, this process reads "
            f"{PLANE_MAP_VERSION}: its planes do not mean what this decode reads"
        )


def read_rows(
    shard_dirs: Sequence[Path],
    *,
    evaluator: Evaluator | None = None,
    batch: int = FORWARD_BATCH,
) -> Rows:
    """Decode every row under `shard_dirs` into one flat table.

    Every ``shard-*.npz`` at or below each directory is read, at any depth (the
    layout between an iteration root and its worker directories is the loop's
    business).  A shard that cannot be read is skipped with a
    :class:`~komugi.train.shards.ShardWarning`; a shard written under another
    plane map is raised on.

    The observations are decoded -- and, with `evaluator`, forward-passed -- one
    shard at a time and then dropped: an observation column is ~41 KB a row, so
    holding a whole iteration's would cost tens of gigabytes to produce two
    ``int64`` columns.
    """
    parts: dict[str, list[npt.NDArray[Any]]] = {
        name: []
        for name in (
            "game_id",
            "ply",
            "to_play",
            "lead",
            "mover",
            "opponent",
            "outcome",
            "root_value",
            "termination",
            "head",
        )
    }
    mismatch = 0
    shards = 0
    paths = sorted(
        {path for directory in shard_dirs for path in Path(directory).rglob(SHARD_GLOB)}
    )
    for path in paths:
        try:
            schema, count, _ = read_columns(path, ())
            missing = [name for name in BASE_COLUMNS if name not in schema.names]
            if missing:
                raise ValueError(f"shard declares no {', '.join(missing)} column")
            _plane_map_guard(path, schema.names)
            if count == 0:
                continue
            _, _, columns = read_columns(path, BASE_COLUMNS)
        except PlaneMapMismatch:
            raise
        except _DAMAGE as error:
            warnings.warn(
                f"skipping unreadable shard {path}: {error}", ShardWarning, stacklevel=2
            )
            continue

        observations = columns["observation"]
        material = decode_material(observations)
        stored_to_play = np.asarray(columns["to_play"], dtype=np.int64)
        mismatch += int(np.count_nonzero(stored_to_play != material.to_play))

        targets = np.asarray(columns["wdl_target"], dtype=np.float64)
        labelled = np.all(np.isfinite(targets), axis=1)
        outcome = OUTCOME_OF_LABEL[np.argmax(targets, axis=1)]

        keep = labelled
        parts["game_id"].append(np.asarray(columns["game_id"], dtype=np.int64)[keep])
        parts["ply"].append(np.asarray(columns["ply"], dtype=np.int64)[keep])
        parts["to_play"].append(material.to_play[keep])
        parts["lead"].append(material.lead[keep])
        parts["mover"].append(material.mover[keep])
        parts["opponent"].append(material.opponent[keep])
        parts["outcome"].append(outcome[keep])
        parts["root_value"].append(
            np.asarray(columns["root_value"], dtype=np.float64)[keep]
        )
        parts["termination"].append(
            np.asarray(columns["termination"], dtype=np.int64)[keep]
        )
        if evaluator is not None:
            parts["head"].append(
                head_values(evaluator, np.asarray(observations)[keep], batch=batch)
            )
        shards += 1

    if not parts["game_id"]:
        return _empty_rows(head=evaluator is not None)

    joined = {name: np.concatenate(values) for name, values in parts.items() if values}
    return Rows(
        game_id=joined["game_id"],
        ply=joined["ply"],
        to_play=joined["to_play"],
        lead=joined["lead"],
        mover_material=joined["mover"],
        opponent_material=joined["opponent"],
        outcome=joined["outcome"],
        root_value=joined["root_value"],
        termination=joined["termination"],
        head_value=joined.get("head") if evaluator is not None else None,
        to_play_mismatch=mismatch,
        shards=shards,
    )


# ---------------------------------------------------------------------------
# bands and statistics
# ---------------------------------------------------------------------------


def band_labels(edges: Sequence[int]) -> list[str]:
    """Return one human label per band ``np.digitize(x, edges)`` can return.

    ``len(edges) + 1`` labels: everything below the first edge, each half-open
    ``[edge_i, edge_{i+1})``, and everything at or above the last.
    """
    if not edges:
        return ["all"]
    labels = [f"<{edges[0]}"]
    labels += [f"{low}..{high}" for low, high in zip(edges, edges[1:])]
    labels.append(f">={edges[-1]}")
    return labels


def band_of(values: npt.NDArray[Any], edges: Sequence[int]) -> npt.NDArray[np.int64]:
    """Return each value's band index under `edges` (half-open, left-closed)."""
    if not edges:
        return np.zeros(values.shape, dtype=np.int64)
    bands: npt.NDArray[np.int64] = np.digitize(
        values, np.asarray(edges), right=False
    ).astype(np.int64)
    return bands


def _average_ranks(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return 1-based ranks with ties averaged -- Spearman's rank transform."""
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    starts = ends - counts
    mean_rank = starts + (counts - 1) / 2.0 + 1.0
    return np.asarray(mean_rank[inverse], dtype=np.float64)


def spearman(x: npt.NDArray[Any], y: npt.NDArray[Any]) -> float | None:
    """Return Spearman's rho between `x` and `y`, or ``None`` when undefined.

    Implemented here rather than imported: it is Pearson over average ranks,
    ten lines of numpy, and the repository carries no scipy dependency for a
    diagnostic script to add one.  ``None`` (rather than NaN, which JSON cannot
    spell) when there are fewer than two points or when either side is constant
    -- a lead band in which every row has the same outcome has no correlation,
    it has a certainty, and the band's ``P(win)`` already says so.
    """
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    if a.size < 2 or a.size != b.size:
        return None
    ra = _average_ranks(a)
    rb = _average_ranks(b)
    if ra.std() == 0.0 or rb.std() == 0.0:
        return None
    rho = float(np.corrcoef(ra, rb)[0, 1])
    return None if not np.isfinite(rho) else rho


def _mean(values: npt.NDArray[Any]) -> float | None:
    """Mean of `values`, or ``None`` when there is nothing to average."""
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()) if array.size else None


def _cell(
    rows: Rows, mask: npt.NDArray[np.bool_], *, label: str, **extra: Any
) -> dict[str, Any]:
    """Return one tabulated cell: the outcome split and the two mean values."""
    outcome = rows.outcome[mask]
    n = int(outcome.size)
    cell: dict[str, Any] = {
        "band": label,
        "n": n,
        "p_win": float(np.count_nonzero(outcome == 1) / n) if n else None,
        "p_draw": float(np.count_nonzero(outcome == 0) / n) if n else None,
        "p_loss": float(np.count_nonzero(outcome == -1) / n) if n else None,
        "mean_root_value": _mean(rows.root_value[mask]),
        "mean_outcome": _mean(outcome),
    }
    if rows.head_value is not None:
        cell["mean_head_value"] = _mean(rows.head_value[mask])
    cell.update(extra)
    return cell


def lead_at_ply(rows: Rows, *, ply: int = LEAD_PLY) -> dict[str, Any]:
    """Did the side ahead on material at `ply` go on to win?

    Restricted to DECISIVE games that reached `ply` at all -- a draw has no
    winner to agree or disagree with, and a game that ended before `ply` has no
    position to read the lead off.  Games level at `ply` are counted separately
    rather than scored: "the side with the larger lead" does not name a side
    when the leads are equal, and folding them in either direction would move
    the fraction without any evidence behind it.

    The row AT `ply` is the mover's frame, and so is its ``outcome``, so the
    question is just whether the two signs agree.
    """
    decisive = (rows.termination == TERMINATION_DECISIVE) & (rows.ply == ply)
    leads = rows.lead[decisive]
    outcomes = rows.outcome[decisive]
    games = int(np.unique(rows.game_id[decisive]).size)
    level = int(np.count_nonzero(leads == 0))
    scored = leads != 0
    n = int(np.count_nonzero(scored))
    agree = int(np.count_nonzero(np.sign(leads[scored]) == np.sign(outcomes[scored])))
    return {
        "ply": ply,
        "decisive_games_reaching_ply": games,
        "level_at_ply": level,
        "scored": n,
        "leader_won": (agree / n) if n else None,
    }


def tabulate(
    rows: Rows,
    *,
    lead_edges: Sequence[int],
    ply_edges: Sequence[int],
    lead_ply: int = LEAD_PLY,
) -> dict[str, Any]:
    """Return the whole report body for `rows`."""
    lead_names = band_labels(lead_edges)
    ply_names = band_labels(ply_edges)
    lead_band = band_of(rows.lead, lead_edges)
    ply_band = band_of(rows.ply, ply_edges)

    by_lead = [
        _cell(rows, lead_band == index, label=label)
        for index, label in enumerate(lead_names)
    ]
    by_lead_ply = [
        _cell(
            rows,
            (lead_band == li) & (ply_band == pi),
            label=lead_label,
            ply_band=ply_label,
        )
        for pi, ply_label in enumerate(ply_names)
        for li, lead_label in enumerate(lead_names)
    ]
    by_ply = [
        _cell(rows, ply_band == index, label=label)
        for index, label in enumerate(ply_names)
    ]
    correlations = [
        {
            "ply_band": label,
            "n": int(np.count_nonzero(ply_band == index)),
            "spearman_lead_outcome": spearman(
                rows.lead[ply_band == index], rows.outcome[ply_band == index]
            ),
        }
        for index, label in enumerate(ply_names)
    ]
    return {
        "rows": len(rows),
        "games": int(np.unique(rows.game_id).size),
        "shards": rows.shards,
        "to_play_mismatch": rows.to_play_mismatch,
        "lead_bands": list(lead_edges),
        "ply_bands": list(ply_edges),
        "lead_band_labels": lead_names,
        "ply_band_labels": ply_names,
        "by_lead": by_lead,
        "by_ply": by_ply,
        "by_lead_ply": by_lead_ply,
        "spearman_by_ply_band": correlations,
        "spearman_overall": spearman(rows.lead, rows.outcome),
        f"lead_at_ply_{lead_ply}": lead_at_ply(rows, ply=lead_ply),
        "mean_lead": _mean(rows.lead),
        "head_values": rows.head_value is not None,
    }


# ---------------------------------------------------------------------------
# the human table
# ---------------------------------------------------------------------------


def _fmt(value: float | None, *, width: int = 7, digits: int = 3) -> str:
    """Format an optional number for the printed table."""
    return f"{'--':>{width}}" if value is None else f"{value:>{width}.{digits}f}"


def format_report(report: dict[str, Any]) -> str:
    """Return the compact table printed beside the JSON."""
    lines: list[str] = []
    lines.append(
        f"rows={report['rows']} games={report['games']} shards={report['shards']} "
        f"plane_map={report['plane_map_version']} "
        f"to_play_mismatch={report['to_play_mismatch']}"
    )
    head = report["head_values"]
    header = f"{'lead':>10} {'n':>9} {'P(win)':>7} {'P(draw)':>7} {'P(loss)':>7} {'root_v':>7}"
    if head:
        header += f" {'head_v':>7}"
    lines.append("")
    lines.append("P(outcome | material lead), mover-relative")
    lines.append(header)
    for cell in report["by_lead"]:
        line = (
            f"{cell['band']:>10} {cell['n']:>9} {_fmt(cell['p_win'])} "
            f"{_fmt(cell['p_draw'])} {_fmt(cell['p_loss'])} "
            f"{_fmt(cell['mean_root_value'])}"
        )
        if head:
            line += f" {_fmt(cell.get('mean_head_value'))}"
        lines.append(line)

    lines.append("")
    lines.append("Spearman(material lead, outcome) by ply band")
    lines.append(f"{'ply':>12} {'n':>9} {'rho':>7}")
    for cell in report["spearman_by_ply_band"]:
        lines.append(
            f"{cell['ply_band']:>12} {cell['n']:>9} "
            f"{_fmt(cell['spearman_lead_outcome'])}"
        )
    lines.append(
        f"{'overall':>12} {report['rows']:>9} {_fmt(report['spearman_overall'])}"
    )

    key = next(name for name in report if name.startswith("lead_at_ply_"))
    at_ply = report[key]
    lines.append("")
    lines.append(
        f"decisive games reaching ply {at_ply['ply']}: "
        f"{at_ply['decisive_games_reaching_ply']} "
        f"(level {at_ply['level_at_ply']}, scored {at_ply['scored']}); "
        f"material leader won: {_fmt(at_ply['leader_won'], width=5)}"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_edges(text: str) -> tuple[int, ...]:
    """Parse ``"-20,-10,-5"`` into band edges, strictly increasing."""
    try:
        edges = tuple(int(part) for part in text.split(",") if part.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"band edges must be comma-separated integers, got {text!r}: {error}"
        ) from error
    if not edges:
        raise argparse.ArgumentTypeError("band edges must name at least one edge")
    if any(high <= low for low, high in zip(edges, edges[1:])):
        raise argparse.ArgumentTypeError(
            f"band edges must be strictly increasing, got {list(edges)}"
        )
    return edges


def build_parser() -> argparse.ArgumentParser:
    """Return the CLI parser."""
    parser = argparse.ArgumentParser(
        prog="material_vs_outcome",
        description=(
            "Tabulate P(outcome) and mean value by material lead over self-play "
            "shards (KOM-243)."
        ),
    )
    parser.add_argument(
        "--shards",
        type=Path,
        nargs="+",
        required=True,
        help="one or more shard directories, read recursively",
    )
    parser.add_argument("--out", type=Path, required=True, help="where the JSON goes")
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=None,
        help="a checkpoint whose value head is forward-passed over the same rows",
    )
    parser.add_argument(
        "--device", default="cpu", choices=("cpu", "cuda"), help="device for --ckpt"
    )
    parser.add_argument(
        "--bands",
        type=parse_edges,
        default=DEFAULT_LEAD_BANDS,
        help=(
            "material-lead band edges, and they start negative, so this one "
            "takes the ATTACHED form: --bands=-20,-10,-5,-2,2,5,10,20 "
            "(argparse reads a detached '-20,...' as an option). "
            f"Default {','.join(map(str, DEFAULT_LEAD_BANDS))}"
        ),
    )
    parser.add_argument(
        "--ply-bands",
        type=parse_edges,
        default=DEFAULT_PLY_BANDS,
        help=f"ply band edges (default {','.join(map(str, DEFAULT_PLY_BANDS))})",
    )
    parser.add_argument(
        "--lead-ply",
        type=int,
        default=LEAD_PLY,
        help=f"the ply the leader-won question is asked at (default {LEAD_PLY})",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=FORWARD_BATCH,
        help=f"positions per --ckpt forward pass (default {FORWARD_BATCH})",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    evaluator_factory: Callable[[Path, str], Evaluator] = load_evaluator,
) -> int:
    """Read the shards, write the report, print the table.

    `evaluator_factory` is the test seam and the only one: everything else is
    the real reader over a real shard tree, so a test that substitutes a stub
    net still exercises the shard read, the decode and the tabulation.
    """
    args = build_parser().parse_args(argv)
    missing = [str(path) for path in args.shards if not Path(path).is_dir()]
    if missing:
        raise SystemExit(f"no such shard directory: {', '.join(missing)}")

    evaluator = (
        None if args.ckpt is None else evaluator_factory(args.ckpt, args.device)
    )
    rows = read_rows(args.shards, evaluator=evaluator, batch=args.batch)
    report: dict[str, Any] = {
        "plane_map_version": PLANE_MAP_VERSION,
        "pieces_per_side": PIECES_PER_SIDE,
        "board_planes": f"0-{OCCUPANCY_PLANES - 1} (colour = plane parity)",
        "hand_planes": (
            f"{HAND_PLANE_START}-{HAND_PLANE_STOP - 1} "
            f"(plane = {HAND_PLANE_START} + piece_index * 2 + colour; "
            f"value = count / MAX_COUNT)"
        ),
        "side_to_move_plane": SIDE_TO_MOVE_PLANE,
        "shard_dirs": [str(Path(path)) for path in args.shards],
        "ckpt": None if args.ckpt is None else str(args.ckpt),
        "device": args.device if args.ckpt is not None else None,
    }
    report.update(
        tabulate(
            rows,
            lead_edges=args.bands,
            ply_edges=args.ply_bands,
            lead_ply=args.lead_ply,
        )
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(format_report(report))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    sys.exit(main())
