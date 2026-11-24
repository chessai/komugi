"""Synthetic self-play rows and the tiny net the behavioural suite trains.

Not a test module -- the four ``test_*.py`` files here import it as
``synthetic`` (the same sibling-import pattern
``tests/net/behavior/trunk_probes.py`` and
``tests/selfplay/behavior/worker_rig.py`` already use).

The rows are deliberately *not* random. Every field is a pure function of the
row index, so the whole dataset is reproducible without threading a generator
through the suite, and -- the point of the exercise -- the targets are a
**deterministic function of the observation**:

* ``policy_target`` is one-hot at ``rule_square(index)``, and plane 0 of the
  observation is hot at exactly that square. The policy head is a 1x1
  convolution over board squares whose flat index *is* the action id (see
  :mod:`komugi.net.heads`), so "put your mass where plane 0 is hot" is a
  spatially aligned, convolution-learnable rule rather than an arbitrary
  lookup the net would have to memorise through a bottleneck.
* ``wdl_target`` is one-hot at ``rule_outcome(index)``, and the constant plane
  ``1 + outcome`` is raised across the whole board. The WDL head sees the
  board only through a (mean, max) pooling, and a constant plane is precisely
  what survives that.

A net that cannot learn *these* two rules is not failing at Gungi; it is
failing at gradient descent, which is what makes them a usable oracle for
"the training step works".

The field names and dtypes are :data:`komugi.selfplay.records.SCHEMA`'s, so a
row here is shaped like a row a real worker emits (the shapes are toy: 4
planes and 81 actions instead of 128 and 37341). ``test_batch_from_shards``
pins that correspondence against the real schema rather than trusting this
docstring.
"""

from __future__ import annotations

from typing import Final

import torch

from komugi.engine.observation import (
    CONTROL_EMPTY,
    CONTROL_SIZE,
    PLANE_MAP_VERSION,
)
from komugi.modes import MODE_FULL
from komugi.net import HeadsConfig, KomugiNet, NetConfig, TrunkConfig
from komugi.terminations import TERMINATION_DECISIVE

IN_PLANES: Final[int] = 4
"""Toy observation depth: plane 0 carries the policy rule, planes 1-3 the WDL one."""

BOARD: Final[int] = 9
ACTIONS: Final[int] = BOARD * BOARD
"""81 actions, i.e. ``policy_planes=1`` -- action id == board square."""

WDL_SIZE: Final[int] = 3
CHANNELS: Final[int] = 8
NET_SEED: Final[int] = 7

_NOISE: Final[float] = 0.05
"""Scale of the per-row noise. Well under the hot cell's 1.0, so the rule is
never ambiguous, but non-zero so no two rows are the same tensor."""

_NOISE_SEED: Final[int] = 4_100
_SQUARE_STRIDE: Final[int] = 7
"""Coprime with 81, so the first 81 rows land on 81 distinct squares."""


def tiny_net(seed: int = NET_SEED) -> KomugiNet:
    """Return the one-block, 81-action net the whole suite trains.

    Small enough that a hundred steps cost about a second, and built from a
    config rather than from a hand-rolled ``nn.Module`` so the suite exercises
    the real :class:`~komugi.net.model.KomugiNet` head layout -- including the
    separate ``soft_policy`` head whose gradient this wave exists to restore.
    """
    trunk = TrunkConfig(
        in_planes=IN_PLANES, channels=CHANNELS, blocks=1, gpool_every=3, seed=seed
    )
    heads = HeadsConfig(
        channels=CHANNELS, num_actions=ACTIONS, policy_planes=1, seed=seed
    )
    return KomugiNet(NetConfig(trunk=trunk, heads=heads))


def rule_square(index: int) -> int:
    """Return the board square -- and hence the action id -- row `index` wants."""
    return (index * _SQUARE_STRIDE) % ACTIONS


def rule_outcome(index: int) -> int:
    """Return row `index`'s WDL class: 0 win, 1 draw, 2 loss."""
    return index % WDL_SIZE


def expected_actions(count: int) -> torch.Tensor:
    """Return the ``(count,)`` int64 vector of the rule's answers."""
    return torch.tensor(
        [rule_square(index) for index in range(count)], dtype=torch.int64
    )


def toy_row(
    index: int,
    *,
    policy_kl: float = 0.0,
    search_mode: int = MODE_FULL,
    termination: int = TERMINATION_DECISIVE,
) -> dict[str, torch.Tensor]:
    """Return one synthetic sample, carrying every :data:`SCHEMA` field name.

    `policy_kl` is the recorded surprise scalar; it defaults to 0 (search
    agreed with the prior), and ``test_recipe_and_weights`` raises it on a
    single row to make that row's contribution visible.

    `search_mode` is the stratum KOM-036's A/B is keyed on. It defaults to
    :data:`~komugi.modes.MODE_FULL`, where the fast-row weight is exactly 1 and
    every arm is the same batch; ``test_fast_policy_weight`` mixes the two
    modes, which is the only way to see the arm at all.

    `termination` is the stratum KOM-160's ``cap_row_value_weight`` is keyed on,
    and it works the same way: it defaults to
    :data:`~komugi.terminations.TERMINATION_DECISIVE`, where the cap-row weight
    is exactly 1 and every arm is the same batch, and
    ``test_cap_row_value_weight`` mixes the two codes.

    It is an ARGUMENT rather than the ``1 if outcome == 1 else 0`` this column
    was derived by before KOM-160, and the change is deliberate: `outcome` also
    fixes ``wdl_target``, so the derived form tied "this row is a draw row" to
    "this row's label is draw" and no caller could vary one without the other.
    Two different claims -- ``the arm reaches the label`` and ``the arm reaches
    the stratum`` -- would then have been indistinguishable. The default is the
    identity arm, so every other node here is unmoved by the change.
    """
    generator = torch.Generator().manual_seed(_NOISE_SEED + index)
    square = rule_square(index)
    outcome = rule_outcome(index)

    observation = (
        torch.randn(IN_PLANES, BOARD, BOARD, generator=generator, dtype=torch.float32)
        * _NOISE
    )
    observation[0, square // BOARD, square % BOARD] += 1.0
    observation[1 + outcome] += 1.0

    policy_target = torch.zeros(ACTIONS, dtype=torch.float32)
    policy_target[square] = 1.0
    wdl_target = torch.zeros(WDL_SIZE, dtype=torch.float32)
    wdl_target[outcome] = 1.0

    return {
        "observation": observation,
        "policy_target": policy_target,
        "wdl_target": wdl_target,
        "mlh_target": torch.tensor((index % 16) / 16.0, dtype=torch.float32),
        "policy_kl": torch.tensor(policy_kl, dtype=torch.float32),
        "root_value": torch.tensor(0.0, dtype=torch.float32),
        "search_mode": torch.tensor(search_mode, dtype=torch.int64),
        "to_play": torch.tensor(index % 2, dtype=torch.int64),
        "ply": torch.tensor(index, dtype=torch.int64),
        "game_id": torch.tensor(index // 8, dtype=torch.int64),
        "sims_used": torch.tensor(32, dtype=torch.int64),
        # The observation above is toy in SHAPE, but the row still claims the
        # running plane map: a row that claimed another version is what
        # ``ConcatSource`` refuses (KOM-002 §4), and these rows are read
        # through it.
        "plane_map_version": torch.tensor(PLANE_MAP_VERSION, dtype=torch.int64),
        # KOM-010's ownership target. Nothing here learns from it -- the two
        # rules above are the oracle -- but the field is part of the row a
        # worker emits, and ``test_the_toy_rows_carry_the_real_schema_field_names``
        # holds these keys to :data:`SCHEMA`'s. "All empty" is the one value
        # that is honest for a board that does not exist.
        "final_control": torch.full(
            (CONTROL_SIZE,), CONTROL_EMPTY, dtype=torch.int64
        ),
        # KOM-014: the move played from this row's position. Distinct per row
        # rather than zero-filled, so a consumer that drops the column is
        # visible: ``square`` is already the row's one interesting action id.
        "action": torch.tensor(square, dtype=torch.int64),
        # KOM-005b's three staged-head columns. ``short_term_value`` and
        # ``draft_outcome`` carry DISTINGUISHABLE values rather than zeros:
        # a carried-column test cannot tell a real column from a zero-fill, so
        # a consumer that drops one of these would look identical to one that
        # reads it. Nothing here learns from them -- the two rules above remain
        # the oracle. ``termination`` is the caller's (KOM-160, see above); a
        # row whose stratum a test is varying cannot also be deriving it.
        "termination": torch.tensor(termination, dtype=torch.int64),
        "short_term_value": torch.tensor(
            [
                (index % 5) / 4.0 - 0.5,
                (index % 7) / 6.0 - 0.5,
                (index % 9) / 8.0 - 0.5,
            ],
            dtype=torch.float32,
        ),
        # Nonzero on the rows an in-draft flag would be true for, zero
        # elsewhere, so the column exercises both halves of the mask.
        "draft_outcome": (
            wdl_target.clone()
            if index % 3 == 0
            else torch.zeros(WDL_SIZE, dtype=torch.float32)
        ),
    }


def toy_rows(
    count: int, *, hot_kl_row: int | None = None, hot_kl: float = 5.0
) -> list[dict[str, torch.Tensor]]:
    """Return `count` rows; row `hot_kl_row` (if given) carries `hot_kl`.

    The returned list satisfies :class:`~komugi.train.window.SampleSource`
    structurally -- ``__len__`` and integer ``__getitem__`` -- which is what
    the window's own docstring says a list of dicts is for.
    """
    return [
        toy_row(index, policy_kl=hot_kl if index == hot_kl_row else 0.0)
        for index in range(count)
    ]


def observations(rows: list[dict[str, torch.Tensor]]) -> torch.Tensor:
    """Return the ``(n, planes, 9, 9)`` observation batch of `rows`."""
    return torch.stack([row["observation"] for row in rows])


def parameter_snapshot(net: KomugiNet) -> dict[str, torch.Tensor]:
    """Return a detached clone of every trainable parameter, by name.

    Parameters only, never buffers: a BatchNorm's running statistics move on
    every forward pass whether or not a gradient ever reached the module, so
    comparing ``state_dict()`` would report a head as "learning" when it is
    merely being *evaluated*. That distinction is the whole content of the
    soft-policy assertion in ``test_learning_signal``.
    """
    return {name: tensor.detach().clone() for name, tensor in net.named_parameters()}


def moved_parameters(
    net: KomugiNet, before: dict[str, torch.Tensor]
) -> frozenset[str]:
    """Return the names of the parameters that differ from `before`."""
    return frozenset(
        name
        for name, tensor in net.named_parameters()
        if not torch.equal(tensor.detach(), before[name])
    )
