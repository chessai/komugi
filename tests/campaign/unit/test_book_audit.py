"""The booked-fraction audit, over a tree with a planted preemption (KOM-014 #17).

§6.1(b)'s audit is one classification and one interval, and each half has a
failure this module is built around.

**The classification is three-way, not two** (N2). A booked game's rows begin at
``ply == L`` because the prefix is played and not recorded, so ``min(ply) > 0``
looks like booking -- and it is *also* what a preemption looks like: a shard
sealed after a kill leaves a game whose first rows are simply gone
(``ShardReader`` calls one entry after a kill the expected shape of one). An
audit that read booked-ness off ``min(ply) > 0`` would count that game as
booked, fabricate an ``L`` for it, and report a booking-rate anomaly where the
truth is data loss. So the planted tree carries a game whose rows start at
**ply 37**, and the assertions pin it into ``truncated`` and out of *both* the
numerator and the denominator.

**The interval is exact, not normal-approximate.** The design's own argument for
replacing the ticket's ``[8%, 12%]`` gate is arithmetic about small ``n``: at
``n = 200`` that band is ±1σ and fails about a third of honest runs, while at
``n = 1500`` the exact 99% interval fits inside it. Both halves are asserted
here, because an audit whose interval was wrong in either direction would make
§9's R4 unfalsifiable rather than merely noisy.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Final

import pytest

pytest.importorskip("torch")

import torch

from komugi.campaign.book_audit import booked_fraction, clopper_pearson
from komugi.engine.game import Game
from komugi.engine.observation import CONTROL_EMPTY, CONTROL_SIZE, PLANE_MAP_VERSION
from komugi.selfplay.book import MAX_PLIES, MIN_PLIES
from komugi.selfplay.records import SCHEMA, num_actions
from komugi.train.shards import ShardWriter

_UNBOOKED: Final[int] = 45
_BOOKED: Final[int] = 5
"""Five booked in fifty classified games: the planted fraction is exactly 0.10,
so the audit's answer is a number the fixture *knows* rather than one it reads
back from the code under test."""

_BOOKED_L: Final[int] = 6
"""The planted prefix length, inside ``[MIN_PLIES, MAX_PLIES]``."""

_TRUNCATED_FIRST_PLY: Final[int] = 37
"""Where the preempted game's surviving rows start -- **outside** ``[4, 12]``,
which is the only case ``min(ply)`` alone can adjudicate honestly (R9 records
the case it cannot: a booked game that also lost rows *within* its own prefix
range reads as booked with a fabricated ``L``)."""

_ROWS_PER_GAME: Final[int] = 3
_SAMPLES_PER_SHARD: Final[int] = 64
_TICKET_BAND: Final[tuple[float, float]] = (0.08, 0.12)


def _row(game_id: int, ply: int) -> dict[str, torch.Tensor]:
    """Return one schema-shaped row.

    Only ``game_id`` and ``ply`` carry meaning: the audit projects exactly those
    two ``int64`` columns and reads nothing else, which is the property that
    makes it affordable over a whole campaign's shards.
    """
    return {
        "observation": torch.zeros(Game.observation_size(), dtype=torch.float32),
        "policy_target": torch.zeros(num_actions(), dtype=torch.float32),
        "wdl_target": torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32),
        "mlh_target": torch.tensor(0.5, dtype=torch.float32),
        "policy_kl": torch.tensor(0.0, dtype=torch.float32),
        "root_value": torch.tensor(0.0, dtype=torch.float32),
        "search_mode": torch.tensor(0, dtype=torch.int64),
        "to_play": torch.tensor(ply % 2, dtype=torch.int64),
        "ply": torch.tensor(ply, dtype=torch.int64),
        "game_id": torch.tensor(game_id, dtype=torch.int64),
        "sims_used": torch.tensor(32, dtype=torch.int64),
        "plane_map_version": torch.tensor(PLANE_MAP_VERSION, dtype=torch.int64),
        "final_control": torch.full((CONTROL_SIZE,), CONTROL_EMPTY, dtype=torch.int64),
        "action": torch.tensor(0, dtype=torch.int64),
        # KOM-005b's three columns, at the real shapes. The audit reads
        # neither, so they carry distinguishable values rather than zeros: a
        # zero-fill would make a dropped column look exactly like a present
        # one to any future assertion added here.
        "termination": torch.tensor(0, dtype=torch.int64),
        "short_term_value": torch.tensor(
            [0.5, -0.25, 0.75], dtype=torch.float32
        ),
        "draft_outcome": (
            torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
            if ply < 2
            else torch.zeros(3, dtype=torch.float32)
        ),
    }


def _plant(
    root: Path, iteration: int, games: Iterable[tuple[int, Sequence[int]]]
) -> None:
    """Write ``(game_id, plies)`` pairs into ``<root>/iter_<i>/shards/worker_0_r0``."""
    directory = root / f"iter_{iteration}" / "shards" / "worker_0_r0"
    with ShardWriter(
        directory,
        SCHEMA,
        samples_per_shard=_SAMPLES_PER_SHARD,
        flush_every_seconds=86_400.0,
    ) as writer:
        for game_id, plies in games:
            for ply in plies:
                writer.append(_row(game_id, ply))


@pytest.fixture(scope="module")
def planted(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A campaign root whose iteration 0 holds all three kinds of game."""
    root = tmp_path_factory.mktemp("book-audit")
    games: list[tuple[int, Sequence[int]]] = [
        (index, list(range(_ROWS_PER_GAME))) for index in range(_UNBOOKED)
    ]
    games.extend(
        (
            1_000 + index,
            list(range(_BOOKED_L, _BOOKED_L + _ROWS_PER_GAME)),
        )
        for index in range(_BOOKED)
    )
    games.append(
        (
            9_999,
            list(range(_TRUNCATED_FIRST_PLY, _TRUNCATED_FIRST_PLY + _ROWS_PER_GAME)),
        )
    )
    _plant(root, 0, games)
    return root


def test_the_audit_separates_booked_from_truncated_games(planted: Path) -> None:
    """#17: booked, unbooked and truncated are three answers, not two."""
    audit = booked_fraction(planted, [0])

    assert audit.booked == _BOOKED
    assert audit.unbooked == _UNBOOKED
    assert audit.truncated == 1, (
        "the game whose rows start at ply 37 is a preemption, not a booking; an "
        "audit reading booked-ness off min(ply) > 0 would have counted it booked"
    )
    assert audit.classified == _UNBOOKED + _BOOKED
    assert audit.fraction == pytest.approx(
        _BOOKED / (_UNBOOKED + _BOOKED)
    )
    # The truncated game is in NEITHER the numerator nor the denominator: a
    # data-loss signal and a booking-rate signal must not be added together.
    assert audit.booked + audit.unbooked + audit.truncated == _UNBOOKED + _BOOKED + 1
    assert audit.classified != _UNBOOKED + _BOOKED + 1

    low, high = audit.interval
    assert low <= audit.fraction <= high
    assert 0.0 <= low < high <= 1.0

    # The planted L is inside the window the classification uses, which is what
    # makes the booked games *booked* rather than a second kind of truncation.
    assert MIN_PLIES <= _BOOKED_L <= MAX_PLIES
    assert not MIN_PLIES <= _TRUNCATED_FIRST_PLY <= MAX_PLIES


def test_an_iteration_with_no_shards_classifies_nothing(tmp_path: Path) -> None:
    """A root with no data is 0 games and the vacuous interval, not a crash."""
    audit = booked_fraction(tmp_path, [0, 1])
    assert (audit.booked, audit.unbooked, audit.truncated) == (0, 0, 0)
    assert audit.classified == 0
    assert audit.fraction == 0.0
    assert audit.interval == (0.0, 1.0)


def test_the_reported_interval_is_the_exact_binomial_one() -> None:
    """§6.1's arithmetic, asserted: ``[8%, 12%]`` is a gate only from n >= 1500.

    At ``n = 1500`` the interval around a 10% observation contains 0.10 and is
    of the band's own order -- which is what makes the ticket's ``[8%, 12%]``
    usable as a gate there. At ``n = 200`` it runs **outside the band at both
    ends**, so gating on the band at production per-iteration sizes would fail
    honest runs; that is §6.1's whole argument, and it is an argument only an
    exact interval can make.

    The 1500-side assertion is deliberately "contains 0.10 and is much
    narrower", not "fits inside ``[8%, 12%]``": the exact interval is
    conservative by construction, so at ``n = 1500`` its endpoints sit *at* the
    band's to within a thousandth, and pinning containment would be pinning a
    rounding rather than the statistic.
    """
    low_1500, high_1500 = clopper_pearson(150, 1500)
    assert low_1500 < 0.10 < high_1500
    assert 0.07 < low_1500 and high_1500 < 0.13

    low_200, high_200 = clopper_pearson(20, 200)
    assert low_200 < _TICKET_BAND[0], (
        "the exact 99% interval at n=200 must reach below 8%: if it did not, "
        "the ticket's band would be a legitimate gate at 200 games and 6.1's "
        "correction would be unnecessary"
    )
    assert high_200 > _TICKET_BAND[1]
    # ~sqrt(1500/200) = 2.7x narrower at the larger n; 2x is the claim that
    # survives the exact interval's conservatism at both ends.
    assert (high_200 - low_200) > 2.0 * (high_1500 - low_1500)

    # The degenerate ends are answered, not approximated: 0 successes has a
    # lower bound of exactly 0, and n successes an upper bound of exactly 1.
    assert clopper_pearson(0, 100)[0] == 0.0
    assert clopper_pearson(100, 100)[1] == 1.0
