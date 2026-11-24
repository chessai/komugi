"""KOM-102 -- property-style invariants over the KOM-002 observation planes.

The companion suite to ``test_observation_goldens.py``: where that file pins
*these* positions to *these* digests, this one states what must hold at **every**
position, so a plane-map change that is uniform across the goldens still has
something to fail.  The invariants are read off
``docs/design/kom-002-input-planes-v2.md`` §2's plane table and §3's semantics,
and each is checked over three sources of positions:

* every ply of G1's replay (the repetition / progress golden),
* every ply of G2's replay (the last-move golden, three segments), and
* a fixed, RNG-free sample of the oracle corpora under
  ``tests/engine/goldens/playouts_*.json``, encoded as *bare* positions through
  ``observation_from_cyboard`` -- which is the path §5.1 gives "no last move, no
  progress, first occurrence" defaults for, and the only way to reach intro,
  beginner and intermediate mode material at all.

The four groups, in the order the ticket asks for them:

a. **Domains and cross-plane agreement** -- every plane's value set, the hand
   planes against the FEN's hands, the occupancy planes against the FEN's
   towers, and the planes that may never be set together.
b. **Towers at every tier that occurs**, with G2 required to exercise tiers 1,
   2 and 3 -- an invariant suite that only ever saw flat boards would be
   asserting nothing about the tier dimension of §2's occupancy formula.
c. **The occurrence 1/2/3 encoding** against the repaired fourfold rule, over
   G1's shuffle plies: ``{R2, R3}`` is a bijection onto the live occurrence
   classes because occurrence 4 ends the game (§3.1).
d. **A mis-shaped observation is refused** by the seam every self-play row
   passes through -- ``records.SCHEMA``, whose ``validate`` is what
   ``ShardWriter.append`` calls (``shards.py:462``) -- and the refusal names the
   shape it wanted.

The golden replays are shared with the goldens module: :func:`replay` there is
cached, so importing it here costs no second replay.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt
import pytest
import torch

from komugi.engine import observation
from komugi.engine.cyboard import board_from_fen
from komugi.engine.fen import ADVANCED_POSITION
from komugi.engine.game import Game
from komugi.selfplay import records

from test_observation_goldens import (  # the replay harness, shared and cached
    G1_NAME,
    G2_NAME,
    MAX_TURNS,
    N_PLANES,
    PLANE_LAST_MOVE_FROM,
    PLANE_LAST_MOVE_TO,
    PLANE_PROGRESS,
    PLANE_R2,
    PLANE_R3,
    PLANE_RESERVED_START,
    PLANE_TURN_CLOCK,
    Golden,
    constant_value,
    occupancy_plane,
    parse_hands,
    parse_towers,
    replay,
)

ENGINE_GOLDENS: Final[Path] = (
    Path(__file__).resolve().parents[1] / "engine" / "goldens"
)

#: The oracle corpora, all four modes plus the betray harvest -- the last is
#: where tall, colour-mixed towers live (its MANIFEST records 223 betray plies
#: out of 1200 games), which is the material the tier invariants need.
CORPORA: Final[tuple[str, ...]] = (
    "playouts_intro.json",
    "playouts_beginner.json",
    "playouts_intermediate.json",
    "playouts_advanced.json",
    "playouts_betray.json",
)

#: How many positions to draw from each corpus.  A fixed stride over the file's
#: own order, so the sample is a pure function of what is checked in: no seed,
#: no shuffle, nothing to reproduce.
CORPUS_SAMPLE: Final[int] = 16

#: §2's hand block divides by the advanced-init count of each piece type
#: (``count / MAX_COUNT[ti]``).  Transcribed from ``ADVANCED_POSITION``'s hand
#: field ``M1G1I1J2W2N3R2S2F2D4C1A2K1T1`` in the occupancy ordering, and checked
#: against that FEN below rather than trusted.
MAX_HAND_COUNT: Final[tuple[int, ...]] = (1, 1, 1, 2, 2, 3, 2, 2, 2, 4, 1, 2, 1, 1)

OCCUPANCY_PLANES: Final[int] = 84
PLANE_DRAFT_WHITE: Final[int] = 112
PLANE_DRAFT_BLACK: Final[int] = 113
PLANE_SIDE_TO_MOVE: Final[int] = 114


@dataclass(frozen=True)
class Position:
    """One encoded position, with the FEN the invariants are checked against."""

    where: str
    fen: str
    obs: npt.NDArray[np.float32]


def golden_positions(golden: Golden) -> tuple[Position, ...]:
    """Every replayed ply of *golden* as a :class:`Position`."""
    return tuple(
        Position(
            where=f"{golden.name} segment {entry.segment} ply {entry.ply}",
            fen=entry.fen,
            obs=entry.obs,
        )
        for entry in golden.plies
    )


@lru_cache(maxsize=None)
def corpus_positions() -> tuple[Position, ...]:
    """A fixed sample of the oracle corpora, encoded as bare positions.

    ``observation_from_cyboard`` with no history arguments is §5.1's "bare
    position" path: repetition count 1, no last move, progress 0.  These
    positions therefore exercise the occupancy, hand and draft blocks over
    material the two goldens never reach -- intro and beginner towers, and the
    betray harvest's mixed-colour towers -- while pinning the history planes at
    the defaults §3.1 and §3.3 give a FEN.
    """
    positions: list[Position] = []
    for name in CORPORA:
        with (ENGINE_GOLDENS / name).open(encoding="utf-8") as handle:
            payload: dict[str, Any] = json.load(handle)
        fens = [
            str(ply["fen_before"]) for game in payload["games"] for ply in game
        ]
        stride = max(1, len(fens) // CORPUS_SAMPLE)
        for index, fen in enumerate(fens[::stride][:CORPUS_SAMPLE]):
            obs: npt.NDArray[np.float32] = np.ascontiguousarray(
                observation.observation_from_cyboard(board_from_fen(fen)),
                dtype=np.float32,
            )
            positions.append(Position(where=f"{name}[{index}]", fen=fen, obs=obs))
    assert positions, "no oracle corpus positions were sampled"
    return tuple(positions)


@pytest.fixture(scope="session")
def g1() -> Golden:
    """G1, replayed once per session (shared with the goldens module)."""
    return replay(G1_NAME)


@pytest.fixture(scope="session")
def g2() -> Golden:
    """G2, replayed once per session (shared with the goldens module)."""
    return replay(G2_NAME)


@pytest.fixture(scope="session")
def positions(g1: Golden, g2: Golden) -> tuple[Position, ...]:
    """Both golden replays plus the sampled oracle corpora, in that order."""
    return golden_positions(g1) + golden_positions(g2) + corpus_positions()


# ---------------------------------------------------------------------------
# (a) Value domains
# ---------------------------------------------------------------------------


def tier_values() -> set[float]:
    """§2: the last-move planes carry ``tier / 3`` with tiers 1..3, or 0."""
    return {float(np.float32(tier / 3.0)) for tier in range(4)}


def test_every_plane_value_lies_in_its_declared_domain(
    positions: tuple[Position, ...],
) -> None:
    """Each block of §2's table admits exactly the values the table declares.

    The blocks and their domains, in order: occupancy is binary and spatial;
    hand is a constant plane of ``count / MAX_COUNT``; draft and side-to-move
    are constant and binary; the two clocks are constant multiples of
    ``1 / MAX_TURNS``; the repetition thresholds are constant and binary; the
    last-move planes carry ``tier / 3`` on **at most one** cell each; and the
    reserved block is identically zero.
    """
    binary = {0.0, 1.0}
    tiers = tier_values()
    for position in positions:
        obs = position.obs
        where = position.where
        assert obs.shape == (N_PLANES, 9, 9), f"{where}: shape {obs.shape}"
        assert obs.dtype == np.float32, f"{where}: dtype {obs.dtype}"

        occupancy = {float(value) for value in np.unique(obs[0:OCCUPANCY_PLANES])}
        assert occupancy <= binary, f"{where}: occupancy values {occupancy}"

        for type_index in range(14):
            for colour in range(2):
                plane = OCCUPANCY_PLANES + type_index * 2 + colour
                value = constant_value(obs, plane)
                assert 0.0 <= value <= 1.0, f"{where}: hand plane {plane} = {value}"
                scaled = value * MAX_HAND_COUNT[type_index]
                assert scaled == pytest.approx(round(scaled)), (
                    f"{where}: hand plane {plane} is not a multiple of "
                    f"1/{MAX_HAND_COUNT[type_index]}"
                )

        for plane in (PLANE_DRAFT_WHITE, PLANE_DRAFT_BLACK, PLANE_SIDE_TO_MOVE):
            assert constant_value(obs, plane) in binary, f"{where}: plane {plane}"

        for plane in (PLANE_TURN_CLOCK, PLANE_PROGRESS):
            value = constant_value(obs, plane)
            assert 0.0 <= value <= 1.0, f"{where}: plane {plane} = {value}"
            assert (value * MAX_TURNS).is_integer(), (
                f"{where}: plane {plane} = {value} is not a multiple of "
                f"1/{MAX_TURNS}"
            )

        for plane in (PLANE_R2, PLANE_R3):
            assert constant_value(obs, plane) in binary, f"{where}: plane {plane}"

        for plane in (PLANE_LAST_MOVE_FROM, PLANE_LAST_MOVE_TO):
            values = {float(cell) for cell in np.unique(obs[plane])}
            assert values <= tiers, f"{where}: plane {plane} values {values}"
            assert int(np.count_nonzero(obs[plane])) <= 1, (
                f"{where}: plane {plane} has more than one nonzero cell"
            )

        assert not obs[PLANE_RESERVED_START:N_PLANES].any(), (
            f"{where}: a reserved plane is nonzero"
        )


def test_hand_planes_match_the_hands_the_fen_declares(
    positions: tuple[Position, ...],
) -> None:
    """Plane ``84 + ti * 2 + ci`` is the constant ``count / MAX_COUNT[ti]``.

    The divisor table is the advanced-init count of each piece type, which is
    checked here against ``ADVANCED_POSITION`` itself so a transcription slip
    cannot make the rest of this test vacuous.
    """
    advanced = parse_hands(ADVANCED_POSITION)
    assert MAX_HAND_COUNT == tuple(
        advanced[(type_index, 0)] for type_index in range(14)
    )

    for position in positions:
        hands = parse_hands(position.fen)
        for type_index in range(14):
            for colour in range(2):
                plane = OCCUPANCY_PLANES + type_index * 2 + colour
                count = hands.get((type_index, colour), 0)
                expected = float(np.float32(count / MAX_HAND_COUNT[type_index]))
                assert constant_value(position.obs, plane) == expected, (
                    f"{position.where}: hand plane {plane} should hold "
                    f"{count}/{MAX_HAND_COUNT[type_index]}"
                )


def test_occupancy_planes_reproduce_the_towers_the_fen_declares(
    positions: tuple[Position, ...],
) -> None:
    """The set of set occupancy cells is exactly what the FEN's towers say.

    §2's formula ``plane = ti * 6 + (tier - 1) * 2 + ci`` and the orientation
    convention together make this an equality rather than a sampling: every
    piece of every tower, at its own tier, and nothing else.  It subsumes "the
    occupancy planes sum to the piece count", which is asserted alongside it
    because that is the form a missing tier would be read in.
    """
    for position in positions:
        expected: set[tuple[int, int, int]] = set()
        pieces = 0
        for square, tower in parse_towers(position.fen).items():
            rank, column = divmod(square, 9)
            for tier, (type_index, colour) in enumerate(tower, start=1):
                expected.add((occupancy_plane(type_index, tier, colour), rank, column))
                pieces += 1
        actual = {
            (int(plane), int(rank), int(column))
            for plane, rank, column in zip(
                *np.nonzero(position.obs[0:OCCUPANCY_PLANES])
            )
        }
        assert actual == expected, (
            f"{position.where}: occupancy-only {sorted(actual - expected)}, "
            f"fen-only {sorted(expected - actual)}"
        )
        assert int(np.count_nonzero(position.obs[0:OCCUPANCY_PLANES])) == pieces


def tier_occupancy(obs: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """``(3, 9, 9)`` -- how many pieces sit at each tier of each square."""
    return np.stack(
        [
            obs[
                [
                    occupancy_plane(type_index, tier, colour)
                    for type_index in range(14)
                    for colour in range(2)
                ]
            ].sum(axis=0)
            for tier in (1, 2, 3)
        ]
    )


def test_mutually_exclusive_planes_are_never_co_set(
    positions: tuple[Position, ...],
) -> None:
    """The plane pairs §2 and §3 make unconstructible are never both set.

    Four of them:

    * **one piece per (square, tier)** -- the 28 occupancy planes of a tier
      partition that tier, so their sum is at most 1 everywhere;
    * **R3 implies R2** -- §3.1's thresholds are monotone, so ``(0, 1)`` is not
      a codeword;
    * **origin set with destination clear** -- §3.2 states the fourth row of
      the last-move decode table is unconstructible;
    * **plane 120 ≤ plane 115** -- §3.3's normalisation shares plane 115's
      denominator and ``progress_plies <= turns`` by construction.
    """
    for position in positions:
        obs = position.obs
        where = position.where

        per_tier = tier_occupancy(obs)
        assert float(per_tier.max()) <= 1.0, (
            f"{where}: two pieces encoded at one (square, tier)"
        )

        r2 = constant_value(obs, PLANE_R2)
        r3 = constant_value(obs, PLANE_R3)
        assert not (r3 == 1.0 and r2 == 0.0), f"{where}: R3 set without R2"

        origin = bool(obs[PLANE_LAST_MOVE_FROM].any())
        destination = bool(obs[PLANE_LAST_MOVE_TO].any())
        assert not (origin and not destination), (
            f"{where}: plane 118 set with plane 119 clear"
        )

        assert bool((obs[PLANE_PROGRESS] <= obs[PLANE_TURN_CLOCK]).all()), (
            f"{where}: the progress clock exceeds the turn clock"
        )


# ---------------------------------------------------------------------------
# (b) Towers, at every tier that occurs
# ---------------------------------------------------------------------------


def test_tower_tiers_are_contiguous_and_g2_exercises_tiers_one_to_three(
    positions: tuple[Position, ...], g2: Golden
) -> None:
    """A tier is occupied only above an occupied tier, and G2 reaches tier 3.

    Contiguity is what makes the tier dimension of §2's occupancy formula
    checkable without a second board reader: a tower has no holes, so the
    tier-2 cells are a subset of the tier-1 cells and the tier-3 cells a subset
    of the tier-2 ones.  The coverage half is the part that keeps this suite
    honest -- an invariant that only ever saw tier 1 would pass on an encoder
    that dropped tiers 2 and 3 entirely, so G2's replay is required to exhibit
    all three.
    """
    for position in positions:
        per_tier = tier_occupancy(position.obs)
        assert bool((per_tier[1] <= per_tier[0]).all()), (
            f"{position.where}: a tier-2 piece with nothing at tier 1"
        )
        assert bool((per_tier[2] <= per_tier[1]).all()), (
            f"{position.where}: a tier-3 piece with nothing at tier 2"
        )

    exercised: set[int] = set()
    for entry in g2.plies:
        for tower in parse_towers(entry.fen).values():
            exercised.update(range(1, len(tower) + 1))
    assert exercised == {1, 2, 3}, (
        f"G2's replay exercises tiers {sorted(exercised)}, not 1..3"
    )

    tiers_in_planes: set[int] = set()
    for entry in g2.plies:
        per_tier = tier_occupancy(entry.obs)
        tiers_in_planes.update(
            tier + 1 for tier in range(3) if bool(per_tier[tier].any())
        )
    assert tiers_in_planes == {1, 2, 3}, (
        f"the occupancy planes only ever encode tiers {sorted(tiers_in_planes)}"
    )


# ---------------------------------------------------------------------------
# (c) The repetition encoding, against the repaired fourfold rule
# ---------------------------------------------------------------------------


def test_repetition_planes_encode_occurrence_one_two_and_three(g1: Golden) -> None:
    """``{R2, R3}`` is a bijection onto the live occurrence classes {1, 2, 3}.

    §3.1's table, and the reason it needs no third plane: the repaired rule
    draws at ``count >= 4`` (``rules-audit`` F/Q4), so a position an
    observation is taken of during play occurs one, two or three times, and
    occurrence 1 is the all-zero codeword.  G1's ``PURE_SHUFFLE`` line is the
    only golden that visits all three -- it repeats its position every four
    plies -- which is what #27 exists to keep true.

    §3.3's P2 is checked on the same plies: two occurrences of one 274-byte key
    bound a span with no progress move in it, so ``repetition_count >= 2``
    forces ``progress_plies >= 2``.
    """
    codewords: dict[int, tuple[float, float]] = {}
    for entry in g1.plies:
        count = entry.repetition_count
        assert count >= 1, f"G1 ply {entry.ply}: occurrence {count}"
        pair = (
            constant_value(entry.obs, PLANE_R2),
            constant_value(entry.obs, PLANE_R3),
        )
        assert pair == (
            1.0 if count >= 2 else 0.0,
            1.0 if count >= 3 else 0.0,
        ), f"G1 ply {entry.ply}: occurrence {count} encodes as {pair}"
        assert codewords.setdefault(count, pair) == pair, (
            f"G1 ply {entry.ply}: occurrence {count} encodes as {pair} here and "
            f"as {codewords[count]} elsewhere"
        )
        if count >= 2:
            assert entry.progress_plies >= 2, (
                f"G1 ply {entry.ply}: P2 says a repeat implies two reversible "
                f"plies, but progress_plies is {entry.progress_plies}"
            )

    # All three live classes occur, and each has its own codeword: without the
    # coverage half this test passes on a golden that never repeats anything.
    assert {1: (0.0, 0.0), 2: (1.0, 0.0), 3: (1.0, 1.0)}.items() <= codewords.items()
    live = {count: word for count, word in codewords.items() if count <= 3}
    assert len(set(live.values())) == len(live), (
        f"two live occurrence classes share a codeword: {live}"
    )


# ---------------------------------------------------------------------------
# (d) The consumer seam refuses a mis-shaped observation
# ---------------------------------------------------------------------------


def schema_sample() -> dict[str, torch.Tensor]:
    """One well-formed self-play row: zeros, with a real observation in it.

    Built from ``records.SCHEMA`` itself rather than from a literal field list,
    so this test keeps testing the shape refusal when the schema grows a field
    instead of failing for an unrelated reason.
    """
    sample = {
        field.name: torch.zeros(field.shape, dtype=field.torch_dtype)
        for field in records.SCHEMA.fields
    }
    game = Game(board_from_fen(ADVANCED_POSITION))
    obs: npt.NDArray[np.float32] = np.ascontiguousarray(
        game.observation(), dtype=np.float32
    )
    sample["observation"] = torch.from_numpy(obs)
    sample["plane_map_version"] = torch.tensor(
        observation.PLANE_MAP_VERSION, dtype=torch.int64
    )
    return sample


def test_a_misshaped_observation_is_refused_by_the_record_seam() -> None:
    """A 118-plane observation is refused by name, and 128 planes are accepted.

    ``records.SCHEMA`` is the seam every self-play row crosses: the worker's
    rows go to ``ShardWriter.append``, which validates against it before a byte
    is written (``shards.py:462``), and the schema's observation field is read
    from the encoder rather than restated (``records.py:57``).  So the shape
    the design pre-registers and the shape a row must have cannot drift apart,
    and a row encoded under the 118-plane map cannot be written as though it
    were this one -- which is D8's "not loadable, loudly" half that the
    fingerprint check backs up at read time.
    """
    assert records.SCHEMA.fields[0].name == "observation"
    assert records.SCHEMA.fields[0].shape == observation.observation_size()
    assert records.SCHEMA.fields[0].shape == (N_PLANES, 9, 9)

    sample = schema_sample()
    records.SCHEMA.validate(sample)  # the well-shaped row is accepted

    stale = dict(sample)
    stale["observation"] = torch.zeros((118, 9, 9), dtype=torch.float32)
    with pytest.raises(ValueError) as failure:
        records.SCHEMA.validate(stale)
    message = str(failure.value)
    assert "observation" in message
    assert str(observation.observation_size()) in message, (
        f"the refusal must name the expected shape; it said: {message}"
    )
    assert "(118, 9, 9)" in message, (
        f"the refusal must name the shape it got; it said: {message}"
    )

    # A plane dropped rather than a whole block is refused just as loudly.
    short = dict(sample)
    short["observation"] = torch.zeros((N_PLANES - 1, 9, 9), dtype=torch.float32)
    with pytest.raises(ValueError):
        records.SCHEMA.validate(short)
