"""KOM-002 §7 A -- the plane map, as the encoder actually declares it.

Four acceptances, and the thing they have in common is that none of them
trusts a docstring:

* **#1** pins the declared size and the plane-map version against the tests'
  own copy of the map (``fixtures.REPAIRED_PLANE_COUNT``), so an accidental
  renumbering in ``observation.py`` is caught rather than followed.
* **#2** pins the produced tensor against the declared size, at ply 0 and
  after real play, because "the encoder says 128" and "the encoder emits 128"
  are two claims.
* **#3** is P1, the append-only invariant: planes 0-117 are bit-identical to
  the pmv1 encoder at ``main@29213f2``, over every position of the golden
  corpus, reached by REPLAYING each golden game rather than by loading its
  FENs -- a FEN carries no history, so under a FEN loop planes 115-117 would
  be constant zero and P1 would only ever be exercised on 0-114 (design rev 2,
  N1).  The frozen side is ``tests/encoding/goldens/pmv1/`` (design rev 4).
  Since KOM-165 raised the turn cap, plane 115 is compared after an EXACT
  float32 rescale onto the frozen cap (``_rescaled_to_pmv1``): the fixture is
  unchanged, P1 is not amended, and plane 115 gains the stronger pin that it is
  a pure rescale of pmv1's.
* **#4** is the differential: the pure-Python oracle path and the cyboard fast
  path must agree on all 128 planes at every golden FEN.  A FEN carries no
  move history and no progress counter, so both paths are obliged to emit
  zeros for 118-127 -- which is exactly what makes the oracle's permanent
  limitation (design §8 gap 4) safe rather than a divergence.
* **#5** pins the reservation: planes 121-127 are identically zero, at every
  ply.  Filling one is a ``PLANE_MAP_VERSION`` bump, not a silent edit, and
  this is the test that makes "silent" impossible.
"""

from __future__ import annotations

import hashlib
from typing import Final

import numpy as np
import numpy.typing as npt

from komugi.engine import observation
from komugi.engine.api import Gungi
from komugi.engine.cyboard import board_from_fen
from komugi.engine.fen import INTRO_POSITION
from komugi.engine.game import Game

from fixtures import (
    GOLDEN_CORPUS_PLIES,
    PLANE_LAST_MOVE_FROM,
    PLANE_LAST_MOVE_TO,
    PLANE_PROGRESS,
    PMV1_MODES,
    PMV1_PLANE_COUNT,
    REPAIRED_PLANE_COUNT,
    RESERVED_PLANES,
    golden_action_sequences,
    golden_fens,
    load_pmv1_capture,
    observation_planes,
    plane_count,
    seeded_playout,
    start_mode,
)

_SEED: Final[int] = 20260904
"""Seed for the playouts below.  Fixed so a failure is reproducible."""

_MISMATCH_BUDGET: Final[int] = 5
"""How many differing FENs to collect before reporting -- the same budget
``tests/engine/test_cygame.py`` uses, so a systematic break prints a handful of
examples instead of every FEN in the corpus."""

PMV1_MAX_TURNS: Final[int] = 256
"""The turn cap the pmv1 capture was frozen under (``main@29213f2``).

Transcribed, not imported: the fixture is a historical artefact and the number
that produced it does not move when :data:`observation.MAX_TURNS` does.  It is
plane 115's (and 120's) denominator on the frozen side, and the only quantity
that separates a pmv1 observation from a current one -- see
:func:`_rescaled_to_pmv1`."""

PMV1_TURN_CLOCK_PLANE: Final[int] = 115
"""The one plane of 0-117 whose VALUES moved when KOM-165 raised the cap."""


def _rescaled_to_pmv1(
    planes: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """*planes* 0-117 expressed on the pmv1 cap, exactly (KOM-165 review F4).

    Plane 115 is ``min(plies, MAX_TURNS) / MAX_TURNS``, so at ply *k* the frozen
    side holds ``k / PMV1_MAX_TURNS`` and this process produces
    ``k / MAX_TURNS``.  Both are dyadic rationals -- exactly representable in
    ``float32`` for every reachable *k* -- and the two caps are powers of two, so
    multiplying by ``MAX_TURNS / PMV1_MAX_TURNS`` is an exact scaling of the
    exponent alone: no mantissa bit moves and no tolerance is introduced.  P1
    therefore survives KOM-165 as a stronger claim than before -- plane 115 is a
    provable pure rescale of pmv1's, and planes 0-114/116-117 are still
    bit-identical -- with the fixture UNCHANGED.

    The rescale is exact only while no golden ply reaches ``PMV1_MAX_TURNS``:
    at or past it the frozen side is clipped to 1.0 and this side is not.  The
    caller asserts that, per game, rather than assuming it.
    """
    rescaled = planes.copy()
    rescaled[PMV1_TURN_CLOCK_PLANE] *= np.float32(
        observation.MAX_TURNS / PMV1_MAX_TURNS
    )
    return rescaled


def _oracle_observation(fen: str) -> npt.NDArray[np.float32]:
    """The pure-Python oracle path's encoding of *fen*."""
    return np.asarray(observation.observation(Gungi(fen)), dtype=np.float32)


def _fast_observation(fen: str) -> npt.NDArray[np.float32]:
    """The cyboard fast path's encoding of *fen*, with no history supplied."""
    return np.asarray(
        observation.observation_from_cyboard(board_from_fen(fen)), dtype=np.float32
    )


def test_the_encoder_declares_128_planes() -> None:
    """#1: 128 planes, both entry points, at plane-map version 3.

    Version 3 is KOM-002's map with planes 115 and 120 re-based on KOM-165's
    raised turn cap: no plane changed index, source or meaning, but every value
    of those two planes is a rescale of a version-2 value, so shards, books and
    checkpoints from either side must not be mixed.
    """
    assert observation.observation_size() == (128, 9, 9)
    assert Game.observation_size() == (128, 9, 9)
    assert observation.PLANE_MAP_VERSION == 3

    # The fixtures hold the tests' independent copy of the pre-registered map
    # (design §5.6); a renumbering that moved the encoder alone stops here.
    assert plane_count() == REPAIRED_PLANE_COUNT == 128
    assert (PLANE_LAST_MOVE_FROM, PLANE_LAST_MOVE_TO, PLANE_PROGRESS) == (118, 119, 120)
    assert RESERVED_PLANES == range(121, 128)
    assert RESERVED_PLANES.stop == REPAIRED_PLANE_COUNT


def _game_at_ply(target: int, seed: int) -> Game:
    """A ``Game`` driven to exactly *target* plies by seeded legal play.

    Retries adjacent seeds, because a random game can end early: a test that
    says "at ply 30" and silently measures ply 9 has stopped testing what its
    name claims.  The observation is taken off the returned ``Game`` rather
    than off a snapshot helper, so the dtype assertion below reads the
    ENCODER's own array and not a coerced view of it.
    """
    for offset in range(16):
        game = Game(board_from_fen(INTRO_POSITION))
        rng = np.random.default_rng(seed + offset)
        while game.turns < target:
            legal = np.nonzero(np.asarray(game.valid_moves(), dtype=np.uint8))[0]
            if legal.size == 0:
                break
            game.play_action(int(rng.choice(legal)))
        if game.turns == target:
            return game
    raise AssertionError(f"no seed near {seed} produced a {target}-ply game")


def test_a_real_observation_has_the_declared_shape() -> None:
    """#2: a real ``Game`` produces (128, 9, 9) float32 at ply 0 and ply 30.

    Ply 30 as well as ply 0 because the appended planes are the ones that only
    become non-trivial once a move has been played: at ply 0 a wrong-width
    last-move block would still be all zeros.
    """
    at_zero = Game(board_from_fen(INTRO_POSITION)).observation()
    assert at_zero.shape == (REPAIRED_PLANE_COUNT, 9, 9)
    assert at_zero.dtype == np.float32

    game = _game_at_ply(30, _SEED)
    assert game.turns == 30
    assert game._last_move is not None, "ply 30 really has a last move to encode"
    at_thirty = game.observation()
    assert at_thirty.shape == (REPAIRED_PLANE_COUNT, 9, 9)
    assert at_thirty.dtype == np.float32


def test_the_two_encoders_agree_on_every_golden_fen() -> None:
    """#4: oracle path == fast path, elementwise, on all 128 planes.

    A FEN is a POSITION: it carries no decoded last move and no progress
    counter (``fen.py``'s six fields), so both encoders are obliged to emit
    zeros for planes 118-127.  That is what keeps the oracle's permanent
    limitation (design §8 gap 4) from being a divergence -- and it is why this
    test extends the historical differential to the full width rather than
    stopping at 117.
    """
    fens = golden_fens()
    assert len(fens) >= 1500, f"expected the full golden corpus, got {len(fens)}"

    mismatches: list[tuple[str, list[int]]] = []
    for fen in fens:
        oracle = _oracle_observation(fen)
        fast = _fast_observation(fen)
        if not np.array_equal(oracle, fast):
            differing = np.where(np.any(oracle != fast, axis=(1, 2)))[0].tolist()
            mismatches.append((fen, differing))
            if len(mismatches) >= _MISMATCH_BUDGET:
                break

    assert not mismatches, (
        f"{len(mismatches)} FEN(s) encode differently; first: "
        f"fen={mismatches[0][0]!r}, differing planes={mismatches[0][1]}"
    )


def test_reserved_planes_are_zero() -> None:
    """#5: planes 121-127 are identically 0.0 at every ply of a 60-ply game.

    The reservation is what makes a future plane addition free of a shape
    change -- and therefore invisible to the shard fingerprint, which is why
    ``PLANE_MAP_VERSION`` exists at all (design §4).  A reserved plane that
    quietly acquired a value would break that bargain without breaking a
    single shape check, so it is pinned here by value.
    """
    snapshots = seeded_playout(60, _SEED)
    assert snapshots[-1].ply == 60

    for snapshot in snapshots:
        reserved = snapshot.observation[RESERVED_PLANES.start : RESERVED_PLANES.stop]
        assert reserved.shape == (len(RESERVED_PLANES), 9, 9)
        assert not reserved.any(), (
            f"ply {snapshot.ply}: reserved planes are not all zero; "
            f"nonzero planes "
            f"{(RESERVED_PLANES.start + np.where(np.any(reserved != 0, axis=(1, 2)))[0]).tolist()}"
        )


def _pmv1_mismatch(
    mode: str,
    game_index: int,
    ply: int,
    planes: npt.NDArray[np.float32],
    sample: npt.NDArray[np.float32] | None,
    actual: str,
    expected: str,
) -> str:
    """The failure message for a position whose pmv1 planes moved.

    Names the mode, the game and the ply always; names the first differing
    PLANE too whenever the capture sampled this position's raw planes (ply 0
    and each game's final ply), which is the localisation design §6.4 asks a
    plane-map regression to print.
    """
    where = f"{mode} game {game_index} ply {ply}"
    if sample is None:
        return (
            f"{where}: planes 0-{PMV1_PLANE_COUNT - 1} are not bit-identical to "
            f"the pmv1 capture (sha256 {actual} != {expected}).  The capture "
            "stores raw planes only for ply 0 and each game's final ply, so the "
            "differing plane index is not recoverable at this ply; the sampled "
            "plies of the same game will name it if the regression is uniform."
        )
    differing = np.where(np.any(planes != sample, axis=(1, 2)))[0]
    if differing.size == 0:
        return (
            f"{where}: the sha256 of planes 0-{PMV1_PLANE_COUNT - 1} differs "
            f"({actual} != {expected}) while every plane compares equal to the "
            "sampled raw planes -- the digest and the sample disagree, so the "
            "capture itself is suspect."
        )
    first = int(differing[0])
    return (
        f"{where}: planes 0-{PMV1_PLANE_COUNT - 1} differ from the pmv1 capture "
        f"(sha256 {actual} != {expected}); first differing plane: {first} "
        f"({differing.size} planes differ in all: {differing.tolist()[:8]}...); "
        f"pmv2 plane {first} first row {planes[first][0].tolist()} vs pmv1 "
        f"{sample[first][0].tolist()}"
    )


def test_planes_0_to_117_are_bit_identical_to_pmv1() -> None:
    """#3, invariant P1: planes 0-117 equal the pmv1 encoder, elementwise.

    The frozen side is ``tests/encoding/goldens/pmv1/``: per-position sha256
    digests of the 118-plane observation taken BEFORE each ply, captured at
    ``main@29213f2`` by replaying the goldens (design rev 4 records why the
    fixture is digests plus sampled raw planes rather than raw arrays -- the
    raw corpus is ~102 MB).  A digest over ``float32`` bytes is an ELEMENTWISE
    equality: every value in the map is an exact binary ratio, nothing is
    summed, so there is no tolerance to argue about and none is offered.

    Positions are reached by replaying each golden game's ``chosen`` sequence,
    not by loading its FENs, so planes 115-117 (turn clock, R2, R3) actually
    vary across the fixture -- the whole point of rev 2's N1.

    **KOM-165 (review F4).** Raising the turn cap re-based plane 115's
    denominator, so the comparison is now made after :func:`_rescaled_to_pmv1`
    puts this process's plane 115 back on the frozen cap -- an EXACT float32
    scaling, not a tolerance.  The fixture is untouched and P1 is not amended:
    planes 0-114 and 116-117 are still bit-identical to ``main@29213f2``, and
    plane 115 is now additionally pinned as a pure rescale of pmv1's.  The
    rescale is only exact while no golden ply reaches the frozen cap, which is
    asserted per game below rather than assumed.
    """
    positions = 0
    for mode in PMV1_MODES:
        capture = load_pmv1_capture(mode)
        sequences = golden_action_sequences(mode)
        assert len(sequences) == len(capture.plies_per_game), (
            f"{mode}: corpus has {len(sequences)} games, capture has "
            f"{len(capture.plies_per_game)}"
        )
        longest = max(capture.plies_per_game)
        assert longest < PMV1_MAX_TURNS, (
            f"{mode}: a golden game runs {longest} plies, at or past the "
            f"{PMV1_MAX_TURNS}-ply cap the pmv1 capture was frozen under. Past "
            "that ply the frozen plane 115 is clipped to 1.0 while this "
            "process's is not, so the rescale in _rescaled_to_pmv1 is no longer "
            "exact -- and the frozen win_state would carry a turn-cap draw this "
            "process no longer adjudicates. The fixture must be regenerated "
            "before this invariant can be read as P1 again."
        )
        unvisited = set(capture.sample_keys)

        with start_mode(mode):
            for game_index, actions in enumerate(sequences):
                assert len(actions) == capture.plies_per_game[game_index], (
                    f"{mode} game {game_index}: corpus has {len(actions)} plies, "
                    f"capture has {capture.plies_per_game[game_index]}"
                )
                game = Game()
                for ply, action in enumerate(actions):
                    planes = np.ascontiguousarray(
                        _rescaled_to_pmv1(
                            np.asarray(
                                observation_planes(game)[:PMV1_PLANE_COUNT],
                                dtype=np.float32,
                            )
                        ),
                        dtype=np.float32,
                    )
                    expected = capture.digests[capture.flat_index(game_index, ply)]
                    actual = hashlib.sha256(planes.tobytes()).hexdigest()
                    sample = capture.sample_for(game_index, ply)

                    if actual != expected:
                        raise AssertionError(
                            _pmv1_mismatch(
                                mode, game_index, ply, planes, sample, actual, expected
                            )
                        )
                    if sample is not None:
                        # The digest already implies this; asserting it as well
                        # is what makes the sampled plies a check on the CAPTURE
                        # (a digest agrees with itself; raw planes do not).
                        assert np.array_equal(planes, sample), _pmv1_mismatch(
                            mode, game_index, ply, planes, sample, actual, expected
                        )
                        unvisited.discard((game_index, ply))

                    game.play_action(action)
                    positions += 1

        assert not unvisited, (
            f"{mode}: the capture samples raw planes at {sorted(unvisited)}, "
            "which the replay never reached -- the sampled plies are the only "
            "elementwise check in this test and they must all be exercised"
        )

    assert positions == GOLDEN_CORPUS_PLIES, (
        f"compared {positions} positions, expected {GOLDEN_CORPUS_PLIES}: the "
        "golden corpus is not the one this invariant was captured over"
    )
