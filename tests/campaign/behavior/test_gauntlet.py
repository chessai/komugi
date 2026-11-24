"""The anchor: pinned by path and by both digests, and loadable back.

KOM-022 §4, acceptances #14, #29, #35 and #39. The rev-1 defect this file
exists for is worth restating, because it is invisible at the call site:
:func:`~komugi.train.checkpoint.load_checkpoint_file` **does not open the path
it is handed**. It lists the parent directory and looks for the path among the
*listed* checkpoints, and the listing admits only
``checkpoint-(\\d{9,})-([0-9a-f]{16})\\.pt``. A copy named ``gen_0.pt`` is
therefore never listed, the loader returns ``None``, and both readings of that
``None`` are bad: treat it as "verification failed" and every gauntlet aborts,
so the whole of W3 never runs; treat it as benign and the anchor is never
verified, so the digest check is a no-op.

That is why **acceptance #29 asserts the positive case first** and #14's
mutation case comes second: rev 1 tested only the mutation, and a mutation test
passes both before *and* after the mutation on an anchor that was never
readable at all.

The gauntlet's own arithmetic -- sharding, merging, scoring -- is exercised over
constructed records, because that is what makes the sharding claim ("the union
of the shards is the serial match") checkable without a 60-minute match. One
test does play: :func:`test_a_two_shard_gauntlet_scores_a_small_net_match` runs
a real four-game, two-shard gauntlet with the small net, because acceptance
#38 is about what a *spawned* shard is handed and no amount of arithmetic over
records can see a process that was started without ``NET_ENV``.
"""

from __future__ import annotations

import dataclasses
import multiprocessing
import os
import time
from pathlib import Path
from typing import Any, Final, NoReturn

import pytest
import torch

from komugi.campaign import pinned as pinned_module
from komugi.campaign.factory import NET_ENV, SMALL_NET, new_net
from komugi.campaign.gauntlet import (
    AnchorPin,
    GauntletAborted,
    GauntletLost,
    GauntletStalled,
    ShardRequest,
    anchor_copy_path,
    checkpoint_player,
    generation_dir,
    merge_outcomes,
    pin_anchor,
    pinfile_path,
    read_pins,
    score_for,
    shard_ranges,
    summarise,
    verify_anchor,
)
from komugi.campaign.pinned import (
    GAUNTLET_FLOOR,
    GAUNTLET_GAMES,
    GAUNTLET_MIN_DECISIVE,
)
from komugi.eval.records import BLACK_WIN, DRAW, WHITE_WIN, GameRecord
from komugi.train.checkpoint import list_checkpoints, load_checkpoint_file, save_checkpoint
from komugi.train.digest import bn_digest, canonical_digest, parameter_keys

STEP: Final[int] = 96
M: Final[int] = 8
N: Final[int] = 32
DRAW_SCORE: Final[float] = 0.20

CHALLENGER: Final[str] = "ema-104"
ANCHOR: Final[str] = "anchor-g0"


@pytest.fixture
def net(monkeypatch: pytest.MonkeyPatch) -> torch.nn.Module:
    """A small net, with ``NET_ENV`` set for the process that builds it."""
    monkeypatch.setenv(NET_ENV, SMALL_NET)
    return new_net()


def _publish(directory: Path, state: dict[str, torch.Tensor], step: int) -> Path:
    """Publish `state` as an ordinary checkpoint and return its path."""
    return save_checkpoint(directory, step, state, {}, {"iteration": 0, "role": "train"})


def _pin(root: Path, source: Path, keys: tuple[str, ...]) -> AnchorPin:
    """Freeze `source` as anchor generation 0 at the fast budget."""
    return pin_anchor(
        root,
        0,
        0,
        source,
        keys,
        m=M,
        n=N,
        games=GAUNTLET_GAMES,
        floor=GAUNTLET_FLOOR,
        draw_score=DRAW_SCORE,
    )


# ---------------------------------------------------------------------------
# acceptance #29 -- the positive case, asserted first
# ---------------------------------------------------------------------------


def test_pinned_anchor_loads_back(tmp_path: Path, net: torch.nn.Module) -> None:
    """Acceptance #29: the anchor round-trips through the loader the players use.

    The copy keeps its **published basename**, so
    :func:`~komugi.train.checkpoint.list_checkpoints` lists it and
    ``load_checkpoint_file`` finds it -- and the digest-in-the-name integrity
    check the checkpoint module was built around covers the anchor copy for
    free, which is most of what a hand-rolled digest comparison was buying.
    The generation lives in the directory name and the pinfile line, never in
    the filename.
    """
    root = tmp_path / "root"
    state = dict(net.state_dict())
    keys = parameter_keys(net)
    source = _publish(tmp_path / "ckpt", state, STEP)

    pin = _pin(root, source, keys)

    copy = anchor_copy_path(root, pin)
    assert copy.parent == generation_dir(root, 0)
    assert copy.name == source.name, (
        "the anchor copy was renamed; a name outside "
        "checkpoint-<step:09d>-<digest16>.pt is never listed, so the loader "
        "returns None and the anchor is unloadable"
    )
    assert copy.name.startswith("checkpoint-")
    assert [ref.path for ref in list_checkpoints(copy.parent)] == [copy]

    loaded = load_checkpoint_file(copy)
    assert loaded is not None, (
        "the pinned anchor does not read back through the loader "
        "checkpoint_player uses: every gauntlet would abort"
    )
    assert canonical_digest(loaded.model_state, keys).sha256 == pin.weight_sha256
    assert bn_digest(loaded.model_state, keys).sha256 == pin.bn_sha256

    # And verification, the call every gauntlet makes before playing anything.
    assert verify_anchor(root, pin, keys) == copy


def test_the_pinfile_line_is_the_commit_point(
    tmp_path: Path, net: torch.nn.Module
) -> None:
    """§4: the pinfile line is appended **last**, after the copy verifies.

    A kill mid-copy then leaves an unpinned directory rather than a *pinned*
    generation whose anchor no longer verifies -- which the abort rule would
    wedge permanently behind an append-only file, escapable only by hand.
    """
    root = tmp_path / "root"
    keys = parameter_keys(net)
    source = _publish(tmp_path / "ckpt", dict(net.state_dict()), STEP)

    assert read_pins(root) == ()
    pin = _pin(root, source, keys)

    pins = read_pins(root)
    assert pins == (pin,)
    assert pinfile_path(root).is_file()
    assert pin.path == str(source), (
        "the pinfile must record the *source*, not the copy: recovery re-copies "
        "from it, and the copy's location is a function of the generation"
    )
    assert pin.games == GAUNTLET_GAMES
    assert pin.m == M and pin.n == N
    assert pin.draw_score == DRAW_SCORE


def test_kill_during_anchor_copy_leaves_no_pinfile_line(
    tmp_path: Path, net: torch.nn.Module
) -> None:
    """Acceptance #35: never a pinned-but-unreadable anchor.

    An unverifiable source aborts *before* anything is appended, so the
    generation is simply not pinned and the next attempt can pin it cleanly.
    """
    root = tmp_path / "root"
    keys = parameter_keys(net)
    torn = tmp_path / "ckpt" / "checkpoint-000000096-0123456789abcdef.pt"
    torn.parent.mkdir(parents=True)
    torn.write_bytes(b"the first half of a checkpoint")

    with pytest.raises(GauntletAborted, match="does not verify"):
        _pin(root, torn, keys)

    assert read_pins(root) == ()
    assert not pinfile_path(root).exists()


# ---------------------------------------------------------------------------
# acceptance #14 -- the mutation case, second
# ---------------------------------------------------------------------------


def test_anchor_digest_mismatch_aborts(tmp_path: Path, net: torch.nn.Module) -> None:
    """Acceptance #14: positive case first, then a mutation confined to BatchNorm.

    **Both digest halves are checked, deliberately.** For an *anchor* the
    question is file identity, not learning, so §3a's narrowing of the liveness
    digest to the parameters must not narrow this: an opponent whose
    normalisation statistics were swapped underneath is a different opponent,
    and the parameter digest alone would wave it through. That is why the
    pinfile carries ``bn_sha256`` beside ``weight_sha256``.

    No games are played: :func:`verify_anchor` is the first thing
    ``run_gauntlet`` does, and it raises before a shard process exists.
    """
    root = tmp_path / "root"
    keys = parameter_keys(net)
    state = dict(net.state_dict())
    source = _publish(tmp_path / "ckpt", state, STEP)
    pin = _pin(root, source, keys)

    # Positive case first (see #29): the freshly pinned anchor verifies.
    assert verify_anchor(root, pin, keys) == anchor_copy_path(root, pin)

    # Now the swap, confined to the running statistics: the file on disk is a
    # different opponent from the one the pinfile names, and not one parameter
    # moved.
    swapped = dict(state)
    for key, value in state.items():
        if key.endswith(("running_mean", "running_var")):
            swapped[key] = value + 1.0
    assert canonical_digest(swapped, keys).sha256 == pin.weight_sha256, (
        "the planted mutation moved a parameter, so it is not the BatchNorm-only "
        "swap this test is about"
    )
    bn_only = dataclasses.replace(pin, bn_sha256=bn_digest(swapped, keys).sha256)

    with pytest.raises(GauntletAborted) as raised:
        verify_anchor(root, bn_only, keys)

    message = str(raised.value)
    assert bn_only.bn_sha256[:16] in message, "the pinned digest is not named"
    assert pin.bn_sha256[:16] in message, "the recomputed digest is not named"
    assert "swapped underneath is not an anchor" in message
    assert "no games were played" in message


def test_a_corrupted_copy_is_recovered_from_the_recorded_source(
    tmp_path: Path, net: torch.nn.Module
) -> None:
    """§4's recovery, defined rather than left to surgery.

    If the newest pinfile line's anchor fails to load, and the ``path`` it
    records still holds a file that verifies to the pinned digests, the copy is
    re-made from it. Only if that also fails does the gauntlet abort -- which is
    the difference between a transient bad copy and an anchor that was swapped.
    """
    root = tmp_path / "root"
    keys = parameter_keys(net)
    source = _publish(tmp_path / "ckpt", dict(net.state_dict()), STEP)
    pin = _pin(root, source, keys)
    copy = anchor_copy_path(root, pin)

    copy.unlink()
    assert verify_anchor(root, pin, keys) == copy
    assert copy.is_file(), "the copy was not re-made from the recorded source"

    # And with the source gone too, there is nothing left to recover from.
    copy.write_bytes(b"torn")
    source.unlink()
    with pytest.raises(GauntletAborted, match="could not supply a replacement"):
        verify_anchor(root, pin, keys)


# ---------------------------------------------------------------------------
# acceptance #38 -- the shards are spawned with the device and the net shape
# ---------------------------------------------------------------------------


SMALL_GAMES: Final[int] = 4
"""A whole number of seat-swapped pairs, and the smallest match with a shard
split: two pairs across two shards is one pair each."""

SMALL_SHARDS: Final[int] = 2
FAST_M: Final[int] = 2
FAST_N: Final[int] = 2
"""The smallest Gumbel budget that still runs the real search."""


def test_a_two_shard_gauntlet_scores_a_small_net_match(
    tmp_path: Path, net: torch.nn.Module, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Acceptance #38, and the first test to run ``run_gauntlet`` at all.

    ``run_gauntlet``, ``run_shard`` and ``play_pairs`` were executed by **no**
    test (review F4): the live campaign rigs run two iterations and the cadence
    first fires at iteration 3, so every claim about the fan-out rested on the
    arithmetic tests above. What those cannot see is the seam: whether a shard
    process can be *spawned* with what it needs.

    Two things it needs, and neither may be inherited:

    * ``NET_ENV``, because a shard spawned without it against a small-net
      campaign builds the production net and fails inside ``load_state_dict``
      with a shape error that reads like a corrupt anchor;
    * the **device**, explicitly. §6's argument for running the gauntlet
      between iterations is about free cores, not about sixteen processes
      sharing one GPU.

    And the third half of #38: ``pin_determinism`` is **not** called in the
    campaign process. It runs per game, inside the shard, which is where the
    determinism claim is made and where the cost belongs.
    """
    from komugi.campaign import gauntlet as gauntlet_module
    from komugi.eval import arena as arena_module

    root = tmp_path / "root"
    keys = parameter_keys(net)
    anchor_source = _publish(tmp_path / "ckpt", dict(net.state_dict()), STEP)
    pin = pin_anchor(
        root,
        0,
        0,
        anchor_source,
        keys,
        m=FAST_M,
        n=FAST_N,
        games=SMALL_GAMES,
        floor=GAUNTLET_FLOOR,
        draw_score=DRAW_SCORE,
    )

    # A challenger that is a *different* net, so the match is not a net playing
    # itself under two names.
    challenger_state = {
        key: value + 0.05 if value.is_floating_point() else value
        for key, value in net.state_dict().items()
    }
    challenger = _publish(tmp_path / "ema", challenger_state, STEP + 8)

    # Observed at the *parent-side* seam, not by substituting ShardRequest.
    # ``pickle`` serialises an instance by looking its class up again as
    # ``komugi.campaign.gauntlet.ShardRequest``; rebinding that attribute --
    # even to a wrapper returning genuine instances -- makes every shard die in
    # ``multiprocessing.reduction`` with "it's not the same object as". So the
    # patch goes on the function that *builds* the requests, which runs in this
    # process and hands back the untouched objects the pool then pickles. The
    # spawned children re-import the module fresh, so nothing patched here
    # (this, or ``pin_determinism`` below) is visible to them.
    requested: list[ShardRequest] = []
    real_requests = gauntlet_module._shard_requests

    def recording(*args: Any, **fields: Any) -> tuple[ShardRequest, ...]:
        """Build the real, picklable requests and keep references to them."""
        built: tuple[ShardRequest, ...] = real_requests(*args, **fields)
        requested.extend(built)
        return built

    pinned_here: list[str] = []
    monkeypatch.setattr(gauntlet_module, "_shard_requests", recording)
    monkeypatch.setattr(
        arena_module, "pin_determinism", lambda: pinned_here.append("parent")
    )

    played = gauntlet_module.run_gauntlet(
        root,
        challenger,
        CHALLENGER,
        pin,
        keys,
        seed=20260904,
        device="cpu",
        shards=SMALL_SHARDS,
        min_decisive=1,
    )

    assert played.wall_s > 0.0, "no time passed; no games were played"
    assert played.summary.games == SMALL_GAMES, (
        f"the merged summary counts {played.summary.games} games for a "
        f"{SMALL_GAMES}-game match: the shards did not cover the pairs"
    )
    assert played.summary.decisive + played.summary.draws == SMALL_GAMES
    assert played.summary.total_plies > 0
    assert 0.0 <= played.score <= 1.0
    assert played.generation == 0

    assert len(requested) == SMALL_SHARDS, (
        f"{len(requested)} shard request(s) for {SMALL_SHARDS} shards"
    )
    for request in requested:
        assert request.device == "cpu", (
            "a shard inherited its device instead of being told one"
        )
        assert request.net_env == SMALL_NET, (
            f"a shard was spawned with net_env={request.net_env!r}: it would "
            f"build the production net and fail inside load_state_dict with a "
            f"shape error that reads like a corrupt anchor"
        )
        assert request.games == SMALL_GAMES
        assert request.m == FAST_M and request.n == FAST_N
        assert request.draw_score == DRAW_SCORE
    covered = sorted(
        (request.pairs_start, request.pairs_stop) for request in requested
    )
    assert covered == [(0, 1), (1, 2)], (
        f"the shards' pair ranges are {covered}: they must be contiguous, "
        f"disjoint and index-preserving, or the union is not the serial match"
    )

    assert pinned_here == [], (
        "pin_determinism ran in the campaign process: it belongs to the shard "
        "processes, one call per game, and calling it here would impose "
        "deterministic kernels on the trainer that spawned them"
    )


# ---------------------------------------------------------------------------
# acceptance #39 -- the shards merge into exactly the serial match
# ---------------------------------------------------------------------------


def test_shard_ranges_are_disjoint_and_cover_every_pair() -> None:
    """§4: contiguous, index-preserving slices of ``range(0, pairs)``.

    Index-preserving is the load-bearing half: every game's randomness is
    ``game_seeds(config, pair, seat)``, a pure function of the match seed and
    the *global* coordinates, so a shard that renumbered its pairs would play a
    different match and the union would not be the serial run.
    """
    pairs = GAUNTLET_GAMES // 2
    ranges = shard_ranges(pairs, 16)

    assert len(ranges) == 16
    covered = [index for piece in ranges for index in piece]
    assert covered == list(range(pairs))
    assert all(piece.step == 1 for piece in ranges)


def test_more_shards_than_pairs_yields_empty_slices_rather_than_an_error() -> None:
    """An empty shard is wasteful, not invalid -- and the merge survives one."""
    ranges = shard_ranges(2, 5)

    assert len(ranges) == 5
    assert sum(len(piece) for piece in ranges) == 2
    assert [index for piece in ranges for index in piece] == [0, 1]
    assert sum(1 for piece in ranges if not piece) == 3


def test_shard_ranges_refuses_a_nonsensical_shard_count() -> None:
    """Zero shards is not a sharding of anything."""
    with pytest.raises(ValueError, match="shards must be >= 1"):
        shard_ranges(32, 0)


def _record(pair: int, seat: int, result: float) -> GameRecord:
    """One constructed game of the challenger-versus-anchor match."""
    white, black = (CHALLENGER, ANCHOR) if seat == 0 else (ANCHOR, CHALLENGER)
    return GameRecord(
        white=white,
        black=black,
        result=result,
        book_id="empty",
        pair_id=f"{CHALLENGER}:{ANCHOR}:{pair:05d}",
    )


def test_sharded_summary_equals_serial(tmp_path: Path) -> None:
    """Acceptance #39: one ``_summarise`` over the union, never a fold of folds.

    ``ArenaSummary.shortest_plies`` uses ``min(plies, default=0)``, so an empty
    shard contributes a ``0`` that a careless per-shard fold would adopt as the
    global minimum -- and ``decisive``, the validity condition the whole floor
    rests on, would inherit the same class of error. The empty shard is in this
    fixture on purpose.
    """
    serial_records = [_record(pair, seat, WHITE_WIN) for pair in range(4) for seat in (0, 1)]
    serial_plies = [40, 41, 42, 43, 44, 45, 46, 47]
    serial = summarise(serial_records, serial_plies)

    shards: list[tuple[list[GameRecord], list[int]]] = [
        (serial_records[:2], serial_plies[:2]),
        ([], []),
        (serial_records[2:6], serial_plies[2:6]),
        (serial_records[6:], serial_plies[6:]),
    ]
    merged = merge_outcomes(shards)

    assert merged.summary == serial
    assert merged.summary.shortest_plies == 40, (
        "the empty shard's default of 0 was adopted as the global minimum"
    )
    assert merged.summary.games == 8
    assert list(merged.records) == serial_records


def test_the_summary_counts_draws_and_decisives_over_the_union() -> None:
    """``decisive`` is what the floor's validity condition reads."""
    records = [
        _record(0, 0, WHITE_WIN),
        _record(0, 1, DRAW),
        _record(1, 0, DRAW),
        _record(1, 1, BLACK_WIN),
    ]
    summary = summarise(records, [10, 20, 30, 40])

    assert summary.games == 4
    assert summary.draws == 2
    assert summary.decisive == 2
    assert summary.decisive < GAUNTLET_MIN_DECISIVE


def test_score_for_reads_from_the_named_players_side() -> None:
    """``GameRecord.result`` is from white's perspective; the score is not.

    A gauntlet plays seat-swapped pairs, so half the challenger's games are as
    black. Reading ``result`` directly for all of them would report the
    challenger's score as roughly ``1 - s`` on half the match.
    """
    swept = [_record(pair, 0, WHITE_WIN) for pair in range(2)] + [
        _record(pair, 1, BLACK_WIN) for pair in range(2)
    ]
    assert score_for(swept, CHALLENGER) == 1.0
    assert score_for(swept, ANCHOR) == 0.0

    drawn = [_record(0, 0, DRAW), _record(0, 1, DRAW)]
    assert score_for(drawn, CHALLENGER) == 0.5

    assert score_for([], CHALLENGER) == 0.0


def test_a_swept_loss_is_below_the_pre_registered_floor() -> None:
    """The floor is ``s < 0.40``, a one-sided 95% bound below 0.50 at n=64.

    ``s + 1.645 * sqrt(0.25/64) < 0.50`` gives ``s < 0.397``, pre-registered at
    0.40. A regression detector, not a progress meter: requiring ``s > 0.5`` at
    64 games would fire on about half of healthy early iterations, when true
    improvement is smaller than the noise.
    """
    # The challenger loses every game in both seats: as white it is a black
    # win, as black it is a white win.
    losing = [_record(pair, 0, BLACK_WIN) for pair in range(16)] + [
        _record(pair, 1, WHITE_WIN) for pair in range(16)
    ]

    assert len(losing) == GAUNTLET_GAMES // 2
    assert score_for(losing, CHALLENGER) == 0.0
    assert score_for(losing, CHALLENGER) < GAUNTLET_FLOOR
    assert score_for(losing, ANCHOR) == 1.0


# ---------------------------------------------------------------------------
# KOM-002 §5.3 -- an entrant from an older plane map (review F3)
# ---------------------------------------------------------------------------

PMV1_PLANES: Final[int] = 118
"""The plane count KOM-002 broke: 118, from KOM-027's map."""


def test_a_stale_plane_count_entrant_is_refused_by_name(
    tmp_path: Path, net: torch.nn.Module
) -> None:
    """The gauntlet's own load site refuses before ``load_state_dict``.

    An anchor is *pinned by path* and lives across generations, so the entrant
    factory is exactly the place an older plane map arrives from: the pinfile
    still names it, the file still verifies, and every digest in it is still
    correct. Without the guard the run dies inside torch with ``size mismatch
    for stem.weight``, in a spawned shard process, mid-gauntlet -- which
    :func:`~komugi.campaign.gauntlet.run_gauntlet`'s own docstring already
    names as the failure to avoid.

    The saved weights are **this** net's, so ``load_state_dict`` would succeed
    if it were reached: the refusal proves ordering, not a shape accident.

    **Deleting ``require_matching_plane_count`` from
    :func:`~komugi.campaign.gauntlet.checkpoint_player` fires
    ``pytest.raises(ValueError)`` below** -- the build then returns an
    evaluator, having silently loaded weights the guard exists to refuse.
    """
    stale = save_checkpoint(
        tmp_path,
        STEP,
        net.state_dict(),
        {},
        {"in_planes": PMV1_PLANES, "plane_map_version": 1},
    )
    assert load_checkpoint_file(stale) is not None, "the file itself verifies"

    player = checkpoint_player(stale, "ema-stale", m=M, n=N)
    with pytest.raises(ValueError) as raised:
        player.evaluator_factory()

    message = str(raised.value)
    assert str(PMV1_PLANES) in message
    assert "KOM-002" in message


# ---------------------------------------------------------------------------
# review CP-04 -- every shard is supervised, and nothing survives the call
# ---------------------------------------------------------------------------
#
# The witness these invert (campaign_crosscheck_lostworker.py) is a shard that
# writes a marker and then calls ``os._exit(91)``: no exception, no result, no
# traceback -- the shape a signal, an OOM kill or a segfault has. ``pool.map``
# waited for it forever, so the campaign's real failure mode was not a crash but
# a run that was still "in progress" days later, and the operator's eventual
# kill landed between a published checkpoint and its boundary record -- CP-01's
# window, which is why the two findings ship together.
#
# The stubs below are **module-level functions** because the spawn context
# pickles the callable by qualified name: a closure or a lambda dies in
# ``multiprocessing.reduction`` before a shard ever starts. They announce their
# pid through a file first and misbehave second, so the parent can ask the one
# question that matters afterwards -- is that pid gone.

SHARD_PID_DIR_ENV: Final[str] = "KOMUGI_TEST_SHARD_PID_DIR"
"""Where a stub shard announces its pid. Read from the environment, which the
spawned child inherits: the request objects are the production ones."""

ABRUPT_EXIT_CODE: Final[int] = 91
"""The witness's own exit code. Any non-zero would do; this one is traceable."""

STALL_SLEEP_S: Final[float] = 900.0
"""Long enough that the stub is only ever ended by the supervisor."""

SHORT_DEADLINE_S: Final[float] = 30.0
"""The deadline the supervision tests pin over the production half hour.

Comfortably above a spawned child's start-up (it imports torch and this module
before it can announce itself) and far below the test suite's patience. A
deadline tight enough to race the import would make the *healthy* case flaky,
which is the failure mode a supervisor must not have.
"""

SHARD_FAILURE: Final[str] = "the shard's own bug"
"""What a *raising* stub raises, to be matched in the parent's re-raise."""

FIRST_SHARD_SLEEP_S: Final[float] = 18.0
"""How long the first shard sleeps in the one-deadline test, in seconds.

Against :data:`SHORT_DEADLINE_S` (30 s) this is the only interesting setting:
long enough that collecting shard 0 spends most of the gauntlet's single
deadline, and short enough that shard 0's future really does **resolve** --
with room for a spawned child to import torch first. A first shard that timed
out instead would make the test pass under either implementation, because the
per-future timer and the absolute one agree the moment nothing returns.
"""

ONE_DEADLINE_SLACK_S: Final[float] = 10.0
"""Teardown allowance over the deadline: ``SIGTERM``, a join, and the merge.

Wide enough for a loaded box, and far below the ``2 x`` deadline a per-future
timer costs -- which is the difference the test is measuring.
"""


def _announce() -> int:
    """Record this process's pid where the parent can read it, and return it."""
    directory = Path(os.environ[SHARD_PID_DIR_ENV])
    directory.mkdir(parents=True, exist_ok=True)
    pid = os.getpid()
    (directory / f"{pid}.pid").write_text(str(pid), encoding="utf-8")
    return pid


def shard_that_exits_abruptly(request: ShardRequest) -> NoReturn:
    """The review's witness: announce, then leave without unwinding anything."""
    _announce()
    os._exit(ABRUPT_EXIT_CODE)


def shard_that_stalls(request: ShardRequest) -> NoReturn:
    """Alive and never returning -- a wedged search, an unreachable seam."""
    _announce()
    time.sleep(STALL_SLEEP_S)
    raise AssertionError("a stalled shard outlived its supervisor")


def shard_that_returns_late_then_stalls(
    request: ShardRequest,
) -> tuple[tuple[GameRecord, ...], tuple[int, ...]]:
    """Shard 0 returns near the deadline; every later shard never returns.

    The shape that tells one deadline from N of them. Shard 0 consumes most of
    the bound and then hands back an (empty) result, so a supervisor that
    restarts its timer for shard 1 spends a second whole deadline on it; one
    that holds a single absolute deadline has only the remainder left.
    """
    _announce()
    if request.pairs_start == 0:
        time.sleep(FIRST_SHARD_SLEEP_S)
        return ((), ())
    time.sleep(STALL_SLEEP_S)
    raise AssertionError("a stalled shard outlived its supervisor")


def shard_that_raises(request: ShardRequest) -> NoReturn:
    """A shard whose own work failed. The parent must still see *this* error."""
    _announce()
    raise ValueError(SHARD_FAILURE)


def shard_that_plays(
    request: ShardRequest,
) -> tuple[tuple[GameRecord, ...], tuple[int, ...]]:
    """Return a shard's worth of constructed games, keeping the global pair index.

    The healthy control runs through the same executor, the same pickling and
    the same reaping as the failures; what it does not do is run a real search,
    because this file already has a test that does
    (:func:`test_a_two_shard_gauntlet_scores_a_small_net_match`) and what is
    under test here is the supervision, not the arena.
    """
    records: list[GameRecord] = []
    plies: list[int] = []
    _announce()
    for pair in range(request.pairs_start, request.pairs_stop):
        for seat in (0, 1):
            white, black = (
                (request.challenger_id, request.anchor_id)
                if seat == 0
                else (request.anchor_id, request.challenger_id)
            )
            records.append(
                GameRecord(
                    white=white,
                    black=black,
                    result=WHITE_WIN,
                    book_id="empty",
                    pair_id=f"{request.challenger_id}:{request.anchor_id}:{pair:05d}",
                )
            )
            plies.append(40 + pair)
    return tuple(records), tuple(plies)


@dataclasses.dataclass(frozen=True)
class _Supervised:
    """A pinned anchor, a challenger and the pid directory the stubs write to."""

    root: Path
    challenger: Path
    pin: AnchorPin
    keys: tuple[str, ...]
    pids: Path

    def announced(self) -> list[int]:
        """Return every pid a stub shard announced, in file order."""
        return sorted(int(path.stem) for path in self.pids.glob("*.pid"))


@pytest.fixture
def supervised(
    tmp_path: Path, net: torch.nn.Module, monkeypatch: pytest.MonkeyPatch
) -> _Supervised:
    """A real pinned anchor and challenger, with the deadline pinned short."""
    root = tmp_path / "root"
    keys = tuple(parameter_keys(net))
    anchor_source = _publish(tmp_path / "ckpt", dict(net.state_dict()), STEP)
    pin = pin_anchor(
        root,
        0,
        0,
        anchor_source,
        keys,
        m=FAST_M,
        n=FAST_N,
        games=SMALL_GAMES,
        floor=GAUNTLET_FLOOR,
        draw_score=DRAW_SCORE,
    )
    challenger_state = {
        key: value + 0.05 if value.is_floating_point() else value
        for key, value in net.state_dict().items()
    }
    challenger = _publish(tmp_path / "ema", challenger_state, STEP + 8)
    pids = tmp_path / "pids"
    pids.mkdir()
    monkeypatch.setenv(SHARD_PID_DIR_ENV, str(pids))
    # Pinned on the module the gauntlet reads it from, not passed as an
    # argument: that the deadline is read from `pinned` at call time is part of
    # what is under test -- a default frozen into the signature at import would
    # make the pin unpinnable.
    monkeypatch.setattr(
        pinned_module, "GAUNTLET_SHARD_DEADLINE_S", SHORT_DEADLINE_S
    )
    return _Supervised(root=root, challenger=challenger, pin=pin, keys=keys, pids=pids)


def _play(supervised: _Supervised) -> Any:
    """Run the two-shard gauntlet exactly as the campaign's boundary does."""
    from komugi.campaign import gauntlet as gauntlet_module

    return gauntlet_module.run_gauntlet(
        supervised.root,
        supervised.challenger,
        CHALLENGER,
        supervised.pin,
        supervised.keys,
        seed=20260912,
        device="cpu",
        shards=SMALL_SHARDS,
        min_decisive=1,
    )


def _assert_no_child_survived(supervised: _Supervised) -> None:
    """Every announced shard pid is gone, and this process has no live child.

    The pid check is the one the review asks for, and it catches both failures
    that matter: a supervisor that raised while leaving its children playing,
    and one that let an exited child stay a **zombie** -- a zombie is not
    running, but its pid still answers ``os.kill(pid, 0)``, so only a reaped
    child passes. ``active_children`` is the second opinion, taken afterwards
    because asking it *reaps as a side effect*.
    """
    announced = supervised.announced()
    assert announced, (
        "no shard announced itself: the supervision assertions below would be "
        "vacuous, and the gauntlet's children never started"
    )
    for pid in announced:
        # Before `active_children()`, which reaps as a side effect: asking it
        # first would *create* the state this assertion is checking for.
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    # Filtered to this gauntlet's own pids: another module's leaked worker is
    # somebody else's failure, and `kill_stragglers` exists because they happen.
    live = [
        child.pid
        for child in multiprocessing.active_children()
        if child.pid in set(announced)
    ]
    assert live == [], f"live shard children survived run_gauntlet: {live}"


def test_a_worker_that_exits_abruptly_is_bounded(
    supervised: _Supervised, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CP-04's witness, inverted: an ``os._exit`` in a shard ends the gauntlet.

    ``pool.map`` had no reconciliation for a child that dies without returning,
    so this hung forever. It now raises ``GauntletLost`` -- a *typed* error, so
    the campaign's boundary handler writes a crash ``PAUSED`` naming a dead
    shard rather than "something threw" -- promptly, and with no process left
    behind.
    """
    from komugi.campaign import gauntlet as gauntlet_module

    monkeypatch.setattr(gauntlet_module, "run_shard", shard_that_exits_abruptly)

    started = time.monotonic()
    with pytest.raises(GauntletLost) as raised:
        _play(supervised)
    elapsed = time.monotonic() - started

    # The *type* is what pins promptness: waiting the dead child out would have
    # produced ``GauntletStalled`` at the deadline, not ``GauntletLost``. The
    # wall bound is the coarse backstop, generous enough that a slow box
    # importing torch in two fresh interpreters cannot fail it spuriously.
    assert elapsed < SHORT_DEADLINE_S + 60.0, (
        f"a dead child was still unreconciled after {elapsed:.1f}s"
    )
    assert "died without returning" in str(raised.value)
    _assert_no_child_survived(supervised)


def test_a_stalled_worker_is_bounded(
    supervised: _Supervised, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A shard that is alive and not returning is ended at its deadline.

    The distinction from the test above is the whole reason there are two
    exception types: nothing is wrong with the *process* here, so no
    reconciliation can detect it and only a deadline can. The stub would sleep
    for fifteen minutes; the supervisor gives it thirty seconds and then kills
    it.
    """
    from komugi.campaign import gauntlet as gauntlet_module

    monkeypatch.setattr(gauntlet_module, "run_shard", shard_that_stalls)

    started = time.monotonic()
    with pytest.raises(GauntletStalled) as raised:
        _play(supervised)
    elapsed = time.monotonic() - started

    assert elapsed >= SHORT_DEADLINE_S, (
        f"the gauntlet gave up after {elapsed:.1f}s, before its "
        f"{SHORT_DEADLINE_S:g}s deadline: a slow shard is not a stalled one"
    )
    assert elapsed < SHORT_DEADLINE_S + 30.0, (
        f"the gauntlet took {elapsed:.1f}s to enforce a {SHORT_DEADLINE_S:g}s "
        f"deadline"
    )
    assert elapsed < STALL_SLEEP_S, "the stub outlived the supervisor"
    assert f"{SHORT_DEADLINE_S:g}s" in str(raised.value)
    _assert_no_child_survived(supervised)


def test_a_raising_worker_still_crash_pauses(
    supervised: _Supervised, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A shard that raises is re-raised unchanged -- the KOM-129 path, unbroken.

    This case was already bounded before the supervisor existed, and pinning it
    is the point: the new machinery must not convert a shard's own ``ValueError``
    into a ``GauntletLost``, because the two send an operator to different
    places (a traceback versus ``dmesg``). It reaches ``run_campaign``'s boundary
    ``except Exception`` exactly as it did, and that handler writes the crash
    ``PAUSED``.
    """
    from komugi.campaign import gauntlet as gauntlet_module

    monkeypatch.setattr(gauntlet_module, "run_shard", shard_that_raises)

    with pytest.raises(ValueError, match=SHARD_FAILURE) as raised:
        _play(supervised)

    assert type(raised.value) is ValueError, (
        f"the shard's own {ValueError.__name__} arrived as "
        f"{type(raised.value).__name__}: a bug in a shard's work is not an "
        f"infrastructure failure, and the two send an operator to different "
        f"places"
    )
    _assert_no_child_survived(supervised)


# KOM-236: the positive control really plays the games, which the gate sandbox
# cannot afford; the negative controls above stub their shards and still run.
@pytest.mark.sandbox_unsafe
def test_a_healthy_gauntlet_completes(
    supervised: _Supervised, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The positive control: supervision does not cost a gauntlet its result.

    Without this, every assertion above is satisfied by a ``run_gauntlet`` that
    raises unconditionally. The shards' games merge into the serial match, the
    score is read off it, and the children are gone afterwards either way.
    """
    from komugi.campaign import gauntlet as gauntlet_module

    monkeypatch.setattr(gauntlet_module, "run_shard", shard_that_plays)

    played = _play(supervised)

    assert played.summary.games == SMALL_GAMES, (
        f"the merged summary counts {played.summary.games} games for a "
        f"{SMALL_GAMES}-game match: a shard's result was dropped"
    )
    assert played.generation == 0
    assert played.wall_s > 0.0
    assert not played.uninformative
    # Every game is a white win and the seats are swapped, so both entrants
    # score exactly half: a value that pins the merge, not the arena.
    assert played.score == 0.5
    # Between one and one-per-shard: the executor is entitled to reuse an idle
    # worker for a second task, so the assertion is that the *work* fanned out
    # (the games above) and that whatever ran is gone (below), not that the
    # pool chose to spawn a second process before the first went idle.
    assert 1 <= len(supervised.announced()) <= SMALL_SHARDS, (
        f"{len(supervised.announced())} shard process(es) ran for "
        f"{SMALL_SHARDS} shards"
    )
    _assert_no_child_survived(supervised)


# KOM-236: a wall-clock bound measured over really-played shards; the gate
# sandbox's single BLAS thread makes the timing meaningless there.
@pytest.mark.sandbox_unsafe
def test_the_whole_gauntlet_is_bounded_by_one_deadline(
    supervised: _Supervised, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bound is one absolute deadline for the fan-out, not one per shard.

    The shards run **concurrently**, so their walls overlap and the deadline is
    a statement about the gauntlet, not about a shard. Collecting them with a
    fresh ``future.result(timeout=deadline)`` per future said otherwise: shard 0
    could spend most of the bound and shard 1 would then be given the whole of
    it again, so the production 16-shard gauntlet was really bounded at
    ``16 x`` the deadline -- eight hours under the old 1800 s pin -- by a
    comment claiming one (KOM-180 impl review F6).

    Two shards: the first returns after most of the deadline, the second never
    returns. Under one absolute deadline the call ends at the deadline; under a
    per-future timer it ends at roughly twice it. The assertion is the elapsed
    wall, because that *is* the guarantee -- and the deadline is the injected
    short one, so the test costs seconds rather than the production hour.

    The pin on the production constant itself lives in
    ``tests/campaign/unit/test_pinned.py``, as
    ``test_the_gauntlet_deadline_clears_the_worst_measured_gauntlet``: it is
    checked there against the measured gauntlet distribution, rather than here
    against a remembered number (review F2 -- the test this one replaced
    hardcoded 120 s and was satisfied by any pin above 1200 s).
    """
    from komugi.campaign import gauntlet as gauntlet_module

    monkeypatch.setattr(
        gauntlet_module, "run_shard", shard_that_returns_late_then_stalls
    )

    started = time.monotonic()
    with pytest.raises(GauntletStalled):
        _play(supervised)
    elapsed = time.monotonic() - started

    assert elapsed < SHORT_DEADLINE_S + ONE_DEADLINE_SLACK_S, (
        f"the gauntlet took {elapsed:.1f}s to enforce a {SHORT_DEADLINE_S:g}s "
        f"deadline over {SMALL_SHARDS} shards: the deadline is being restarted "
        f"per shard, so the whole call is bounded by shards x deadline rather "
        f"than by the one deadline its docstring promises"
    )
    assert elapsed >= SHORT_DEADLINE_S, (
        f"the gauntlet gave up after {elapsed:.1f}s, inside its "
        f"{SHORT_DEADLINE_S:g}s deadline: a shard that returned late is not a "
        f"stalled gauntlet, and the bound must be spent before it is enforced"
    )
    assert elapsed < STALL_SLEEP_S, "the stub outlived the supervisor"
    _assert_no_child_survived(supervised)
