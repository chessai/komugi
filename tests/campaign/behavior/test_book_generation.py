"""The campaign's two book build sites (KOM-014 §8 #30, #31, #31b, #32, #37).

Four claims, and each is about a *seam* rather than about the builder (whose own
suite is ``tests/selfplay/unit/test_book_unit.py``):

* one book per iteration, named for the iteration it is **for**, and none for
  iteration 0 -- which has no source below it and is bookless by design;
* a resumed iteration rebuilds its book **byte-identically**, which is what lets
  the fallback site overwrite rather than trust the file it finds;
* the **primary** site (end of iteration ``i-1``, after training) and the
  **fallback** site (top of iteration ``i``) produce the same bytes over the
  same shard tree -- §2.1 argues this and rev 2's acceptance never tested it;
* a corrupt book fails **before any worker is spawned**, as one exception rather
  than as ``max_shard_restarts`` worker deaths with the real error N
  incarnations deep.

**Self-play is stubbed, and the stub is the point.** ``run_selfplay`` is
replaced by a function that plants a deterministic shard tree and records the
``SelfPlayConfig`` it was handed. Everything else is the real campaign: the real
resume logic, the real training iteration, the real window build, the real
builder, the real validation. Playing real games would add spawned processes and
minutes of wall clock to nodes whose subject is *where the build happens and
what the workers are handed* -- and would make the recorded config, which is how
#32 sees the override at all, unreachable.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import pytest

import campaign_rig as rig

camp = pytest.importorskip("komugi.campaign.loop")

if not hasattr(camp, "run_campaign"):
    pytest.skip(
        "komugi.campaign.loop exports no run_campaign yet (KOM-031)",
        allow_module_level=True,
    )

import torch

from komugi.engine.game import Game
from komugi.engine.observation import CONTROL_EMPTY, CONTROL_SIZE, PLANE_MAP_VERSION
from komugi.selfplay.adapter import GameState
from komugi.selfplay.book import book_path_for, load_book, running_gungi_mode
from komugi.selfplay.orchestrate import RunSummary
from komugi.selfplay.records import SCHEMA, num_actions
from komugi.train.shards import ShardWriter

_SUPPORT_WIDTH: Final[int] = 16
"""Legal actions a planted ``policy_target`` puts mass on -- the builder's own
``SUPPORT_TOP_K``, so every candidate draws from a full-width support."""

_SHAPE: Final[tuple[float, ...]] = (0.70, 0.10, 0.05) + (0.0115,) * 13
"""§3.2's worked peaked restriction, so the entries carry a defined flattening
ratio rather than the ``null`` a one-action support would produce."""

_GAMES: Final[int] = 20
_PLIES: Final[int] = 6
"""Twenty six-ply games. Six plies gives eligible deviations ``d in {3, 4}``,
and twenty games x 2 deviations x the 8-per-game quota is comfortably past the
32-entry floor -- so a book IS written, which every node here depends on."""

_BOOKLESS_PLIES: Final[int] = 4
"""Four-ply games have **no** eligible deviation (``[3,11]`` intersected with
``[0, n_s-2]`` is empty), so a tree of them yields no entry and no file. That is
how #37 keeps a hand-planted corrupt book on disk: the primary site tries,
writes nothing, and the corrupt file is still there when iteration 1 reads it."""

_SAMPLES_PER_SHARD: Final[int] = 64
_ITERATIONS: Final[int] = 2


class _SimulatedKill(RuntimeError):
    """What the stub raises to stop a campaign where a kill would have."""


@dataclass(frozen=True)
class _Call:
    """One recorded entry into the stubbed ``run_selfplay``."""

    iteration: int
    shard_root: Path
    book_path: str
    book_fraction: float


def _plant(directory: Path, seed: int, games: int, plies: int) -> int:
    """Write `games` real, replayable games into `directory`, returning the rows.

    Every action is drawn from the engine's own legal mask at the position it is
    played into, and every ``policy_target`` puts its mass on legal actions of
    that same position -- which is what makes the prefixes the builder mines
    from this tree replay, and therefore what makes the campaign's §4.1
    validation (which replays one) a check rather than a formality.
    """
    written = 0
    worker_dir = directory / "worker_0_r0"
    with ShardWriter(
        worker_dir,
        SCHEMA,
        samples_per_shard=_SAMPLES_PER_SHARD,
        flush_every_seconds=86_400.0,
    ) as writer:
        for index in range(games):
            rng = np.random.default_rng(seed + index)
            state = GameState(Game(), 0.0)
            for ply in range(plies):
                assert not state.is_terminal()
                legal = np.nonzero(state.legal_mask())[0]
                assert legal.size >= _SUPPORT_WIDTH
                target = torch.zeros(num_actions(), dtype=torch.float32)
                for slot, mass in enumerate(_SHAPE):
                    target[int(legal[slot])] = mass
                action = int(legal[int(rng.integers(0, legal.size))])
                writer.append(
                    {
                        "observation": torch.zeros(
                            Game.observation_size(), dtype=torch.float32
                        ),
                        "policy_target": target,
                        "wdl_target": torch.tensor(
                            [1.0, 0.0, 0.0], dtype=torch.float32
                        ),
                        "mlh_target": torch.tensor(0.5, dtype=torch.float32),
                        "policy_kl": torch.tensor(0.0, dtype=torch.float32),
                        "root_value": torch.tensor(0.0, dtype=torch.float32),
                        "search_mode": torch.tensor(0, dtype=torch.int64),
                        "to_play": torch.tensor(state.to_play(), dtype=torch.int64),
                        "ply": torch.tensor(ply, dtype=torch.int64),
                        "game_id": torch.tensor(
                            seed * 1000 + index, dtype=torch.int64
                        ),
                        "sims_used": torch.tensor(32, dtype=torch.int64),
                        "plane_map_version": torch.tensor(
                            PLANE_MAP_VERSION, dtype=torch.int64
                        ),
                        "final_control": torch.full(
                            (CONTROL_SIZE,), CONTROL_EMPTY, dtype=torch.int64
                        ),
                        "action": torch.tensor(action, dtype=torch.int64),
                        # KOM-005b's three columns, at the real shapes and with
                        # distinguishable values -- a zero-fill here would be
                        # indistinguishable from a column the writer dropped.
                        "termination": torch.tensor(0, dtype=torch.int64),
                        "short_term_value": torch.tensor(
                            [0.5, -0.25, 0.75 - 0.25 * (ply % 3)],
                            dtype=torch.float32,
                        ),
                        "draft_outcome": (
                            torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
                            if ply < 2
                            else torch.zeros(3, dtype=torch.float32)
                        ),
                    }
                )
                written += 1
                state.play(action)
    return written


def _stub_selfplay(
    calls: list[_Call], *, games: int, plies: int, fail_from: int | None
) -> Callable[..., RunSummary]:
    """Return a ``run_selfplay`` replacement that plants shards and records.

    ``fail_from`` is the kill: from that iteration onwards the stub raises
    instead of planting, which is how a campaign that "died mid-iteration" is
    produced without killing an interpreter.
    """

    def fake(
        factory: Any, orchestrator: Any, selfplay: Any, inference: Any
    ) -> RunSummary:
        directory = Path(orchestrator.shard_root)
        iteration = int(directory.parent.name.removeprefix("iter_"))
        calls.append(
            _Call(
                iteration=iteration,
                shard_root=directory,
                book_path=selfplay.book_path,
                book_fraction=selfplay.book_fraction,
            )
        )
        if fail_from is not None and iteration >= fail_from:
            raise _SimulatedKill(f"self-play killed at iteration {iteration}")
        rows = _plant(directory, 100 + iteration, games, plies)
        return RunSummary(
            games_completed=games,
            rows_written=rows,
            shard_dirs=(str(directory / "worker_0_r0"),),
            restarts=0,
            worker_incarnations=(0,),
        )

    return fake


def _run(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    iterations: int = _ITERATIONS,
    games: int = _GAMES,
    plies: int = _PLIES,
    fail_from: int | None = None,
    selfplay: Any = None,
) -> tuple[list[_Call], BaseException | None]:
    """Run a campaign over `root` with self-play stubbed; return calls and error."""
    calls: list[_Call] = []
    monkeypatch.setattr(
        camp,
        "run_selfplay",
        _stub_selfplay(calls, games=games, plies=plies, fail_from=fail_from),
    )
    error: BaseException | None = None
    with rig.small_net_env():
        try:
            camp.run_campaign(
                rig.campaign_config(root, iterations),
                rig.tiny_selfplay_config() if selfplay is None else selfplay,
                orchestrator_overrides=rig.orchestrator_overrides(),
                inference=rig.inference_config(),
            )
        except BaseException as exc:  # noqa: BLE001 -- returned, not swallowed
            error = exc
    return calls, error


# ---------------------------------------------------------------------------
# #30, #32 -- one shared two-iteration campaign
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def two_iterations(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, list[_Call]]:
    """A complete two-iteration campaign, run once, with a caller-supplied book.

    The caller's ``book_path`` points at a file that does not exist: if the
    campaign passed it through, iteration 0's worker would fail to load it and
    the run would die. It finishing at all is half of #32's claim; the recorded
    configs are the other half.
    """
    root = tmp_path_factory.mktemp("book-campaign")
    monkeypatch = pytest.MonkeyPatch()
    selfplay = rig.tiny_selfplay_config(
        book_path=str(root / "caller-supplied-book.json"), book_fraction=0.25
    )
    try:
        calls, error = _run(root, monkeypatch, selfplay=selfplay)
    finally:
        monkeypatch.undo()
    assert error is None, f"the campaign did not finish: {error!r}"
    return root, calls


def test_a_two_iteration_campaign_writes_one_book_per_iteration(
    two_iterations: tuple[Path, list[_Call]],
) -> None:
    """#30: ``books/gen_1.json`` exists, sourced from iteration 0, and nothing else.

    ``gen_0.json`` is absent because iteration 0 has no source below it, and
    ``gen_2.json`` is absent because the primary site is skipped on the last
    iteration -- building a book for an iteration that never runs is ~5 GB of
    decompression for a file no worker opens (R10b).
    """
    root, _calls = two_iterations
    books = sorted(path.name for path in (root / "books").iterdir())
    assert books == ["gen_1.json"], f"unexpected book files: {books}"
    assert not book_path_for(root, 0).exists()
    assert not book_path_for(root, _ITERATIONS).exists()

    built = load_book(book_path_for(root, 1))
    assert built.generation == 1
    assert built.source_iterations == (0,)
    assert built.gungi_mode == running_gungi_mode()
    assert built.action_size == num_actions()
    assert len(built.entries) >= 32
    for entry in built.entries:
        assert entry.source_iteration == 0
        assert 4 <= entry.plies <= 12


def test_the_campaign_owns_the_book_path(
    two_iterations: tuple[Path, list[_Call]],
) -> None:
    """#32: the caller's ``book_path`` never reaches a worker.

    ``run_campaign`` derives it per iteration exactly as it derives ``run_seed``
    and the shard root, and the two iterations show both branches: iteration 0
    is bookless (path ``""``, and the fraction forced to zero with it, because
    ``SelfPlayConfig`` refuses a positive fraction with no book), iteration 1
    gets the file the campaign itself built, at the caller's fraction.
    """
    root, calls = two_iterations
    assert [call.iteration for call in calls] == [0, 1]
    caller_path = str(root / "caller-supplied-book.json")
    assert not Path(caller_path).exists()
    for call in calls:
        assert call.book_path != caller_path, (
            "the caller's book_path reached a worker; it is the campaign's to "
            "derive, like the seed and the shard root"
        )

    assert calls[0].book_path == ""
    assert calls[0].book_fraction == 0.0
    assert calls[1].book_path == str(book_path_for(root, 1))
    assert calls[1].book_fraction == 0.25


# ---------------------------------------------------------------------------
# #31, #31b -- the two sites, and a resume
# ---------------------------------------------------------------------------


def test_a_resumed_iteration_rebuilds_the_same_book(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#31: two fallback builds over one shard tree are byte-identical.

    The campaign runs one iteration (the primary site is skipped on the last
    one), so ``gen_1.json`` does not exist yet. Two subsequent runs each enter
    iteration 1 as their resume boundary, rebuild the book at the fallback site
    and then die in self-play -- the second one after the first's file has been
    deleted, as a kill would have. Byte-identity is what lets the fallback site
    *overwrite* the file it finds instead of trusting it: a book that depended
    on which process wrote it could not be rebuilt at all.
    """
    first, error = _run(tmp_path, monkeypatch, iterations=1)
    assert error is None, f"the seeding iteration failed: {error!r}"
    assert [call.iteration for call in first] == [0]
    assert not book_path_for(tmp_path, 1).exists(), (
        "the primary site built a book for iteration 1 while the campaign's own "
        "last iteration was 0"
    )

    _calls, killed = _run(tmp_path, monkeypatch, iterations=2, fail_from=1)
    assert isinstance(killed, _SimulatedKill)
    path = book_path_for(tmp_path, 1)
    assert path.exists(), "the fallback site did not build the missing book"
    before = path.read_bytes()

    path.unlink()
    _again, killed_again = _run(tmp_path, monkeypatch, iterations=2, fail_from=1)
    assert isinstance(killed_again, _SimulatedKill)
    assert path.exists()
    assert path.read_bytes() == before, (
        "the resumed campaign rebuilt a different book from the same shard tree "
        "and the same seed"
    )


def test_the_primary_and_fallback_sites_agree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#31b: the same bytes whichever site built them (R10a).

    Run one: iteration 0 trains, the **primary** site writes ``gen_1.json``, and
    iteration 1 dies in self-play -- but only after the campaign validated that
    primary book and handed its path to the (stubbed) workers, which is itself
    the claim that the primary artefact is usable. Run two: the file is gone, so
    iteration 1's **fallback** site rebuilds it over the identical, unmodified
    shard tree. The two files are compared as bytes.
    """
    calls, killed = _run(tmp_path, monkeypatch, iterations=2, fail_from=1)
    assert isinstance(killed, _SimulatedKill)
    assert [call.iteration for call in calls] == [0, 1]
    path = book_path_for(tmp_path, 1)
    assert path.exists(), "the primary site wrote no book at the end of iteration 0"
    primary = path.read_bytes()
    assert calls[1].book_path == str(path), (
        "iteration 1 was not handed the primary site's book, so this node would "
        "compare a file nobody used"
    )

    path.unlink()
    again, killed_again = _run(tmp_path, monkeypatch, iterations=2, fail_from=1)
    assert isinstance(killed_again, _SimulatedKill)
    assert [call.iteration for call in again] == [1], (
        "the resumed campaign replayed iteration 0, so its shard tree is not the "
        "one the primary book was built from"
    )
    assert path.read_bytes() == primary


# ---------------------------------------------------------------------------
# #37 -- a corrupt book stops the iteration before it spawns anything
# ---------------------------------------------------------------------------


def _write_corrupt_book(path: Path) -> None:
    """Write a syntactically fine book built for a different action space."""
    payload: dict[str, Any] = {
        "book_version": 1,
        "generation": 1,
        "source_iterations": [0],
        "gungi_mode": running_gungi_mode(),
        # The one field that actually protects a prefix: an action id means
        # nothing across an encoding change, so a stale book would replay a
        # plausible WRONG line rather than fail.
        "action_size": num_actions() + 1,
        "plane_map_version": PLANE_MAP_VERSION,
        "source_schema_fingerprint": SCHEMA.fingerprint,
        "temperature": 2.0,
        "support_epsilon": 0.001,
        "support_top_k": 16,
        "support_floor": 0.0001,
        "support_min_k": 2,
        "target_shard_budget": 32,
        "cap": 1024,
        "min_plies": 4,
        "max_plies": 12,
        "entries_per_source_game": 8,
        "build_seed": 1,
        "bit_generator": "PCG64",
        "entries": [
            {
                "id": "gen0001-000000-deadbeef",
                "actions": [0, 1, 2, 3],
                "plies": 4,
                "source_iteration": 0,
                "source_game_id": 0,
                "deviation_ply": 3,
                "deviated": True,
                "support_size": 16,
                "support_fallback": False,
                "truncated_mass": 0.0,
                "drawn_target_prob": 0.5,
                "restricted_entropy_bits": 1.0,
                "tempered_entropy_bits": 2.0,
                "flattening_ratio": 0.75,
            }
        ],
        "stats": {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def test_a_corrupt_book_fails_before_any_worker_is_spawned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#37: one exception, named, instead of a restart storm (N4).

    Red against load-time-only refusal. The worker's own refusal (§6.3) is on
    the wrong side of the spawn boundary to be the only check: a book that
    exists and cannot be loaded fails inside *every* worker, and the
    orchestrator restarts each shard up to ``max_shard_restarts`` times under a
    fresh incarnation before the iteration fails -- so the operator sees N dead
    workers and the real error buried in the last one's traceback.

    The tree is planted with four-ply games so the primary site finds no
    mineable deviation and writes nothing: the corrupt file planted here is
    therefore still on disk when iteration 1 reads it, which is the only way to
    reach the state this node is about.
    """
    _write_corrupt_book(book_path_for(tmp_path, 1))
    calls, error = _run(
        tmp_path, monkeypatch, iterations=2, games=8, plies=_BOOKLESS_PLIES
    )

    assert isinstance(error, ValueError), (
        f"a book built for a different action space did not stop the campaign: "
        f"{error!r}"
    )
    message = str(error)
    assert str(book_path_for(tmp_path, 1)) in message, (
        "the refusal must name the book path; 'the book is wrong' is not a fix"
    )
    assert str(num_actions()) in message and str(num_actions() + 1) in message

    assert [call.iteration for call in calls] == [0], (
        "iteration 1 entered self-play with a corrupt book: the refusal happened "
        "on the worker side of the spawn boundary, which is the restart storm "
        "this node exists to prevent"
    )
