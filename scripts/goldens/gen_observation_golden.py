#!/usr/bin/env python3
"""Generate KOM-002's observation goldens, G1 and G2 (design §6.1-§6.5).

The design's §6 asks for **two** golden files of per-ply, per-plane digests:

* **G1** -- ``tests/encoding/goldens/observation_g1.json`` -- the repetition /
  progress golden.  Its primary source is the v1 1600-simulation shuffle game
  (``--source v1-1600sim``); its fallback is the ``PURE_SHUFFLE`` line from
  ``tests/rules/fixtures.py`` driven to repetition occurrence 3
  (``--source fallback-shuffle``).  Which one was used is recorded in the
  manifest's ``source`` field, which acceptance #25 asserts -- so the golden
  can never quietly become the fallback while claiming to be the v1 game.
  §6.1's obligation stands on top of that: when the fallback is used, the
  delegation log in ``tickets/KOM-002.md`` must record WHAT was searched under
  ``gungi-az-runs`` and WHY it failed, because "we looked and it is gone" and
  "nobody looked" are otherwise indistinguishable.
* **G2** -- ``tests/encoding/goldens/observation_g2.json`` -- the last-move
  golden.  It must contain at least one ply of each move class (#24): route,
  tsuke, capture, betray, arata (block B), arata-with-conversion (``bsel=1``)
  and block C (終).  G2 is always generated; it takes no ``--source``.

**Naming (design rev 4).**  §6.2 called these files
``observation_shuffle_1600.json`` and ``observation_movetypes.json``.  They are
``observation_g1.json`` / ``observation_g2.json`` here, because the fallback
branch makes the first name a lie about provenance and the second says less
than "G2" does about which golden it is.  ``--out-dir`` moves them; the names
are fixed.

**G2 is SEGMENTED, and G1 is not.**  A betray is rare enough that the oracle
had to play 1200 games to harvest 223 betray plies
(``tests/engine/goldens/MANIFEST.json``), so no single game reliably contains
one alongside a full draft.  G2 is therefore a sequence of segments, each with
its own starting position, and every ply record carries a ``segment`` index
into the file's top-level ``segments`` array.  The per-ply record is otherwise
§6.3's schema verbatim.  G1 is one segment; the field is present there too so
the two files read the same way.

G2's segments, in the order they are appended:

1. ``advanced-golden-N`` -- the spine, a replayed ``playouts_advanced.json``
   game (``--advanced-game``).  Its draft supplies block B and block C (終);
   its middlegame supplies routes, tsuke and captures.
2. ``betray-fragment-N`` -- position-anchored fragments from
   ``playouts_betray.json`` (``--betray-games``), each replaying into a betray
   ply.
3. ``arata-conversion`` -- a ONE-PLY **scripted** segment, appended when the
   segments above left ``arata_conversion`` uncovered (they always do).  A
   conversion is a tactician (謀) dropped from hand onto a tower whose TOP is
   friendly, below the mode's tier cap, and which still holds a convertible
   enemy underneath -- a conjunction of preconditions a playout essentially
   never walks into, so the seeded tail below cannot be relied on to find one:
   at ``--max-seek-plies 4000`` it found none under any of four seeds.  The
   segment is therefore *found* rather than *sought*: the vendored corpora are
   scanned in a fixed order for the first ply whose ORACLE legal list contains
   an arata-with-conversion SAN, and that ply's ``fen_before`` plus that one
   encoded action is the entire segment.  No RNG, so it is byte-stable.
4. ``class-seeking`` -- the seeded tail (``--seed``, ``--max-seek-plies``),
   appended only if a required class is STILL missing.  With the scripted
   segment in place it is normally not appended at all, which leaves G2 free
   of RNG end to end.

A class that none of the four covers is still an error, not a quietly
under-covered file (#24).

Per ply, after the move that reached it (ply 0 has ``action: null``)::

    {
      "segment": 0,
      "ply": 17,
      "action": 6813,
      "fen": "...",
      "turns": 17,
      "repetition_count": 1,
      "progress_plies": 5,
      "last_move": [40, 1, 31, 2],
      "planes": ["3f9a1c22", ...]      // 128 entries, sha256(plane bytes)[:8]
    }

A digest per PLANE rather than one hash per observation, because the failure
this ticket exists to catch is a plane-map regression and a whole-array hash
cannot name the plane that moved (§6.3).  The digest is over the plane's
``float32`` bytes in C order: every value in the map is an exact binary ratio
and nothing is summed, so it round-trips identically on any platform.

CLI
---

::

    # G1 from the recovered v1 game, plus G2:
    python scripts/goldens/gen_observation_golden.py \
        --source v1-1600sim path/to/game.json

    # G1 from the in-repo fallback, plus G2:
    python scripts/goldens/gen_observation_golden.py --source fallback-shuffle

    # options
    --out-dir DIR          where the two files are written
                           (default: tests/encoding/goldens)
    --advanced-game N      which playouts_advanced.json game spines G2 (0)
    --betray-games N       how many betray fragments to append to G2 (1)
    --seed N               seed for G2's class-seeking tail (20260904)
    --max-seek-plies N     cap on that tail (400)
    --only {g1,g2}         write just one of the two files (``--source`` is
                           required unless ``--only g2`` is given)

The ``--source v1-1600sim`` argument is a JSON file holding the recovered game:
``{"mode": "advanced", "start_fen": null, "moves": ["SAN", ...]}`` -- or
``"actions": [id, ...]`` if the recovery produced action ids directly.
``start_fen: null`` means "the mode's start position".

Discipline (§6.5): this generator is tracked in-repo, and regenerating either
golden requires a new dated entry in ``tests/encoding/goldens/GENERATION.md``
naming the reviewer.  A golden generated by the code it pins proves nothing on
its own -- the hand-written expected values acceptance #26 requires are what
make it mean something, and they are written by reading the board, not by
copying this script's output.

Run from the repo root, inside the dev shell::

    PYTHONPATH=$PWD/src python scripts/goldens/gen_observation_golden.py ...

Exit codes: 0 written, 2 the requested golden could not be built (missing move
class, unreachable repetition occurrence, illegal recovered move) -- never a
silently under-covered file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "tests" / "rules"))

from komugi.engine import action_encoding, observation  # noqa: E402
from komugi.engine.api import Gungi  # noqa: E402
from komugi.engine.cyboard import board_from_fen, board_to_fen  # noqa: E402
from komugi.engine.cymovegen import generate_all_moves  # noqa: E402
from komugi.engine.game import Game  # noqa: E402

import fixtures  # noqa: E402  (tests/rules is on the path above)

#: Hex characters kept per plane digest -- §6.3's ``sha256(plane bytes)[:8]``.
DIGEST_CHARS = 8

#: Move-type codes, ``cymovegen``'s ``MT_*`` (they live in a .pyx).
MT_ROUTE, MT_TSUKE, MT_CAPTURE, MT_BETRAY, MT_ARATA = 0, 1, 2, 3, 4

#: The classes #24 requires G2 to contain at least one ply of each.
REQUIRED_CLASSES = (
    "route",
    "tsuke",
    "capture",
    "betray",
    "arata_block_b",
    "arata_conversion",
    "shu_block_c",
)

GOLDENS = _ROOT / "tests" / "engine" / "goldens"

#: SAN markers (``cymovegen._build_san``): 新 opens every arata, and 返 -- with
#: the converted pieces' kanji after it -- is emitted for an arata whose
#: ``captured`` list is non-empty.  Both present is exactly the SAN shape of
#: what :func:`move_classes` calls ``arata_conversion``; it is a cheap
#: pre-filter over an oracle legal list, and every hit is re-checked by
#: decoding and classifying the action it resolves to.
SAN_ARATA = "新"
SAN_CONVERT = "返"

#: Corpora scanned, in this order, for a position with a legal
#: arata-with-conversion.  Both are advanced-mode (tier cap 3), the only mode
#: in which a tower can hold an enemy UNDER a friendly top -- at a cap of 2 a
#: tower whose top is friendly is a tower of one piece, so intro, beginner and
#: intermediate cannot contain a conversion at all.  ``playouts_betray.json``
#: leads because it was harvested for tall mixed towers (its MANIFEST records
#: 223 betray plies out of 1200 games), which is where conversions live.
CONVERSION_CORPORA = ("playouts_betray.json", "playouts_advanced.json")


class GenerationError(RuntimeError):
    """A golden could not be built as specified.  Never written partially."""


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def move_classes(action: int, record: dict[str, Any]) -> tuple[str, ...]:
    """The classes *action* belongs to, given its decoded move *record*.

    Flags, not a partition: a 終 that converts is a block-C ply AND an
    arata-with-conversion ply, and #24 counts both.  The move-type name is
    always the first element.
    """
    move_type = int(record["move_type"])
    board_moves = {
        MT_ROUTE: "route",
        MT_TSUKE: "tsuke",
        MT_CAPTURE: "capture",
        MT_BETRAY: "betray",
    }
    if move_type != MT_ARATA:
        if move_type not in board_moves:
            raise GenerationError(f"unknown move_type {move_type} in {record}")
        return (board_moves[move_type],)
    classes = ["arata"]
    classes.append(
        "shu_block_c" if action >= action_encoding.C_START else "arata_block_b"
    )
    if record["captured"]:
        classes.append("arata_conversion")
    return tuple(classes)


def record_classes(record: dict[str, Any]) -> tuple[str, ...]:
    """:func:`move_classes` for a cymovegen record, encoding it to get its id."""
    return move_classes(int(action_encoding.encode_cython_move(record)), record)


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


@dataclass
class Segment:
    """One continuously-played run of plies inside a golden file."""

    name: str
    source: str
    mode: str
    start_fen: str | None
    plies: list[dict[str, Any]] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)

    def header(self, index: int) -> dict[str, Any]:
        """The segment's entry in the file's top-level ``segments`` array."""
        return {
            "index": index,
            "name": self.name,
            "source": self.source,
            "mode": self.mode,
            "start_fen": self.start_fen,
            "plies": len(self.plies),
        }


def plane_digests(game: Game) -> list[str]:
    """``sha256(plane bytes)[:8]`` for each of the observation's planes."""
    planes = np.ascontiguousarray(game.observation(), dtype=np.float32)
    return [
        hashlib.sha256(np.ascontiguousarray(plane).tobytes()).hexdigest()[:DIGEST_CHARS]
        for plane in planes
    ]


def ply_record(
    game: Game, segment_index: int, ply: int, action: int | None
) -> dict[str, Any]:
    """§6.3's per-ply record for *game*'s CURRENT position."""
    last_move = game._last_move
    return {
        "segment": segment_index,
        "ply": ply,
        "action": action,
        "fen": board_to_fen(game._board),
        "turns": int(game.turns),
        "repetition_count": int(game.repetition_count()),
        "progress_plies": int(game._progress_plies),
        "last_move": list(last_move) if last_move is not None else None,
        "planes": plane_digests(game),
    }


def new_game(mode: str, start_fen: str | None) -> Game:
    """A ``Game`` on *start_fen*, or on *mode*'s start position."""
    if start_fen is not None:
        return Game(board_from_fen(start_fen))
    Game.set_start_mode(mode)
    return Game()


def play_into(
    segment: Segment, game: Game, actions: Iterable[int], segment_index: int
) -> None:
    """Play *actions*, recording ply 0 and every position they reach."""
    segment.plies.append(ply_record(game, segment_index, 0, None))
    for offset, action in enumerate(actions, start=1):
        board = game._board
        record = action_encoding.decode_cython_action(action, board)
        for name in move_classes(action, record):
            segment.counts[name] = segment.counts.get(name, 0) + 1
        game.play_action(action)
        segment.plies.append(ply_record(game, segment_index, offset, action))


def action_for_san(fen_before: str, san: str) -> int:
    """The action id of *san* at *fen_before*, via the reference generator.

    The same resolution ``tests/engine/test_action_encoding.py``'s round-trip
    uses, so a sequence that replays here replays there.
    """
    normalized = san.rstrip("=#")
    for move in Gungi(fen_before).moves({"verbose": True}):
        if move["san"] == normalized or move["san"] == san:
            return int(action_encoding.encode_move(move))
    raise GenerationError(f"SAN {san!r} is not legal at {fen_before!r}")


def golden_games(name: str) -> list[list[dict[str, Any]]]:
    """The ``games`` array of ``tests/engine/goldens/<name>``."""
    path = GOLDENS / name
    if not path.is_file():
        raise GenerationError(f"golden corpus {path} is not vendored")
    with path.open(encoding="utf-8") as handle:
        data: dict[str, Any] = json.load(handle)
    games: list[list[dict[str, Any]]] = data["games"]
    return games


# ---------------------------------------------------------------------------
# G1 -- the repetition / progress golden
# ---------------------------------------------------------------------------


def g1_from_v1(path: Path) -> Segment:
    """G1 from a recovered v1 1600-sim game (§6.1's primary source)."""
    with path.open(encoding="utf-8") as handle:
        recovered: dict[str, Any] = json.load(handle)
    mode = str(recovered.get("mode", "advanced"))
    start_fen = recovered.get("start_fen")
    segment = Segment(
        name="v1-1600sim", source="v1-1600sim", mode=mode, start_fen=start_fen
    )
    game = new_game(mode, start_fen)

    actions: list[int] = []
    if "actions" in recovered:
        actions = [int(value) for value in recovered["actions"]]
    else:
        replay = new_game(mode, start_fen)
        for san in recovered["moves"]:
            action = action_for_san(board_to_fen(replay._board), str(san))
            actions.append(action)
            replay.play_action(action)

    play_into(segment, game, actions, 0)
    return segment


def g1_fallback() -> Segment:
    """G1 from the in-repo ``PURE_SHUFFLE`` line, driven to occurrence 3.

    ``PURE_SHUFFLE`` is four routes -- white 帥 out and back, black 槍 out and
    back -- from position P, so it repeats the position every four plies and
    resets the progress clock never.  Two cycles reach occurrence 3, which is
    what acceptance #27 requires of G1; a third is played only if the count
    has not got there, and failing to reach it is an error rather than a
    quietly weaker golden.
    """
    segment = Segment(
        name="pure-shuffle",
        source="fallback-shuffle",
        mode="advanced",
        start_fen=fixtures.POSITION_P_FEN,
    )
    game = new_game("advanced", fixtures.POSITION_P_FEN)
    segment.plies.append(ply_record(game, 0, 0, None))

    ply = 0
    for _ in range(4):
        for action in fixtures.PURE_SHUFFLE:
            ply += 1
            record = action_encoding.decode_cython_action(action, game._board)
            for name in move_classes(action, record):
                segment.counts[name] = segment.counts.get(name, 0) + 1
            game.play_action(action)
            segment.plies.append(ply_record(game, 0, ply, action))
        if game.repetition_count() >= 3:
            break

    reached = max(int(entry["repetition_count"]) for entry in segment.plies)
    if reached < 3:
        raise GenerationError(
            f"the fallback shuffle reached occurrence {reached}, not 3; G1 "
            "would be a golden of nothing (#27)"
        )
    return segment


# ---------------------------------------------------------------------------
# G2 -- the last-move golden
# ---------------------------------------------------------------------------


def g2_advanced_spine(game_index: int) -> Segment:
    """G2's spine: a replayed advanced golden game.

    Advanced mode because its draft is where block B and block C (終) plies
    come from; the middlegame supplies routes, tsuke and captures.
    """
    games = golden_games("playouts_advanced.json")
    if not 0 <= game_index < len(games):
        raise GenerationError(
            f"playouts_advanced.json has {len(games)} games; asked for {game_index}"
        )
    plies = games[game_index]
    segment = Segment(
        name=f"advanced-golden-{game_index}",
        source="playouts_advanced.json",
        mode="advanced",
        start_fen=None,
    )
    actions = [action_for_san(ply["fen_before"], ply["chosen"]) for ply in plies]
    play_into(segment, new_game("advanced", None), actions, 0)
    return segment


def g2_betray_segments(limit: int, first_index: int) -> list[Segment]:
    """Betray fragments from ``playouts_betray.json``.

    That corpus is 121 short position-anchored games harvested from 1200
    played games precisely because a betray is rare (its MANIFEST records the
    yield).  Each fragment replays from its own ``fen_before``, which is why
    G2 is segmented at all.
    """
    segments: list[Segment] = []
    for game_plies in golden_games("playouts_betray.json"):
        if len(segments) >= limit:
            break
        start_fen = str(game_plies[0]["fen_before"])
        actions = [
            action_for_san(ply["fen_before"], ply["chosen"]) for ply in game_plies
        ]
        index = first_index + len(segments)
        candidate = Segment(
            name=f"betray-fragment-{index}",
            source="playouts_betray.json",
            mode="advanced",
            start_fen=start_fen,
        )
        play_into(candidate, new_game("advanced", start_fen), actions, index)
        if candidate.counts.get("betray", 0) > 0:
            segments.append(candidate)
    if not segments:
        raise GenerationError(
            "no game in playouts_betray.json replayed into a betray ply"
        )
    return segments


def conversion_sans(ply: dict[str, Any]) -> list[str]:
    """The SANs in *ply*'s oracle ``legal`` list shaped like a conversion.

    A pre-filter, not a decision: ``新…返…`` is necessary for an
    arata-with-conversion and nothing else produces it, but the caller still
    encodes the SAN and re-classifies the action rather than trusting a string.
    """
    legal = ply.get("legal") or ()
    return [
        text
        for text in (str(san) for san in legal)
        if text.startswith(SAN_ARATA) and SAN_CONVERT in text
    ]


def g2_arata_conversion_segment(segment_index: int) -> Segment:
    """A one-ply scripted segment whose single move converts (#24's ``bsel=1``).

    The spine, the betray fragments and the seeded tail all fail to produce an
    arata-with-conversion: it needs a tactician in hand AND a tower whose top
    is friendly AND an enemy still under it AND the hand holding at least as
    many of that enemy's type as the tower holds, and a random walk does not
    assemble that (4000 plies × 4 seeds found none).  So the position is taken
    from the corpora instead of searched for.

    The scan is over vendored files in a fixed order -- corpus, then game, then
    ply, then the ply's legal list, all in file order -- and stops at the first
    hit, so it is a pure function of what is checked in: no RNG, no ``--seed``,
    byte-identical run to run.  The segment is that ply's ``fen_before`` plus
    the one action, which replays through :func:`play_into` like any other.

    The SAN comes from the ORACLE's legal list, so it is legal in the full
    king-safety sense and not merely pseudo-legal, and it is resolved through
    :func:`action_for_san` -- the same resolution the spine and the fragments
    use, so a SAN that replays here replays in the encoding round-trip tests.
    """
    scanned: list[str] = []
    for corpus in CONVERSION_CORPORA:
        games = golden_games(corpus)
        scanned.append(f"{corpus} ({len(games)} games)")
        for game_index, plies in enumerate(games):
            for ply_index, ply in enumerate(plies):
                candidates = conversion_sans(ply)
                if not candidates:
                    continue
                start_fen = str(ply["fen_before"])
                board = board_from_fen(start_fen)
                for san in candidates:
                    action = action_for_san(start_fen, san)
                    record = action_encoding.decode_cython_action(action, board)
                    if "arata_conversion" not in move_classes(action, record):
                        continue
                    segment = Segment(
                        name="arata-conversion",
                        source=f"{corpus}[game {game_index}][ply {ply_index}] {san}",
                        mode="advanced",
                        start_fen=start_fen,
                    )
                    play_into(
                        segment,
                        new_game("advanced", start_fen),
                        [action],
                        segment_index,
                    )
                    return segment
    raise GenerationError(
        "no ply in " + ", ".join(scanned) + " offers a legal "
        "arata-with-conversion; #24's class cannot be scripted from the "
        "vendored corpora, so G2 is not written"
    )


def g2_class_seeking_tail(
    missing: Sequence[str], seed: int, max_plies: int, segment_index: int
) -> Segment:
    """A seeded playout that prefers legal moves of a still-missing class.

    The scripted segments are deterministic but cannot promise every class, so
    this walks an advanced game choosing, at each ply, a legal move of a class
    still missing when one exists and a seeded random legal move otherwise.  It
    stops the moment nothing is missing.

    It is the LAST resort and no longer covers ``arata_conversion``: preferring
    a class only helps when the walk reaches a position where that class is
    legal at all, and for a conversion it does not -- see
    :func:`g2_arata_conversion_segment`, which is appended before this and
    covers the class outright.
    """
    segment = Segment(
        name="class-seeking",
        source=f"seeded-playout(seed={seed})",
        mode="advanced",
        start_fen=None,
    )
    game = new_game("advanced", None)
    segment.plies.append(ply_record(game, segment_index, 0, None))
    rng = np.random.default_rng(seed)
    wanted = set(missing)

    for ply in range(1, max_plies + 1):
        if not wanted:
            break
        if np.asarray(game.win_state(), dtype=np.uint8).sum() > 0:
            break
        records = list(generate_all_moves(game._board))
        if not records:
            break
        chosen: dict[str, Any] | None = None
        for record in records:
            if wanted & set(record_classes(record)):
                chosen = record
                break
        if chosen is None:
            chosen = records[int(rng.integers(len(records)))]
        action = int(action_encoding.encode_cython_move(chosen))
        for name in move_classes(action, chosen):
            segment.counts[name] = segment.counts.get(name, 0) + 1
            wanted.discard(name)
        game.play_action(action)
        segment.plies.append(ply_record(game, segment_index, ply, action))

    return segment


def build_g2(args: argparse.Namespace) -> list[Segment]:
    """G2's segments, extended until every required class is covered."""
    segments = [g2_advanced_spine(int(args.advanced_game))]
    segments.extend(g2_betray_segments(int(args.betray_games), len(segments)))

    covered = total_counts(segments)
    if not covered.get("arata_conversion"):
        # Scripted, not sought: the seeded tail below cannot find a conversion
        # (see g2_arata_conversion_segment), so it is handed one first and is
        # then usually not needed at all.
        segments.append(g2_arata_conversion_segment(len(segments)))

    covered = total_counts(segments)
    missing = [name for name in REQUIRED_CLASSES if not covered.get(name)]
    if missing:
        segments.append(
            g2_class_seeking_tail(
                missing, int(args.seed), int(args.max_seek_plies), len(segments)
            )
        )
        covered = total_counts(segments)
        missing = [name for name in REQUIRED_CLASSES if not covered.get(name)]
    if missing:
        raise GenerationError(
            f"G2 does not cover {missing}; #24 requires at least one ply of "
            "every move class, so the file is not written.  Raise "
            "--max-seek-plies, change --seed, or add a scripted segment."
        )
    return segments


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def total_counts(segments: Sequence[Segment]) -> dict[str, int]:
    """Move-class counts summed over *segments*."""
    totals: dict[str, int] = {}
    for segment in segments:
        for name, count in segment.counts.items():
            totals[name] = totals.get(name, 0) + count
    return totals


def git_sha() -> str:
    """``HEAD``'s sha, or ``"unknown"`` outside a checkout."""
    try:
        result = subprocess.run(
            ["git", "-C", str(_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def write_golden(
    path: Path, segments: Sequence[Segment], source: str, seed: int | None
) -> None:
    """Write the golden file for *segments* -- manifest first, then plies."""
    planes = int(observation.observation_size()[0])
    payload = {
        "manifest": {
            "source": source,
            "plane_map_version": int(observation.PLANE_MAP_VERSION),
            "n_planes": planes,
            "git_sha": git_sha(),
            "generated_by": "scripts/goldens/gen_observation_golden.py",
            "command": " ".join(sys.argv),
            "seed": seed,
            "gungi_mode": segments[0].mode,
            "digest": f"sha256(plane float32 bytes, C order)[:{DIGEST_CHARS}]",
            "move_class_counts": total_counts(segments),
        },
        "segments": [
            segment.header(index) for index, segment in enumerate(segments)
        ],
        "plies": [ply for segment in segments for ply in segment.plies],
    }
    for ply in payload["plies"]:
        if len(ply["planes"]) != planes:
            raise GenerationError(
                f"ply {ply['ply']} recorded {len(ply['planes'])} digests for "
                f"{planes} planes"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1, sort_keys=False)
        handle.write("\n")
    print(f"wrote {path} ({len(payload['plies'])} plies, {planes} planes)")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """The CLI documented in the module docstring."""
    parser = argparse.ArgumentParser(
        description="Generate KOM-002's observation goldens G1 and G2."
    )
    parser.add_argument(
        "--source",
        choices=("v1-1600sim", "fallback-shuffle"),
        default=None,
        help="G1's source; 'v1-1600sim' takes the recovered game's path",
    )
    parser.add_argument(
        "v1_game",
        nargs="?",
        type=Path,
        help="recovered v1 1600-sim game (required by --source v1-1600sim)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_ROOT / "tests" / "encoding" / "goldens",
    )
    parser.add_argument("--advanced-game", type=int, default=0)
    parser.add_argument("--betray-games", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--max-seek-plies", type=int, default=400)
    parser.add_argument("--only", choices=("g1", "g2"), default=None)
    args = parser.parse_args(argv)
    if args.only != "g2" and args.source is None:
        parser.error("--source is required unless --only g2 is given")
    if args.source == "v1-1600sim" and args.v1_game is None:
        parser.error("--source v1-1600sim needs the recovered game's path")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    """Build the requested goldens, or explain why they cannot be built."""
    args = parse_args(argv)
    try:
        if args.only != "g2":
            g1 = (
                g1_from_v1(args.v1_game)
                if args.source == "v1-1600sim"
                else g1_fallback()
            )
            write_golden(
                args.out_dir / "observation_g1.json", [g1], args.source, None
            )
            if args.source == "fallback-shuffle":
                print(
                    "NOTE (§6.1): record in tickets/KOM-002.md's delegation log "
                    "what was searched under gungi-az-runs for the 1600-sim game "
                    "and why it failed -- acceptance #25 checks for that entry."
                )
        if args.only != "g1":
            write_golden(
                args.out_dir / "observation_g2.json",
                build_g2(args),
                "movetype-coverage",
                int(args.seed),
            )
    except GenerationError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(
        "Record a dated entry in tests/encoding/goldens/GENERATION.md naming "
        "the reviewer (§6.5)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
