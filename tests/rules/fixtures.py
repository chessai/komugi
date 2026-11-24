"""Shared fixtures for the repetition-repair suite (KOM-127).

Everything here is read off ``docs/design/rules-audit.md`` -- the authoritative
audit for the repetition defect (F1-F4) and its repair (KOM-027).  The two
aliasing positions ``P`` and ``Q`` and the scripted move sequences come from
that document's *Aliasing defect* section verbatim; the mate fixture is this
module's own construction (see ``MATE_AND_REPETITION_FEN``).

The suite is authored RED-FIRST, before KOM-027 exists.  Two consequences show
up in this file:

* The repaired API (``cyengine.repetition_key``, ``Game.repetition_count``) is
  reached through the ``repetition_key`` / ``repetition_count`` shims below.
  They go through an ``Any``-typed handle because the hand-written stub
  ``src/komugi/engine/cyengine.pyi`` predates the repair and this ticket may
  not touch ``src/**``; once KOM-027 lands and the stub declares the function,
  the shims can collapse to direct calls.
* ``has_repair()`` is the single gate every module that touches the new API
  skips on.

Coordinates are 1-based ``(rank, file)`` throughout, matching the audit's
``rank-file`` SAN.  Action ids are built by hand rather than by searching the
legal-move list, so that a sequence which stops being legal fails loudly in
``play_sequence`` instead of silently re-targeting some other move.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import numpy as np
import numpy.typing as npt
import pytest

from komugi.engine import action_encoding, api, cyengine, observation
from komugi.engine.cyboard import Board, board_from_fen
from komugi.engine.cymovegen import generate_all_moves
from komugi.engine.fen import INTRO_POSITION
from komugi.engine.game import Game
from komugi.engine.repetition import RepetitionRule
from komugi.engine.utils import LANCER, MARSHAL, RIDER, SOLDIER, WARRIOR

# ---------------------------------------------------------------------------
# Positions
# ---------------------------------------------------------------------------

#: Position P -- rules-audit "Scenario A: side-to-move alias".  Advanced mode,
#: both draft bits clear, empty hands.  White 帥(9-5) 侍(5-1); black 帥(1-5)
#: 槍(1-1).  A non-marshal stays on the board for each side so the
#: insufficient-material draw cannot fire.
POSITION_P_FEN: str = "4m3n/9/9/9/8W/9/9/9/4M4 -/- w 3 - 1"

#: Position Q -- rules-audit "Scenario B: hand-contents alias".  P plus a black
#: 馬 rider on 3-9 and two 兵 soldiers in White's hand.
POSITION_Q_FEN: str = "4m3n/9/r8/9/8W/9/9/9/4M4 D2/- w 3 - 1"

#: A position that is checkmate for the side to move (Black), used by RT-19 to
#: pin the F8 precedence question (U10).  NOT from the audit -- constructed
#: here, and verified against the *current* engine by an unconditional test in
#: ``test_fixtures_sanity.py`` so the construction cannot rot unnoticed.
#:
#: Black 帥 is on 1-1 with three flight squares: 1-2, 2-1 and 2-2.
#:   * 大(3-1) slides north up file 1 -- checks 1-1 and covers 2-1;
#:   * 兵(2-2) covers 1-2, and is itself defended by 中(3-3) on the long
#:     north-east diagonal, so 帥取(2-2) walks into check;
#:   * the mode is **beginner** (FEN field 4 = ``1``) precisely so that
#:     ``marshal_can_stack`` is False (``cymovegen``: ``b.mode >= 2``): in
#:     advanced mode Black would escape by 帥付(2-2), perching on tier 2 where
#:     neither tier-1 attacker can reach it.
#: Black has no other piece and an empty hand, so there is no block and no
#: counter-capture.  Mode is not part of the repetition key, so using beginner
#: mode here costs the fixture nothing.
MATE_AND_REPETITION_FEN: str = "8m/7D1/6I1G/9/9/9/9/9/4M4 -/- b 1 - 1"

#: ``hand_arr`` row for 兵 soldier, and the white/black column indices.
SOLDIER_INDEX: int = action_encoding.PIECE_INDEX[SOLDIER]
WHITE_COLOR: int = 0
BLACK_COLOR: int = 1

Square = tuple[int, int]

# ---------------------------------------------------------------------------
# Action construction
# ---------------------------------------------------------------------------


def sq_index(square: Square) -> int:
    """Map a 1-based ``(rank, file)`` to the 0..80 index the encoding uses.

    Mirrors ``action_encoding._RF_TO_SQ`` / ``cymovegen._sq_index``: squares
    run rank 1->9, and within a rank file 9->1.
    """
    rank, file = square
    if not (1 <= rank <= 9 and 1 <= file <= 9):
        raise ValueError(f"square out of range: {square!r}")
    return (rank - 1) * 9 + (9 - file)


def move_action(src: Square, dst: Square) -> int:
    """Action id for a NO-STACK board move (route or capture) ``src`` -> ``dst``.

    Block A0.  Every board move in the audit's scripted lines is either a route
    to an empty square or a capture of a lone enemy piece, so none of them needs
    the stacking block A1.
    """
    return action_encoding.A0_START + sq_index(src) * 81 + sq_index(dst)


def drop_action(piece_kanji: str, dst: Square, bsel: int = 0) -> int:
    """Action id for a post-draft 新 (arata) hand drop of *piece_kanji* on *dst*.

    Block B, in the selector-widened layout of KOM-028
    (``action_encoding``'s docstring): the block is plane-major over the
    destination square with

        plane = piece_idx * 2 + bsel

    -- the one-bit arata conversion selector ``bsel`` is the MINOR component,
    so consecutive planes are the two variants of one piece type.  ``bsel = 0``
    is the plain drop ("convert nothing"); ``bsel = 1`` converts the tier-1
    enemy under the drop square.  Every drop in this module's scripted lines
    lands on an empty square, so they all take the default ``bsel = 0``.

    Block C would be the draft-finishing variant; both fixture positions have
    their draft bits clear, so ``cymovegen`` only ever emits B.
    """
    if bsel not in (0, 1):
        raise ValueError(f"arata selector out of range: {bsel!r}")
    plane = action_encoding.PIECE_INDEX[piece_kanji] * 2 + bsel
    return action_encoding.B_START + plane * 81 + sq_index(dst)


# ---------------------------------------------------------------------------
# Scripted sequences (rules-audit "Aliasing defect")
# ---------------------------------------------------------------------------

#: Scenario A, plies 1-5: White's 3-move 帥 triangle against Black's 2-move 槍
#: shuffle.  ``w = 3, b = 2``, so the cells return to P with the side to move
#: FLIPPED -- the whole point of the scenario.
SCENARIO_A_CYCLE: tuple[int, ...] = (
    move_action((9, 5), (8, 5)),   # ply 1  W  帥(9-5)(8-5)
    move_action((1, 1), (2, 1)),   # ply 2  B  槍(1-1)(2-1)
    move_action((8, 5), (8, 4)),   # ply 3  W  帥(8-5)(8-4)
    move_action((2, 1), (1, 1)),   # ply 4  B  槍(2-1)(1-1)
    move_action((8, 4), (9, 5)),   # ply 5  W  帥(8-4)(9-5)
)

#: Scenario A, plies 6-10: the colour-mirrored cycle -- black 帥 triangle
#: ``(1-5)(2-5)(2-4)(1-5)`` against the white 侍 shuffle ``(5-1)(4-1)(5-1)``.
#: Flips the side to move back, so the cells return to P with White to move.
SCENARIO_A_MIRROR_CYCLE: tuple[int, ...] = (
    move_action((1, 5), (2, 5)),   # ply 6   B  帥(1-5)(2-5)
    move_action((5, 1), (4, 1)),   # ply 7   W  侍(5-1)(4-1)
    move_action((2, 5), (2, 4)),   # ply 8   B  帥(2-5)(2-4)
    move_action((4, 1), (5, 1)),   # ply 9   W  侍(4-1)(5-1)
    move_action((2, 4), (1, 5)),   # ply 10  B  帥(2-4)(1-5)
)

#: The full 15-ply Scenario A line: cells equal P at plies 0, 5, 10 and 15,
#: alternating side to move, so the v1 cells-only key counts four occurrences
#: while no true game state has occurred more than twice.
SCENARIO_A_15PLY: tuple[int, ...] = (
    SCENARIO_A_CYCLE + SCENARIO_A_MIRROR_CYCLE + SCENARIO_A_CYCLE
)

#: Scenario B, plies 1-8: a drop that is captured (irreversible progress for
#: both sides) wrapped in matched 4-ply tours, ``w = b = 4``.  Cells return to
#: Q with White still to move -- but White's hand is one 兵 lighter and Black
#: is a soldier up.
SCENARIO_B_LOOP: tuple[int, ...] = (
    drop_action(SOLDIER, (5, 9)),  # ply 1  W  新兵(5-9)      hand 兵 n -> n-1
    move_action((3, 9), (5, 9)),   # ply 2  B  馬(3-9)取(5-9)
    move_action((9, 5), (8, 5)),   # ply 3  W  帥(9-5)(8-5)
    move_action((1, 1), (2, 1)),   # ply 4  B  槍(1-1)(2-1)
    move_action((8, 5), (8, 4)),   # ply 5  W  帥(8-5)(8-4)
    move_action((5, 9), (3, 9)),   # ply 6  B  馬(5-9)(3-9)
    move_action((8, 4), (9, 5)),   # ply 7  W  帥(8-4)(9-5)
    move_action((2, 1), (1, 1)),   # ply 8  B  槍(2-1)(1-1)
)

#: The genuine 4-ply pure shuffle: white 帥 out-and-back, black 槍
#: out-and-back, ``w = b = 2``.  No drop, no capture, no hand change -- so the
#: repaired key and the v1 key agree that this really is the same position
#: coming round again.  Legal from both P's and Q's cell arrangement (the two
#: differ only on 3-9 and 5-9, which nothing here touches).
PURE_SHUFFLE: tuple[int, ...] = (
    move_action((9, 5), (8, 5)),   # W  帥(9-5)(8-5)
    move_action((1, 1), (2, 1)),   # B  槍(1-1)(2-1)
    move_action((8, 5), (9, 5)),   # W  帥(8-5)(9-5)
    move_action((2, 1), (1, 1)),   # B  槍(2-1)(1-1)
)

#: Piece kanji the sequences above move, re-exported so tests can name them.
FIXTURE_PIECES: tuple[str, ...] = (MARSHAL, WARRIOR, LANCER, RIDER, SOLDIER)

# ---------------------------------------------------------------------------
# Loading and playing
# ---------------------------------------------------------------------------


def board_of(game: Game) -> Board:
    """Return *game*'s live ``Board``.

    ``Game`` exposes no public accessor; the engine's own tests reach through
    ``_board`` the same way.
    """
    board: Board = game._board
    return board


def load_game(fen: str, rule: RepetitionRule = RepetitionRule.FULL) -> Game:
    """Return a fresh ``Game`` on *fen*, history seeded with its start key.

    *rule* is KOM-141's per-game repetition key and defaults to
    ``RepetitionRule.FULL`` -- Komugi's repaired 274-byte key, which is what
    every pre-KOM-141 caller of this helper got and still gets. Pass
    ``RepetitionRule.PLACEMENT`` to referee the fixture under gungi.js's
    cells-only key instead; that is the ONLY way to reach it here, and it is
    deliberately not a module-level switch.
    """
    return Game(board_from_fen(fen), repetition_rule=rule)


def load_position_p(rule: RepetitionRule = RepetitionRule.FULL) -> Game:
    """Scenario A's position P, White to move."""
    return load_game(POSITION_P_FEN, rule)


def load_position_q(rule: RepetitionRule = RepetitionRule.FULL) -> Game:
    """Scenario B's position Q, White to move, White holding 兵 x2."""
    return load_game(POSITION_Q_FEN, rule)


def load_mate_position() -> Game:
    """The checkmate fixture, Black to move and mated."""
    return load_game(MATE_AND_REPETITION_FEN)


def cells_bytes(game: Game) -> bytes:
    """The 243-byte ``cells_arr`` snapshot -- the whole of the v1 key."""
    cells: npt.NDArray[np.int8] = np.asarray(
        board_of(game).cells_arr, dtype=np.int8
    )
    return cells.tobytes()


def hand_count(game: Game, piece_index: int, color: int) -> int:
    """``hand[piece_index, color]`` as a plain int."""
    return int(board_of(game).hand_arr[piece_index, color])


def valid_mask(game: Game) -> npt.NDArray[np.uint8]:
    """``game.valid_moves()`` as a typed uint8 vector."""
    return np.asarray(game.valid_moves(), dtype=np.uint8)


def win_state(game: Game) -> npt.NDArray[np.uint8]:
    """``game.win_state()`` as a typed 3-vector ``[white, black, draw]``."""
    return np.asarray(game.win_state(), dtype=np.uint8)


def observation_planes(game: Game) -> npt.NDArray[np.float32]:
    """``game.observation()`` as a typed ``(planes, 9, 9)`` float array."""
    return np.asarray(game.observation(), dtype=np.float32)


def is_terminal(game: Game) -> bool:
    """True iff any slot of ``win_state()`` is set."""
    return bool(win_state(game).sum() > 0)


def play_action_checked(game: Game, action: int, ply: int) -> None:
    """Play *action*, asserting it is legal first.

    *ply* is the 1-based index of this action within its sequence and appears
    in the failure message, so a sequence that goes illegal names the ply that
    broke rather than just the id.
    """
    mask = valid_mask(game)
    board = board_of(game)
    assert mask[action] == 1, (
        f"ply {ply}: action {action} is not legal "
        f"(side to move: {'wb'[board.turn]}, "
        f"legal action ids: {int(mask.sum())})"
    )
    game.play_action(action)


def play_sequence(game: Game, actions: Sequence[int]) -> None:
    """Play every action of *actions* in order, asserting legality of each."""
    for ply, action in enumerate(actions, start=1):
        play_action_checked(game, action, ply)


# ---------------------------------------------------------------------------
# Move records and the golden playouts (KOM-002)
# ---------------------------------------------------------------------------

#: Move-type codes, the tests' own copy of ``cymovegen``'s ``MT_*`` constants.
#: They live in a ``.pyx`` and have no python-level name to import, so
#: ``action_encoding`` keeps a copy too; restating them here rather than
#: reaching into ``game._MT_*`` keeps the tests an independent witness.
MT_ROUTE: int = 0
MT_TSUKE: int = 1
MT_CAPTURE: int = 2
MT_BETRAY: int = 3
MT_ARATA: int = 4

#: Move types that RESET the progress clock (KOM-002 §3.3).
PROGRESS_MOVE_TYPES: frozenset[int] = frozenset({MT_CAPTURE, MT_BETRAY, MT_ARATA})

#: ``move_type`` recorded for ply 0, which was reached by no move at all.
NO_MOVE: int = -1

#: The oracle playout corpus.  It lives under ``tests/engine/`` because the
#: ported v1 suite owns it; this module resolves it by path rather than
#: importing that package's conftest, which pytest does not put on the path of
#: a sibling directory.
GOLDENS_DIR: Path = Path(__file__).resolve().parent.parent / "engine" / "goldens"

#: The four CONTIGUOUS playouts: within one ``games`` entry, ply *i*'s
#: ``fen_after`` is ply *i+1*'s ``fen_before``, so a test may load ply 0's
#: ``fen_before`` once and chain the recorded ``chosen`` moves.
#:
#: ``playouts_betray.json`` (KOM-101 G2) is deliberately NOT here: it is a
#: FILTERED corpus holding only the plies where a ``返`` move was legal, so its
#: consecutive plies are unrelated positions and chaining them walks the board
#: off the corpus after the first ply (``scripts/oracle/README.md``, "a
#: ``games`` entry in ``playouts_betray.json`` is not a contiguous playout";
#: ``tests/engine/test_move_gen.py``'s ``_FULL_GAME_FILES`` comment).  A test
#: that wants it must load EVERY ply from that ply's own ``fen_before``.
GOLDEN_PLAYOUT_FILES: tuple[str, ...] = (
    "playouts_intro.json",
    "playouts_beginner.json",
    "playouts_intermediate.json",
    "playouts_advanced.json",
)


def load_golden_playouts(name: str) -> list[list[dict[str, Any]]]:
    """Return the games of golden playout file *name*, one list of plies each.

    Skips -- rather than fails -- when the corpus is not vendored, matching
    ``tests/engine/conftest.load_golden``: the corpus is a build input, and a
    checkout without it should not report a plane-map regression.

    A ply is always self-contained -- ``fen_before`` / ``legal`` / ``chosen`` /
    ``fen_after`` -- but a *game* is a contiguous playout only for the files in
    :data:`GOLDEN_PLAYOUT_FILES`; see that constant's note before chaining.
    """
    path = GOLDENS_DIR / name
    if not path.is_file():
        pytest.skip(f"oracle golden {name} not available (corpus not vendored)")
    with path.open(encoding="utf-8") as handle:
        data: dict[str, Any] = json.load(handle)
    games: list[list[dict[str, Any]]] = data["games"]
    return games


def golden_fens(files: Sequence[str] = GOLDEN_PLAYOUT_FILES) -> list[str]:
    """Every ply's ``fen_before`` across *files*, in file and game order."""
    return [
        ply["fen_before"]
        for name in files
        for game in load_golden_playouts(name)
        for ply in game
    ]


# ---------------------------------------------------------------------------
# Replaying the goldens (KOM-002 §7 #3, #17)
# ---------------------------------------------------------------------------

#: The pure-Python reference engine, reached through an ``Any`` handle: its
#: ``moves()`` carries no return annotation, so a direct call from a strict
#: module is a call into an untyped function.  Same device as ``_CYENGINE``.
_API: Any = api


@contextmanager
def start_mode(mode: str) -> Iterator[None]:
    """Run the block with ``Game``'s CLASS-level start mode set to *mode*.

    ``Game.set_start_mode`` mutates a class attribute (``game.py:326-331``),
    so a replay that sets it and walks away changes what a no-argument
    ``Game()`` means for every test that runs afterwards.  Restoring is not
    optional; that is why this exists rather than a bare call.
    """
    previous: str = Game._START_MODE
    Game.set_start_mode(mode)
    try:
        yield
    finally:
        Game.set_start_mode(previous)


def golden_ply_action(fen_before: str, san: str) -> int:
    """The action id of the golden's ``chosen`` SAN at *fen_before*.

    Resolved the way ``tests/engine/test_action_encoding.py``'s round-trip
    resolves it -- generate the legal moves of ``api.Gungi(fen_before)``, match
    the SAN (with the goldens' ``=``/``#`` decorations tolerated), and encode
    the record -- so a golden that replays here replays there.
    """
    handle: Any = _API.Gungi(fen_before)
    moves: list[dict[str, Any]] = handle.moves({"verbose": True})
    normalized = san.rstrip("=#")
    for move in moves:
        if move["san"] == normalized or move["san"] == san:
            return int(action_encoding.encode_move(move))
    raise AssertionError(
        f"golden SAN {san!r} is not in the legal moves of {fen_before!r}"
    )


@lru_cache(maxsize=None)
def golden_action_sequences(mode: str) -> tuple[tuple[int, ...], ...]:
    """Every golden game of *mode* as a tuple of action ids, game by game.

    Cached because resolving a SAN goes through the pure-Python move generator
    and is the expensive half of a replay: #3 and #17 replay the SAME games
    from the same corpus, and paying for the resolution twice buys nothing.
    Replaying the ids through ``Game`` afterwards is the cheap half, and each
    test does its own so they cannot contaminate one another's state.
    """
    return tuple(
        tuple(golden_ply_action(ply["fen_before"], ply["chosen"]) for ply in game)
        for game in load_golden_playouts(f"playouts_{mode}.json")
    )


# ---------------------------------------------------------------------------
# The frozen pmv1 capture (KOM-002 §7 #3, #17)
# ---------------------------------------------------------------------------

#: Where the pre-change capture lives.  It sits under ``tests/encoding/``
#: because that tree owns the encoding goldens; this module resolves it by
#: path for the same reason it resolves ``tests/engine/goldens`` by path.
PMV1_DIR: Path = (
    Path(__file__).resolve().parent.parent / "encoding" / "goldens" / "pmv1"
)

#: The four modes captured, one ``.npz`` each.
PMV1_MODES: tuple[str, ...] = ("intro", "beginner", "intermediate", "advanced")

#: Width of the pmv1 map -- the range invariant P1 pins (design §2).
PMV1_PLANE_COUNT: int = 118

#: Positions in the golden corpus, per ``tests/engine/goldens/MANIFEST.json``'s
#: per-mode ply counts.  Restated here so a corpus that silently shrank cannot
#: make a replay-over-everything test pass by comparing almost nothing.
GOLDEN_CORPUS_PLIES: int = 707 + 791 + 511 + 669


@dataclass(frozen=True)
class Pmv1Capture:
    """One mode's frozen pmv1 capture, as ``gen_pmv1_fixture.py`` wrote it.

    The capture is per-position sha256 digests of the 118-plane observation
    taken BEFORE each ply, plus the ``win_state`` AFTER each ply, plus the raw
    planes of a sampled subset (ply 0 and each game's final ply).  Digests
    rather than raw arrays because the raw corpus is ~102 MB of float32; the
    samples are what let a mismatch name a PLANE rather than only a ply
    (design rev 4).
    """

    mode: str
    #: sha256 hex of the 118 planes before each ply, flat, in golden order.
    digests: tuple[str, ...]
    #: ``win_state()`` after each ply, flat, in golden order.
    win_state: npt.NDArray[np.uint8]
    #: Plies per game, in golden order -- the flat arrays' segmentation.
    plies_per_game: tuple[int, ...]
    #: Raw ``(118, 9, 9)`` planes for the sampled positions.
    sample_planes: npt.NDArray[np.float32]
    #: ``(game, ply)`` of each sample, in ``sample_planes`` order.
    sample_keys: tuple[tuple[int, int], ...]

    def flat_index(self, game: int, ply: int) -> int:
        """The flat index of *ply* of *game* in the per-position arrays."""
        return sum(self.plies_per_game[:game]) + ply

    def game_win_state(self, game: int) -> npt.NDArray[np.uint8]:
        """The frozen ``win_state`` sequence of *game*, ``(plies, 3)``."""
        start = self.flat_index(game, 0)
        return self.win_state[start : start + self.plies_per_game[game]]

    def sample_for(self, game: int, ply: int) -> npt.NDArray[np.float32] | None:
        """The raw sampled planes at ``(game, ply)``, or ``None`` if unsampled."""
        for index, key in enumerate(self.sample_keys):
            if key == (game, ply):
                return np.asarray(self.sample_planes[index], dtype=np.float32)
        return None


def load_pmv1_capture(mode: str) -> Pmv1Capture:
    """Load and self-check the frozen pmv1 capture for *mode*.

    Skips -- with the regeneration recipe in the reason -- only when the
    fixture DIRECTORY is absent, matching how ``load_golden_playouts`` treats
    an unvendored corpus.  A present directory missing one of its four files
    is a corrupt fixture, not an absent one, and fails.
    """
    if not PMV1_DIR.is_dir():
        pytest.skip(
            f"pmv1 capture directory {PMV1_DIR} is absent; regenerate it with "
            "scripts/goldens/gen_pmv1_fixture.py inside a worktree at "
            "main@29213f2 (see tests/encoding/goldens/GENERATION.md)"
        )
    path = PMV1_DIR / f"pmv1_{mode}.npz"
    assert path.is_file(), (
        f"{PMV1_DIR} exists but {path.name} is missing: the capture is "
        "incomplete, which is a corrupt fixture rather than an absent one"
    )

    with np.load(path) as data:
        digests = tuple(str(value) for value in np.asarray(data["digests"]).tolist())
        wins = np.asarray(data["win_state"], dtype=np.uint8)
        plies_per_game = tuple(
            int(value) for value in np.asarray(data["plies_per_game"]).tolist()
        )
        sample_planes = np.asarray(data["sample_planes"], dtype=np.float32)
        sample_keys = tuple(
            (int(game), int(ply))
            for game, ply in np.asarray(data["sample_keys"]).tolist()
        )

    total = sum(plies_per_game)
    assert len(digests) == total, (
        f"pmv1_{mode}.npz: {len(digests)} digests for {total} plies"
    )
    assert wins.shape == (total, 3), f"pmv1_{mode}.npz: win_state {wins.shape}"
    assert sample_planes.shape == (
        len(sample_keys),
        PMV1_PLANE_COUNT,
        9,
        9,
    ), f"pmv1_{mode}.npz: sample_planes {sample_planes.shape}"

    return Pmv1Capture(
        mode=mode,
        digests=digests,
        win_state=wins,
        plies_per_game=plies_per_game,
        sample_planes=sample_planes,
        sample_keys=sample_keys,
    )


def move_by_san(board: Board, san: str) -> dict[str, Any]:
    """The generated move record whose SAN is *san*.

    The goldens decorate a SAN with ``=`` (draw) and ``#`` (mate) suffixes the
    generator does not emit, so both spellings are accepted -- the same
    normalisation ``tests/engine/test_cyengine.py`` uses.
    """
    normalized = san.rstrip("=#")
    for move in generate_all_moves(board):
        if move["san"] == normalized or move["san"] == san:
            return move
    raise AssertionError(f"no generated move has SAN {san!r}")


def move_matching(
    board: Board, move_type: int, destination: Square
) -> dict[str, Any]:
    """The one generated move of *move_type* landing on *destination*.

    Selected by the record's own fields rather than by a hand-built action id:
    a square can carry several move types at once (a tsuke, a betray and a
    capture onto the same tower all exist in ``cymovegen``'s output), and the
    A0/A1 action id does not separate them.

    Asserts there is exactly one, so a fixture position that stops producing
    the move -- or starts producing two -- fails here rather than silently
    testing a different move.
    """
    to_rank, to_file = destination
    found = [
        move
        for move in generate_all_moves(board)
        if int(move["move_type"]) == move_type
        and int(move["to_r"]) == to_rank - 1
        and int(move["to_f"]) == to_file - 1
    ]
    assert len(found) == 1, (
        f"expected exactly one move_type={move_type} move to {destination}, "
        f"got {len(found)}: {[move['san'] for move in found]}"
    )
    return found[0]


def action_of(move: dict[str, Any]) -> int:
    """The flat action id of a cymovegen move record."""
    return int(action_encoding.encode_cython_move(move))


def decoded_action(board: Board, action: int) -> dict[str, Any]:
    """The move record *action* decodes to against *board*.

    The same call ``Game.play_action`` makes, so a test can read the
    ``to_tier`` the last-move planes will be built from BEFORE the move is
    applied -- after it, the board no longer holds the origin tower.
    """
    record: dict[str, Any] = action_encoding.decode_cython_action(action, board)
    return record


# ---------------------------------------------------------------------------
# Seeded playouts (KOM-002 §7 D)
# ---------------------------------------------------------------------------


def potential(game: Game) -> int:
    """``Φ(b) = 2 * hand_arr.sum() + count_nonzero(cells_arr)`` (design §3.3).

    The potential-function half of the progress-move definition: Φ is
    non-increasing over every legal move, and strictly decreases exactly on a
    capture, a betrayal or an arata.  Integer arithmetic throughout, so the
    comparison is exact.
    """
    board = board_of(game)
    hand: npt.NDArray[np.int64] = np.asarray(board.hand_arr, dtype=np.int64)
    cells: npt.NDArray[np.int8] = np.asarray(board.cells_arr, dtype=np.int8)
    return 2 * int(hand.sum()) + int(np.count_nonzero(cells))


@dataclass(frozen=True)
class PlySnapshot:
    """One ply of a playout, with everything KOM-002's acceptances read."""

    ply: int
    #: The move type played to REACH this ply, or :data:`NO_MOVE` at ply 0.
    move_type: int
    observation: npt.NDArray[np.float32]
    progress_plies: int
    repetition_count: int
    potential: int
    last_move: tuple[int, int, int, int] | None


def _snapshot(game: Game, ply: int, move_type: int) -> PlySnapshot:
    """Capture *game*'s current ply as a :class:`PlySnapshot`."""
    return PlySnapshot(
        ply=ply,
        move_type=move_type,
        observation=observation_planes(game),
        progress_plies=game._progress_plies,
        repetition_count=repetition_count(game),
        potential=potential(game),
        last_move=game._last_move,
    )


def seeded_playout(
    plies: int, seed: int, fen: str = INTRO_POSITION
) -> list[PlySnapshot]:
    """Snapshots of plies ``0..plies`` of a uniformly-random legal game.

    Retries adjacent seeds until one game runs the full length.  A random game
    can end early -- a marshal capture is always on the table -- and a test
    that says "over a 120-ply game" while silently measuring a 12-ply one is
    the failure mode this loop exists to prevent; it raises rather than
    returning short.
    """
    for offset in range(16):
        game = load_game(fen)
        snapshots = [_snapshot(game, 0, NO_MOVE)]
        rng = np.random.default_rng(seed + offset)
        while len(snapshots) <= plies:
            if is_terminal(game):
                break
            legal = np.nonzero(valid_mask(game))[0]
            if legal.size == 0:
                break
            action = int(rng.choice(legal))
            move_type = int(decoded_action(board_of(game), action)["move_type"])
            game.play_action(action)
            snapshots.append(_snapshot(game, len(snapshots), move_type))
        if len(snapshots) > plies:
            return snapshots
    raise AssertionError(
        f"no seed in [{seed}, {seed + 16}) produced a {plies}-ply game from "
        f"{fen!r}; the driver, not the encoder, is what to look at"
    )


# ---------------------------------------------------------------------------
# The repaired API (KOM-027) -- shims, see the module docstring
# ---------------------------------------------------------------------------

#: Byte length of the repaired key: 243 cells + 3 scalars + 28 hand slots.
REPETITION_KEY_LEN: int = 274
#: Offsets within the repaired key, per rules-audit "Proposed key".  Left
#: un-annotated on purpose: ``slice`` is generic in recent typeshed, so a bare
#: ``: slice`` annotation trips mypy's ``disallow_any_generics``.
CELLS_SLICE = slice(0, 243)
SCALARS_SLICE = slice(243, 246)
HAND_SLICE = slice(246, 274)

#: Threshold the repaired rule adjudicates on -- ``>=``, never ``==``.
FOURFOLD: int = 4

#: Plane layout after the repair (rules-audit "Encoding-plane consequences"):
#: 115 is the ply-denominated turn clock, the old scalar plane 116 is gone, and
#: R2/R3 are the two monotone repetition thresholds appended in that order.
PLANE_TURN_CLOCK: int = 115
PLANE_R2: int = 116
PLANE_R3: int = 117

#: Plane layout after input planes v2 (KOM-002 §2), appended above the repair's
#: block: two last-move planes, the progress clock, then the reserved padding to
#: ``N = 128``.  Restated here as LITERALS on purpose -- these are the tests'
#: independent copy of the pre-registered map, so an accidental renumbering in
#: the encoder is caught rather than followed.
PLANE_LAST_MOVE_FROM: int = 118
PLANE_LAST_MOVE_TO: int = 119
PLANE_PROGRESS: int = 120
RESERVED_PLANES: range = range(121, 128)

#: Total planes the encoder declares.  Named ``REPAIRED_`` because KOM-027 was
#: what first moved it off v1's 117; KOM-002 moved it again, to 128.
REPAIRED_PLANE_COUNT: int = 128

_CYENGINE: Any = cyengine


def has_repair() -> bool:
    """True once KOM-027 has landed ``cyengine.repetition_key``."""
    return hasattr(cyengine, "repetition_key")


def repetition_key(board: Board) -> bytes:
    """The repaired 274-byte key for *board* (KOM-027)."""
    fn: Callable[[Board], bytes] = _CYENGINE.repetition_key
    return fn(board)


def repetition_count(game: Game) -> int:
    """Occurrences of *game*'s CURRENT key in its history, current included."""
    handle: Any = game
    fn: Callable[[], int] = handle.repetition_count
    return fn()


def plane_count() -> int:
    """The plane count ``komugi.engine.observation`` currently declares."""
    return int(observation._TOTAL_PLANES)


def game_with_repeated_history(fen: str, occurrences: int) -> Game:
    """A ``Game`` on *fen* whose history already holds its key *occurrences* times.

    This is the bridge-shaped construction the audit calls out under F2: a
    ``Game`` handed a history it did not play itself (``gungi.py:200-208``).
    It is the only way to reach a count the engine cannot reach by play --
    5 occurrences (RT-02), or a checkmate that has "already happened" three
    times (RT-19) -- because per-ply adjudication ends the game first.

    The history convention is v1's and the repair keeps it: the history
    CONTAINS the current position's key (seeded at construction, appended after
    every ply), so ``repetition_count()`` is the multiplicity of that key in
    the list, not one more than it.  Callers assert that as a precondition.
    """
    board = board_from_fen(fen)
    key = repetition_key(board)
    return Game(board, [key] * occurrences)
