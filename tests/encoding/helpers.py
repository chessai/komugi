"""Shared machinery for the betrayal-encoding suite (KOM-128).

Everything here is read off ``docs/design/betrayal-encoding.md`` -- the
authoritative design for the betrayal-choice action-space extension (KOM-028)
-- and off the frozen reference dumps in ``goldens/betrayal_dumps.json``
(generated from the verbatim gungi-js engine; see ``goldens/GENERATION.md``).

The suite is authored RED-FIRST, before KOM-028 exists.  Three consequences
show up in this file:

* The extended surface (``ACTION_SIZE == 37341``, ``sel_from_captured``,
  ``komugi.engine.compat``) is reached through ``Any``-typed handles, because
  today's ``komugi.engine.action_encoding`` declares none of it and this
  ticket may not touch ``src/**``.
* ``has_extension()`` is the single predicate every gated module skips on.
  Modules that only read the dumps or exercise the CURRENT engine run
  unconditionally, so the suite has teeth before KOM-028 lands.
* The v1 and v2 id arithmetic are BOTH re-implemented here as pure functions
  over the design's tables, rather than imported.  A test that imported the
  encoder it is meant to pin would assert nothing.

FINDING (recorded here because it shapes every dump reader below): the
generation harness dumps ``m.piece?.square`` / ``m.piece?.tier`` / ``m.type``
into the ``from`` / ``fromTier`` / ``pieceType`` / ``color`` fields, and the
reference ``Move`` object carries no ``piece`` member -- so those four fields
are ``null`` / ``"?"`` for EVERY move in the corpus.  The origin square and
the moved piece type are still fully recoverable from the ``san`` field, whose
grammar is fixed by ``cymovegen._build_san`` / the JS ``createMove``:

    board move   {kanji}({r}-{f}-{t})[取]({r}-{f}-{t})[付 | 返{kanji}...]
    hand drop    新{kanji}({r}-{f}-{t})[付 | 返{kanji}...][終]

so ``from`` is the FIRST parenthesised triple of a non-arata SAN and the moved
piece is the leading kanji (after 新 for a drop).  ``parse_san_origin`` and
``san_piece_kanji`` below do exactly that.  The goldens are read-only, so the
recovery lives here rather than in a regenerated dump.

Coordinates are 1-based ``(rank, file)`` throughout, matching the dumps'
``rank-file-tier`` strings.
"""

from __future__ import annotations

import importlib
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Final, Iterable, Literal, NamedTuple, TypedDict, cast

import numpy as np
import numpy.typing as npt
import pytest

from komugi.engine import action_encoding, cyengine, cymovegen
from komugi.engine.cyboard import Board, board_from_fen

# ---------------------------------------------------------------------------
# The pinned v2 layout (docs/design/betrayal-encoding.md, "New layout")
# ---------------------------------------------------------------------------

#: 461 planes x 81 destination squares.
V2_PLANES: Final = 461
V2_ACTION_SIZE: Final = 37341

V2_A0_START: Final = 0      # board, no-stack (route / capture)
V2_A1_START: Final = 6561   # board, stack (tsuke / betray), selector-widened
V2_B_START: Final = 32805   # arata, betray-bit widened
V2_C_START: Final = 35073   # arata + 終, betray-bit widened

#: Plane counts per block, in block order -- the running sums of these are the
#: block starts above (design "Arithmetic check").
V2_BLOCK_PLANES: Final = (81, 324, 28, 28)

# ---------------------------------------------------------------------------
# The v1 layout it replaces (design "Current encoding") -- also frozen, so it
# is spelled out rather than imported: after KOM-028 lands, ``action_encoding``
# no longer holds these numbers, but π still has to project onto them.
# ---------------------------------------------------------------------------

V1_ACTION_SIZE: Final = 15390
V1_A0_START: Final = 0
V1_A1_START: Final = 6561
V1_B_START: Final = 13122
V1_C_START: Final = 14256

# ---------------------------------------------------------------------------
# Move-type vocabulary
# ---------------------------------------------------------------------------

Kind = Literal["route", "tsuke", "capture", "betray", "arata"]

#: cymovegen MT_* codes -> the JS ``move.type`` strings the dumps use.
KIND_BY_MT: Final[dict[int, Kind]] = {
    0: "route",
    1: "tsuke",
    2: "capture",
    3: "betray",
    4: "arata",
}

#: The kinds that carry a conversion selector.  ``capture`` records also carry
#: a non-empty ``captured`` list -- the whole captured tower, tiers 1..3 -- but
#: that is a capture, not a conversion, and block A0 has no selector.
CONVERTING_KINDS: Final[frozenset[str]] = frozenset({"betray", "arata"})

TACTICIAN_KANJI: Final = "謀"  # 謀

# ---------------------------------------------------------------------------
# Reference dumps
# ---------------------------------------------------------------------------

GOLDENS_DIR: Final = Path(__file__).resolve().parent / "goldens"
DUMPS_PATH: Final = GOLDENS_DIR / "betrayal_dumps.json"

DumpedMove = TypedDict(
    "DumpedMove",
    {
        "kind": str,
        "from": str | None,
        "fromTier": int | None,
        "pieceType": str,
        "color": str,
        "to": str,
        "capturedTiers": list[int],
        "capturedTypes": list[str],
        "draftFinished": bool,
        "san": str | None,
    },
)


class Dump(TypedDict):
    fen: str
    moves: list[DumpedMove]


class DumpFile(TypedDict):
    generator: str
    generated: str
    dumps: dict[str, Dump]
    perfts: dict[str, dict[str, dict[str, int]]]


@lru_cache(maxsize=1)
def load_dumps() -> DumpFile:
    """Parse ``goldens/betrayal_dumps.json`` (cached)."""
    with DUMPS_PATH.open(encoding="utf-8") as handle:
        return cast(DumpFile, json.load(handle))


def corpus_names() -> tuple[str, ...]:
    """Every fixture name in the frozen corpus, in generation order."""
    return tuple(load_dumps()["dumps"])


def corpus_fen(name: str) -> str:
    """The frozen FEN of corpus fixture *name*."""
    return load_dumps()["dumps"][name]["fen"]


def dumped_moves(name: str) -> list[DumpedMove]:
    """Every reference move dumped for fixture *name*."""
    return load_dumps()["dumps"][name]["moves"]


def perft_table(name: str) -> dict[str, dict[str, int]]:
    """The reference depth -> {move type: count} table for fixture *name*."""
    return load_dumps()["perfts"][name]


def perft_fixture_names() -> tuple[str, ...]:
    """Fixtures for which the reference recorded a perft table (T9)."""
    return tuple(load_dumps()["perfts"])


def kind_counts(name: str) -> dict[str, int]:
    """``{move type: count}`` over fixture *name*'s dumped moves."""
    counts: dict[str, int] = {}
    for move in dumped_moves(name):
        counts[move["kind"]] = counts.get(move["kind"], 0) + 1
    return counts


def is_draft_fen(fen: str) -> bool:
    """True iff *fen*'s drafting field names a side still drafting.

    Field 5 of a Gungi FEN (``board hand turn mode drafting moveNumber``) is
    ``-`` when neither side is drafting.
    """
    return fen.split(" ")[4] != "-"


# ---------------------------------------------------------------------------
# Squares
# ---------------------------------------------------------------------------


def sq_index(rank: int, file: int) -> int:
    """Map a 1-based ``(rank, file)`` to the 0..80 index the encoding uses.

    Mirrors ``action_encoding._RF_TO_SQ`` (``r * 9 + (8 - f)`` on 0-based
    coordinates) and ``cymovegen._sq_index``: squares run rank 1->9, and within
    a rank file 9->1.
    """
    if not (1 <= rank <= 9 and 1 <= file <= 9):
        raise ValueError(f"square out of range: ({rank}, {file})")
    return (rank - 1) * 9 + (9 - file)


def rank_file(sq: int) -> tuple[int, int]:
    """Inverse of :func:`sq_index` -- 1-based ``(rank, file)``."""
    if not 0 <= sq < 81:
        raise ValueError(f"square index out of range: {sq}")
    return sq // 9 + 1, 9 - sq % 9


def mirror_sq(sq: int) -> int:
    """File reflection ``f -> 10 - f`` on a square index (``game._mirror_sq``)."""
    rank, file = rank_file(sq)
    return sq_index(rank, 10 - file)


def parse_square(text: str) -> tuple[int, int, int]:
    """Split a dump square string into 1-based ``(rank, file, tier)``.

    Accepts ``'rank-file'`` (tier defaults to 1) and ``'rank-file-tier'``.
    """
    parts = text.split("-")
    if len(parts) == 2:
        return int(parts[0]), int(parts[1]), 1
    if len(parts) == 3:
        return int(parts[0]), int(parts[1]), int(parts[2])
    raise ValueError(f"not a square string: {text!r}")


# ---------------------------------------------------------------------------
# SAN recovery (see the module docstring's FINDING)
# ---------------------------------------------------------------------------

_SAN_SQUARE_RE: Final = re.compile(r"\((\d)-(\d)-(\d)\)")

_ARATA_MARKER: Final = "新"  # 新


def san_piece_kanji(san: str) -> str:
    """The kanji of the piece that moves, from a reference SAN string."""
    if not san:
        raise ValueError("empty SAN")
    return san[1] if san[0] == _ARATA_MARKER else san[0]


def parse_san_origin(san: str) -> tuple[int, int, int]:
    """The 1-based ``(rank, file, tier)`` a board move departs from.

    The first parenthesised triple of a non-arata SAN is always the origin;
    the second is the destination.  Raises for an arata SAN, which has one
    triple and no origin.
    """
    if san.startswith(_ARATA_MARKER):
        raise ValueError(f"arata SAN has no origin square: {san!r}")
    found = _SAN_SQUARE_RE.findall(san)
    if len(found) != 2:
        raise ValueError(f"expected two squares in board SAN, got {san!r}")
    rank, file, tier = found[0]
    return int(rank), int(file), int(tier)


# ---------------------------------------------------------------------------
# Conversion selectors
# ---------------------------------------------------------------------------


def tier_mask(tiers: Iterable[int]) -> int:
    """``sel`` for a set of converted tiers -- ``design sel_from_captured``.

    ``sel = Σ 1 << (tier - 1)``.  Tiers above 2 are unreachable for a
    conversion (the betray branch requires tower height <= 2), so they raise
    rather than silently producing a 3-bit selector.
    """
    sel = 0
    for tier in tiers:
        if not 1 <= tier <= 2:
            raise ValueError(
                f"conversion at tier {tier} is unreachable "
                "(design: 'the enemies can only ever sit at tiers 1 and 2')"
            )
        bit = 1 << (tier - 1)
        if sel & bit:
            raise ValueError(f"tier {tier} converted twice")
        sel |= bit
    return sel


def submasks(mask: int) -> tuple[int, ...]:
    """Every submask of *mask*, including 0 and *mask* itself, ascending.

    ``{0} ∪ non-empty subsets of Q`` in one expression -- the design's
    ``(sel & ~Q) == 0`` predicate, enumerated.  A mask outside two bits raises:
    the selector is 2 bits wide, so a wider ``Q`` would silently lose subsets
    rather than produce a wrong answer that a caller could notice.
    """
    if not 0 <= mask <= 0b11:
        raise ValueError(f"Q outside the two-bit selector range: {mask:#b}")
    return tuple(s for s in range(4) if (s & ~mask) == 0)


# ---------------------------------------------------------------------------
# The canonical move normal form
# ---------------------------------------------------------------------------


class MoveKey(NamedTuple):
    """The shared normal form of a move, reference-side or engine-side.

    ``from_sq`` is ``-1`` for an arata (a hand drop has no origin square).
    ``sel`` is the conversion tier-mask and is 0 for every non-converting kind
    -- notably for ``capture``, whose ``captured`` list is the captured tower
    (tiers 1..3) rather than a conversion choice, and which lands in block A0
    where no selector exists.
    """

    kind: str
    from_sq: int
    to_sq: int
    to_tier: int
    piece_idx: int
    sel: int
    draft_finished: bool


def piece_index(kanji: str) -> int:
    """0..13 index of a piece kanji (``action_encoding.PIECE_INDEX``)."""
    return int(action_encoding.PIECE_INDEX[kanji])


def normalize_dump_move(move: DumpedMove) -> MoveKey:
    """Normal form of one reference-dump move record."""
    kind = move["kind"]
    san = move["san"]
    if san is None:
        raise ValueError(f"dump move without SAN: {move!r}")
    to_rank, to_file, to_tier = parse_square(move["to"])
    if kind == "arata":
        from_sq = -1
    else:
        from_rank, from_file, _ = parse_san_origin(san)
        from_sq = sq_index(from_rank, from_file)
    sel = tier_mask(move["capturedTiers"]) if kind in CONVERTING_KINDS else 0
    return MoveKey(
        kind=kind,
        from_sq=from_sq,
        to_sq=sq_index(to_rank, to_file),
        to_tier=to_tier,
        piece_idx=piece_index(san_piece_kanji(san)),
        sel=sel,
        draft_finished=bool(move["draftFinished"]),
    )


def normalize_engine_move(record: dict[str, Any]) -> MoveKey:
    """Normal form of one ``cymovegen`` move record."""
    kind = KIND_BY_MT[int(record["move_type"])]
    if kind == "arata":
        from_sq = -1
    else:
        from_sq = int(record["from_r"]) * 9 + (8 - int(record["from_f"]))
    captured = cast("list[tuple[int, int]]", record["captured"])
    sel = tier_mask(tier for _, tier in captured) if kind in CONVERTING_KINDS else 0
    return MoveKey(
        kind=kind,
        from_sq=from_sq,
        to_sq=int(record["to_r"]) * 9 + (8 - int(record["to_f"])),
        to_tier=int(record["to_tier"]),
        piece_idx=int(record["piece_type_index"]),
        sel=sel,
        draft_finished=bool(record["draft_finished"]),
    )


# ---------------------------------------------------------------------------
# Id arithmetic -- both layouts, spelled out from the design's tables
# ---------------------------------------------------------------------------


def v2_id(key: MoveKey) -> int:
    """The extended-layout action id of *key* (design "New layout")."""
    if key.kind == "arata":
        if not 0 <= key.sel <= 1:
            raise ValueError(
                f"arata selector must be one bit (tier 1 only), got {key.sel}"
            )
        start = V2_C_START if key.draft_finished else V2_B_START
        return start + (key.piece_idx * 2 + key.sel) * 81 + key.to_sq
    if key.kind in ("route", "capture"):
        return V2_A0_START + key.from_sq * 81 + key.to_sq
    return V2_A1_START + (key.from_sq * 4 + key.sel) * 81 + key.to_sq


def v1_id(key: MoveKey) -> int:
    """The v1 action id of *key* -- the collapse the extension removes.

    Selector-blind by construction: every betrayal at a square pair, and the
    tsuke beside it, share one A1 id, and an arata conversion shares the plain
    drop's B/C id (design "Current encoding").
    """
    if key.kind == "arata":
        start = V1_C_START if key.draft_finished else V1_B_START
        return start + key.piece_idx * 81 + key.to_sq
    if key.kind in ("route", "capture"):
        return V1_A0_START + key.from_sq * 81 + key.to_sq
    return V1_A1_START + key.from_sq * 81 + key.to_sq


def project_v1(action_id: int) -> int:
    """π -- the pure, total, board-free v2 -> v1 projection (design "How old
    goldens survive").  Forgets the selector."""
    if not 0 <= action_id < V2_ACTION_SIZE:
        raise ValueError(f"not a v2 action id: {action_id}")
    if action_id < V2_A1_START:
        return action_id
    if action_id < V2_B_START:
        plane, to_sq = divmod(action_id - V2_A1_START, 81)
        from_sq = plane // 4
        return V1_A1_START + from_sq * 81 + to_sq
    if action_id < V2_C_START:
        plane, to_sq = divmod(action_id - V2_B_START, 81)
        return V1_B_START + (plane // 2) * 81 + to_sq
    plane, to_sq = divmod(action_id - V2_C_START, 81)
    return V1_C_START + (plane // 2) * 81 + to_sq


def v2_block(action_id: int) -> str:
    """``'A0' | 'A1' | 'B' | 'C'`` -- which v2 block *action_id* lands in."""
    if not 0 <= action_id < V2_ACTION_SIZE:
        raise ValueError(f"not a v2 action id: {action_id}")
    if action_id < V2_A1_START:
        return "A0"
    if action_id < V2_B_START:
        return "A1"
    if action_id < V2_C_START:
        return "B"
    return "C"


def a1_parts(action_id: int) -> tuple[int, int, int]:
    """``(from_sq, sel, to_sq)`` of a block-A1 v2 id."""
    if v2_block(action_id) != "A1":
        raise ValueError(f"{action_id} is not a block-A1 id")
    plane, to_sq = divmod(action_id - V2_A1_START, 81)
    from_sq, sel = divmod(plane, 4)
    return from_sq, sel, to_sq


def arata_parts(action_id: int) -> tuple[int, int, int, bool]:
    """``(piece_idx, bsel, to_sq, draft_finished)`` of a block-B/C v2 id."""
    block = v2_block(action_id)
    if block == "B":
        plane, to_sq = divmod(action_id - V2_B_START, 81)
        finished = False
    elif block == "C":
        plane, to_sq = divmod(action_id - V2_C_START, 81)
        finished = True
    else:
        raise ValueError(f"{action_id} is not a block-B/C id")
    piece_idx, bsel = divmod(plane, 2)
    return piece_idx, bsel, to_sq, finished


# ---------------------------------------------------------------------------
# Engine access
# ---------------------------------------------------------------------------

#: Module handles for the surface KOM-028 adds.  ``action_encoding`` is under
#: mypy's ``ignore_errors`` override but its *callers* are not, so the new
#: names have to come through an ``Any`` until the module declares them.
_AE: Any = action_encoding


def has_extension() -> bool:
    """True once ``ACTION_SIZE`` has moved to the extended layout."""
    return int(_AE.ACTION_SIZE) == V2_ACTION_SIZE


def action_size() -> int:
    """The action size the installed encoding declares."""
    return int(_AE.ACTION_SIZE)


def sel_from_captured(captured: Iterable[tuple[int, int]]) -> int:
    """``action_encoding.sel_from_captured`` (KOM-028), as an int."""
    return int(_AE.sel_from_captured(list(captured)))


def compat_module() -> Any:
    """``komugi.engine.compat`` -- π and ρ (design "How old goldens survive").

    Test-only module, deleted once the policy head freezes.

    The soft import is conditioned on the extension NOT having landed.  Before
    KOM-028, ``komugi.engine.compat`` cannot exist and a gated module should
    skip rather than error.  After it -- once ``ACTION_SIZE`` has moved to the
    extended layout -- a missing compat module is a hole in T7, not a reason to
    skip: an unconditional ``importorskip`` would retire the whole
    golden-survival module silently and forever, exactly the state this suite
    exists to prevent.  So on the extended surface the import is hard and its
    failure is a test failure that names the contract.
    """
    if has_extension():
        try:
            return importlib.import_module("komugi.engine.compat")
        except ImportError as exc:
            pytest.fail(
                "KOM-028 landed without compat.py -- T7 golden survival "
                f"cannot run ({exc})"
            )
    return pytest.importorskip(
        "komugi.engine.compat",
        reason="KOM-028 compat module (π / ρ) not yet implemented",
    )


def _resolve(module: Any, names: tuple[str, ...], what: str) -> Any:
    """First attribute of *module* named in *names*.

    The design fixes the two functions' semantics and calls them π and ρ but
    does not spell their Python names, so a small alias set is accepted.  A
    module exposing none of them fails loudly and names the contract.
    """
    for name in names:
        attribute = getattr(module, name, None)
        if attribute is not None:
            return attribute
    raise AssertionError(
        f"komugi.engine.compat exposes no {what}; expected one of {names} "
        "(docs/design/betrayal-encoding.md, 'How old goldens survive')"
    )


def compat_pi(module: Any) -> Any:
    """The compat module's π (``v2_id -> v1_id``)."""
    return _resolve(module, ("project", "pi", "project_v1", "to_v1"), "π projection")


def compat_rho(module: Any) -> Any:
    """The compat module's ρ (``(position, v1_id) -> v2_id``)."""
    return _resolve(module, ("lift", "rho", "lift_v2", "to_v2"), "ρ lift")


def board_of_fen(fen: str) -> Board:
    """A fresh ``cyboard.Board`` on *fen*."""
    return board_from_fen(fen)


def engine_records(board: Board) -> list[dict[str, Any]]:
    """``cymovegen.generate_all_moves`` -- the record path."""
    return cymovegen.generate_all_moves(board)


def engine_keys(board: Board) -> list[MoveKey]:
    """Normal forms of every record the engine generates for *board*."""
    return [normalize_engine_move(record) for record in engine_records(board)]


def legal_mask(board: Board) -> npt.NDArray[np.uint8]:
    """``cymovegen.fill_legal_mask`` into a fresh ``ACTION_SIZE`` vector."""
    mask: npt.NDArray[np.uint8] = np.zeros(action_size(), dtype=np.uint8)
    cymovegen.fill_legal_mask(board, mask)
    return mask


def masked_ids(mask: npt.NDArray[np.uint8]) -> set[int]:
    """The set of set bits of *mask*."""
    nonzero: npt.NDArray[np.intp] = np.nonzero(mask)[0]
    return {int(index) for index in nonzero}


# ---------------------------------------------------------------------------
# Board inspection -- the test's OWN reading of the position
# ---------------------------------------------------------------------------


def cells(board: Board) -> npt.NDArray[np.int8]:
    """The ``(9, 9, 3)`` piece-code grid, indexed ``[rank-1, file-1, tier-1]``."""
    return np.asarray(board.cells_arr, dtype=np.int8)


def split_code(code: int) -> tuple[int, int]:
    """``(type_index, color)`` of a non-zero cyboard piece code.

    cyboard encodes ``piece_type_index * 2 + color + 1``.
    """
    return (code - 1) // 2, (code - 1) & 1


def tower(board: Board, rank: int, file: int) -> list[tuple[int, int]]:
    """``(type_index, color)`` bottom-to-top for the tower on ``(rank, file)``."""
    column = cells(board)[rank - 1, file - 1]
    stack: list[tuple[int, int]] = []
    for code in column:
        value = int(code)
        if value == 0:
            break
        stack.append(split_code(value))
    return stack


def top_piece(board: Board, rank: int, file: int) -> tuple[int, int] | None:
    """``(type_index, color)`` of the top piece, or None on an empty square."""
    stack = tower(board, rank, file)
    return stack[-1] if stack else None


def hand_count(board: Board, type_index: int, color: int) -> int:
    """``hand[type_index, color]`` as a plain int."""
    return int(board.hand_arr[type_index, color])


def qualifying_mask(board: Board, rank: int, file: int, color: int) -> int:
    """``Q`` -- the tier-mask of convertible enemies on ``(rank, file)``.

    The test's OWN pure-Python transcription of ``move_gen.ts:402-423`` (quoted
    in the design under "Which subsets are legal"), deliberately independent of
    ``cymovegen._get_betrayal_combos`` and of whatever KOM-028 exposes:

    * ``E`` is the enemy pieces in the tower;
    * a type ``T`` qualifies iff the mover's hand holds at least
      ``count(T in E)`` copies of it -- all-or-nothing over the whole tower,
      not over the chosen subset;
    * ``Q`` is every instance of every qualifying type.

    Returns a raw tier bitmask; tiers above 2 cannot occur where a betray is
    generated (tower height <= 2 there), but this function does not enforce
    that, so a caller can catch an out-of-range bit rather than crash on it.
    """
    enemies = [
        (type_index, tier)
        for tier, (type_index, piece_color) in enumerate(
            tower(board, rank, file), start=1
        )
        if piece_color != color
    ]
    if not enemies:
        return 0

    counts: dict[int, int] = {}
    for type_index, _ in enemies:
        counts[type_index] = counts.get(type_index, 0) + 1

    qualifying = 0
    for type_index, count in counts.items():
        if hand_count(board, type_index, color) >= count:
            for enemy_type, tier in enemies:
                if enemy_type == type_index:
                    qualifying |= 1 << (tier - 1)
    return qualifying


# ---------------------------------------------------------------------------
# FEN mirroring (T8)
# ---------------------------------------------------------------------------


def _tokenize_rank(rank_str: str) -> list[str]:
    """Split a FEN rank into digits, piece letters and ``|X:Y|`` tower groups.

    Same tokenisation as ``game._tokenize_rank``; re-implemented rather than
    imported so the symmetry test does not depend on a private engine helper
    it is partly there to check.
    """
    tokens: list[str] = []
    index = 0
    while index < len(rank_str):
        if rank_str[index] == "|":
            close = rank_str.index("|", index + 1)
            tokens.append(rank_str[index : close + 1])
            index = close + 1
        else:
            tokens.append(rank_str[index])
            index += 1
    return tokens


def mirror_fen(fen: str) -> str:
    """The file-mirrored position: every rank's token order reversed.

    Tiers within a tower are untouched -- a file reflection does not reorder a
    tower -- which is exactly why the selector must survive ``_mirror_action``
    unchanged.
    """
    parts = fen.split(" ")
    parts[0] = "/".join(
        "".join(reversed(_tokenize_rank(rank))) for rank in parts[0].split("/")
    )
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Perft (T9)
# ---------------------------------------------------------------------------


def perft_by_type(fen: str, depth: int) -> dict[str, int]:
    """Move-type-bucketed perft over the Cython engine, reference-shaped.

    Mirrors the generation harness's ``perft``: counts are CUMULATIVE over
    every ply from 1 to *depth* (the harness increments its bucket for each
    move at each level, and re-walks from scratch per depth), a node with no
    moves contributes nothing, and a terminal node is not expanded -- the JS
    walk calls ``Gungi.moves()``, which returns ``[]`` once the game is over.

    The one structural difference: at the last ply the child board is not
    built at all, since the harness's ``walk(child, 0)`` returns without
    looking at it.  That is a pure saving -- tens of thousands of copies at
    depth 3 -- and cannot change a count.
    """
    counts: dict[str, int] = {}

    def walk(board: Board, remaining: int) -> None:
        if remaining == 0:
            return
        if cyengine.is_game_over(board):
            return
        for record in engine_records(board):
            kind = KIND_BY_MT[int(record["move_type"])]
            counts[kind] = counts.get(kind, 0) + 1
            if remaining == 1:
                continue
            child = board.copy()
            cyengine.apply_move(child, record)
            walk(child, remaining - 1)

    walk(board_of_fen(fen), depth)
    return counts
