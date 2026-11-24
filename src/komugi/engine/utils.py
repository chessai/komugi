"""
gungi.utils — Foundation types, constants, and board helpers for the gungi engine.

Types (dict-based):
  Piece:     {"square": str, "tier": int, "type": PieceType (kanji), "color": Color}
             Optional key "id": str.
  HandPiece: {"type": PieceType, "color": Color, "count": int}
             Optional key "id": str.
  Board:     list[list[list[Piece | None]]]
             9×9 grid indexed as board[rank-1][9-file].
             Each cell is a tower: [None] when empty, or [piece, ...] bottom-to-top.
  Move:      {"color": Color, "piece": PieceType, "to": str, "san": str,
              "before": str, "after": str, "type": str}
             Optional keys: "from", "draftFinished", "captured".
"""

from __future__ import annotations

import json
from typing import Any

# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def inverse_map(m: dict[str, Any]) -> dict[Any, str]:
    """Return {v: k for k, v in m.items()}."""
    return {v: k for k, v in m.items()}

# ---------------------------------------------------------------------------
# Color constants
# ---------------------------------------------------------------------------

WHITE: str = "w"
BLACK: str = "b"

# ---------------------------------------------------------------------------
# Move-type constants
# ---------------------------------------------------------------------------

TSUKE: str = "付"
TAKE: str = "取"
BETRAY: str = "返"
ARATA: str = "新"

# ---------------------------------------------------------------------------
# Piece-type constants (kanji)
# ---------------------------------------------------------------------------

MARSHAL: str = "帥"
GENERAL: str = "大"
LIEUTENANT_GENERAL: str = "中"
MAJOR_GENERAL: str = "小"
WARRIOR: str = "侍"
LANCER: str = "槍"
RIDER: str = "馬"
SPY: str = "忍"
FORTRESS: str = "砦"
SOLDIER: str = "兵"
CANNON: str = "砲"
ARCHER: str = "弓"
MUSKETEER: str = "筒"
TACTICIAN: str = "謀"

# ---------------------------------------------------------------------------
# Setup modes
# ---------------------------------------------------------------------------

setup_mode_to_code: dict[str, int] = {
    "intro": 0,
    "beginner": 1,
    "intermediate": 2,
    "advanced": 3,
}

setup_code_to_mode: dict[int, str] = inverse_map(setup_mode_to_code)

# ---------------------------------------------------------------------------
# SQUARES — 81 squares in the same order as utils.ts
# For x in 1..9, for y in 0..8: f"{x}-{9-y}"
# ---------------------------------------------------------------------------

SQUARES: list[str] = [
    f"{x}-{9 - y}" for x in range(1, 10) for y in range(9)
]

# ---------------------------------------------------------------------------
# Piece name / symbol / FEN-code mappings
# ---------------------------------------------------------------------------

# name -> kanji symbol
piece: dict[str, str] = {
    "marshal": "帥",
    "general": "大",
    "lieutenant_general": "中",
    "major_general": "小",
    "warrior": "侍",
    "lancer": "槍",
    "rider": "馬",
    "spy": "忍",
    "fortress": "砦",
    "soldier": "兵",
    "cannon": "砲",
    "archer": "弓",
    "musketeer": "筒",
    "tactician": "謀",
}

# kanji symbol -> name
symbol_to_name: dict[str, str] = inverse_map(piece)

# name -> FEN code
piece_to_fen_code: dict[str, str] = {
    "marshal": "m",
    "general": "g",
    "lieutenant_general": "i",
    "major_general": "j",
    "warrior": "w",
    "lancer": "n",
    "rider": "r",
    "spy": "s",
    "fortress": "f",
    "soldier": "d",
    "cannon": "c",
    "archer": "a",
    "musketeer": "k",
    "tactician": "t",
}

# FEN code -> name
fen_code_to_piece: dict[str, str] = inverse_map(piece_to_fen_code)

# kanji -> romanized canonical name
CANONICAL_NAMES: dict[str, str] = {
    "帥": "sui",
    "大": "taishou",
    "中": "chuujou",
    "小": "shoushou",
    "侍": "samurai",
    "槍": "yari",
    "馬": "kiba",
    "忍": "shinobi",
    "砦": "toride",
    "兵": "hyou",
    "砲": "oodzutsu",
    "弓": "yumi",
    "筒": "tsutsu",
    "謀": "boushou",
}

# kanji -> English name
ENGLISH_NAMES: dict[str, str] = {
    "帥": "marshal",
    "大": "general",
    "中": "lieutenant general",
    "小": "major general",
    "侍": "warrior",
    "槍": "lancer",
    "馬": "rider",
    "忍": "spy",
    "砦": "fortress",
    "兵": "soldier",
    "砲": "cannon",
    "弓": "archer",
    "筒": "musketeer",
    "謀": "tactician",
}

# kanji -> FEN code
FEN_CODES: dict[str, str] = {
    "帥": "m",
    "大": "g",
    "中": "i",
    "小": "j",
    "侍": "w",
    "槍": "n",
    "馬": "r",
    "忍": "s",
    "砦": "f",
    "兵": "d",
    "砲": "c",
    "弓": "a",
    "筒": "k",
    "謀": "t",
}

# ---------------------------------------------------------------------------
# Type aliases (for documentation; plain dicts at runtime)
# ---------------------------------------------------------------------------

# Piece  = {"square": str, "tier": int, "type": str, "color": str}
# HandPiece = {"type": str, "color": str, "count": int}
# Board = list[list[list[Piece | None]]]

# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def create_piece_from_fen_code(
    code: str, position: tuple[int, int, int]
) -> dict[str, Any]:
    """Create a Piece dict from a FEN code and (rank, file, tier) position.

    Lowercase code → black; uppercase → white.
    """
    y, x, z = position
    color = BLACK if code == code.lower() else WHITE
    name = fen_code_to_piece[code.lower()]
    return {
        "square": f"{y}-{x}",
        "tier": z,
        "type": piece[name],
        "color": color,
    }


def create_hand_piece_from_fen_code(
    code: str, count: int
) -> dict[str, Any]:
    """Create a HandPiece dict from a FEN code and count.

    Lowercase code → black; uppercase → white.
    """
    color = BLACK if code == code.lower() else WHITE
    name = fen_code_to_piece[code.lower()]
    return {
        "type": piece[name],
        "count": count,
        "color": color,
    }

# ---------------------------------------------------------------------------
# Board helpers
# ---------------------------------------------------------------------------

def get(square: str, board: list) -> list | None:
    """Return the tower (list of Piece dicts) at *square*, or None if empty/OOB."""
    parts = square.split("-")
    rank, file = int(parts[0]), int(parts[1])
    if rank < 1 or rank > 9 or file < 1 or file > 9:
        return None
    s = board[rank - 1][9 - file]
    if s[0] is None:
        return None
    return s


def get_top(square: str, board: list) -> dict | None:
    """Return the top piece of the tower at *square*, or None."""
    tower = get(square, board)
    if tower is None:
        return None
    return tower[-1]


def put(p: dict, board: list) -> None:
    """Place *p* (a Piece dict) on top of its tower on the board (mutates board)."""
    parts = p["square"].split("-")
    rank, file = int(parts[0]), int(parts[1])
    occupied = board[rank - 1][9 - file]
    if occupied[0] is None:
        board[rank - 1][9 - file] = [p]
    else:
        board[rank - 1][9 - file].append(p)


def remove(square: str, pieces: list[dict], board: list) -> None:
    """Remove *pieces* from the tower at *square* (by JSON equality). Mutates board."""
    parts = square.split("-")
    rank, file = int(parts[0]), int(parts[1])
    tower = get(square, board)
    if tower is None:
        return

    pieces_json = [json.dumps(p, sort_keys=True) for p in pieces]
    new_tower = [
        tp for tp in tower
        if json.dumps(tp, sort_keys=True) not in pieces_json
    ]

    if len(new_tower) == 0:
        board[rank - 1][9 - file] = [None]
    else:
        board[rank - 1][9 - file] = new_tower


def remove_top(square: str, board: list) -> None:
    """Pop the top piece from the tower at *square*. Mutates board."""
    parts = square.split("-")
    rank, file = int(parts[0]), int(parts[1])
    tower = get(square, board)
    if tower is None:
        return
    tower.pop()
    if len(tower) == 0:
        board[rank - 1][9 - file] = [None]


def update_hand(
    pieces: list[dict], hand: list[dict], opposite_color: bool = False
) -> list[dict]:
    """Decrement hand counts for each piece; remove entry when count hits 0. Mutates hand."""
    for p in pieces:
        if opposite_color:
            c = WHITE if p["color"] == BLACK else BLACK
        else:
            c = p["color"]
        hp = None
        hpi = -1
        for i, h in enumerate(hand):
            if h["type"] == p["type"] and h["color"] == c:
                hp = h
                hpi = i
                break
        if hp is not None:
            hp["count"] -= 1
            if hp["count"] == 0:
                hand.pop(hpi)
    return hand


def convert(
    square: str, pieces: list[dict], board: list
) -> None:
    """Flip the color of matching pieces in the tower at *square*. Mutates board."""
    parts = square.split("-")
    rank, file = int(parts[0]), int(parts[1])
    tower = get(square, board)
    if tower is None:
        return

    pieces_json = {json.dumps(p, sort_keys=True) for p in pieces}
    new_tower = []
    for tp in tower:
        if json.dumps(tp, sort_keys=True) in pieces_json:
            tp["color"] = WHITE if tp["color"] == BLACK else BLACK
        new_tower.append(tp)
    board[rank - 1][9 - file] = new_tower


def is_game_over(board: list) -> bool:
    """Return True when fewer or more than 2 marshals remain on the board."""
    count = 0
    for row in board:
        for cell in row:
            for p in cell:
                if p is not None and p.get("type") == MARSHAL:
                    count += 1
    return count != 2
