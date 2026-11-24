"""
gungi.move_gen — Move generation, makeMove, SAN, and check detection.

Port of gungi.js move_gen.ts.  Produces PSEUDO-LEGAL moves (no self-check
filtering).
"""

from __future__ import annotations

import math
from typing import Any

from komugi.engine.fen import encode_fen, parse_fen
from komugi.engine.utils import (
    MARSHAL,
    SQUARES,
    ARCHER,
    CANNON,
    MUSKETEER,
    TACTICIAN,
    convert,
    get,
    get_top,
    put,
    remove,
    remove_top,
    update_hand,
)

# ---------------------------------------------------------------------------
# Direction table: 8 directions
#   [NW, N, NE, E, W, SE, S, SW]  — matches gungi.js dirs exactly
# ---------------------------------------------------------------------------

dirs: list[list[int]] = [
    [-1,  1],
    [-1,  0],
    [-1, -1],
    [ 0,  1],
    [ 0, -1],
    [ 1,  1],
    [ 1,  0],
    [ 1, -1],
]

# ---------------------------------------------------------------------------
# pieceProbes table
#
# Each entry is a list of 8 cells (one per direction in *dirs*).
# A cell is either:
#   int n      → pval=n, pcarry=1
#   [a, b]     → pval=a (start offset), pcarry=b (carry)
#   math.inf   → slider (pval=inf, pcarry=1)
# ---------------------------------------------------------------------------

piece_probes: dict[str, list] = {
    "帥": [1, 1, 1, 1, 1, 1, 1, 1],
    "大": [1, math.inf, 1, math.inf, math.inf, 1, math.inf, 1],
    "中": [math.inf, 1, math.inf, 1, 1, math.inf, 1, math.inf],
    "小": [1, 1, 1, 1, 1, 0, 1, 0],
    "侍": [1, 1, 1, 0, 0, 0, 1, 0],
    "槍": [1, [1, 2], 1, 0, 0, 0, 1, 0],
    "馬": [0, [1, 2], 0, 1, 1, 0, [1, 2], 0],
    "忍": [[1, 2], 0, [1, 2], 0, 0, [1, 2], 0, [1, 2]],
    "砦": [0, 1, 0, 1, 1, 1, 0, 1],
    "兵": [0, 1, 0, 0, 0, 0, 1, 0],
    "砲": [0, 3, 0, 1, 1, 0, 1, 0],
    "弓": [2, 2, 2, 0, 0, 0, 1, 0],
    "筒": [0, 2, 0, 0, 0, 1, 0, 1],
    "謀": [1, 0, 1, 0, 0, 0, 1, 0],
}

# ---------------------------------------------------------------------------
# getAvailableSquares
# ---------------------------------------------------------------------------

_LEAP_PIECES = {CANNON, MUSKETEER, ARCHER}


def _get_available_squares(
    direction: list[int],
    start: list[int],
    origin: list[int],
    length: float,
    board: list,
) -> list[str]:
    """Return reachable squares along *direction* from *start*, given *origin* piece."""
    y, x = start
    py, px = origin
    dy, dx = direction
    origin_piece = get_top(f"{py}-{px}", board)
    if origin_piece is None:
        return []

    # Archer diagonal wing check
    if (
        origin_piece["type"] == ARCHER
        and abs(dy) == 1
        and abs(dx) == 1
    ):
        wing_sq = f"{py + dy}-{px + dx}"
        if 1 <= py + dy <= 9 and 1 <= px + dx <= 9:
            wing = get_top(wing_sq, board)
            if wing is not None and wing["tier"] > origin_piece["tier"]:
                return []

    available: list[str] = []

    # Reverse probe: walk backward from start toward origin
    rx, ry = x, y
    while rx != px or ry != py:
        rx -= dx
        ry -= dy
        if not (1 <= ry <= 9 and 1 <= rx <= 9):
            break
        p = get_top(f"{ry}-{rx}", board)
        if p is not None and p["tier"] > origin_piece["tier"]:
            return []

        side = -1 if origin_piece["color"] == "b" else 1
        below_sq = f"{ry + side}-{rx}"
        if 1 <= ry + side <= 9 and 1 <= rx <= 9:
            below = get_top(below_sq, board)
            if below is not None and below["square"] == f"{py}-{px}":
                break

    # Forward probe
    fx, fy = x, y
    step = 0
    while step < length:
        if fx < 1 or fx > 9 or fy < 1 or fy > 9:
            break

        p = get_top(f"{fy}-{fx}", board)
        if p is not None and p["tier"] > origin_piece["tier"]:
            break

        available.append(f"{fy}-{fx}")

        if p is not None:
            if origin_piece["type"] not in _LEAP_PIECES:
                break
            moving_forward = (dy < 0) if origin_piece["color"] == "w" else (dy > 0)
            if not moving_forward:
                break

        fx += dx
        fy += dy
        step += 1

    return available


# ---------------------------------------------------------------------------
# Betrayal combinations
# ---------------------------------------------------------------------------

def _generate_combinations(items: list) -> list[list]:
    """Return all non-empty subsets of *items* in recursive pre-order."""
    result: list[list] = []

    def helper(start: int, current: list):
        if len(current) > 0:
            result.append(list(current))
        for i in range(start, len(items)):
            current.append(items[i])
            helper(i + 1, current)
            current.pop()

    helper(0, [])
    return result


def _get_betrayal_combos(
    tower: list[dict],
    hand: list[dict],
    color: str,
) -> list[list[dict]]:
    """Return all betrayal combo subsets for a tactician move."""
    enemies = [p for p in tower if p["color"] != color]
    if not enemies:
        return []

    player_hand = [p for p in hand if p["color"] == color]

    # Count enemies by type
    enemy_count: dict[str, int] = {}
    for e in enemies:
        enemy_count[e["type"]] = enemy_count.get(e["type"], 0) + 1

    # A type qualifies iff hand has count >= number of that type in tower
    betrayal_options: list[dict] = []
    for etype, ecount in enemy_count.items():
        if any(p["type"] == etype and p["count"] >= ecount for p in player_hand):
            betrayal_options.extend(e for e in enemies if e["type"] == etype)

    return _generate_combinations(betrayal_options)


# ---------------------------------------------------------------------------
# generateMovesForSquare
# ---------------------------------------------------------------------------

def generate_moves_for_square(square: str, fen: str) -> list[dict]:
    """Generate all pseudo-legal moves for the piece on top of *square*."""
    parsed = parse_fen(fen)
    board = parsed["board"]
    turn = parsed["turn"]
    mode = parsed["mode"]
    hand = parsed["hand"]

    pc = get_top(square, board)
    if pc is None or turn != pc["color"]:
        return []

    parts = square.split("-")
    py, px = int(parts[0]), int(parts[1])

    probes = piece_probes[pc["type"]]
    reachable: list[str] = []
    for i, probe in enumerate(probes):
        if isinstance(probe, list):
            pval = probe[0]
            pcarry = probe[1]
        else:
            pval = probe
            pcarry = 1

        if pval < 1:
            continue

        dy, dx = dirs[i]
        if pc["color"] == "b":
            dy = -dy
            dx = -dx

        if pval == math.inf:
            sy, sx = py + dy, px + dx
            length: float = math.inf
        else:
            sy, sx = py + pval * dy, px + dx
            length = pc["tier"] + pcarry - 1

        reachable.extend(
            _get_available_squares([dy, dx], [sy, sx], [py, px], length, board)
        )

    max_tier = 3 if mode == "advanced" else 2
    marshal_can_stack = mode in ("advanced", "intermediate")

    moves: list[dict] = []
    for s in reachable:
        p = get_top(s, board)
        t = get(s, board)

        if p is None or t is None:
            # empty square → route
            moves.append(_create_move(pc, f"{s}-1", fen, "route"))
        else:
            # tsuke
            if p["tier"] < max_tier and p["type"] != MARSHAL:
                if pc["type"] != MARSHAL or marshal_can_stack:
                    moves.append(
                        _create_move(pc, f"{s}-{p['tier'] + 1}", fen, "tsuke")
                    )

                    # betrayal (tactician only)
                    if (
                        pc["type"] == TACTICIAN
                        and any(tp["color"] != pc["color"] for tp in t)
                    ):
                        combos = _get_betrayal_combos(t, hand, pc["color"])
                        for combo in combos:
                            moves.append(
                                _create_move(
                                    pc,
                                    f"{s}-{p['tier'] + 1}",
                                    fen,
                                    "betray",
                                    combo,
                                )
                            )

            # capture (enemy on top)
            if p["color"] != pc["color"]:
                new_tier = sum(1 for tp in t if tp["color"] == pc["color"]) + 1
                captured = [tp for tp in t if tp["color"] != pc["color"]]
                moves.append(
                    _create_move(pc, f"{s}-{new_tier}", fen, "capture", captured)
                )

    return moves


# ---------------------------------------------------------------------------
# generateArata
# ---------------------------------------------------------------------------

def generate_arata(piece_hp: dict, fen: str) -> list[dict]:
    """Generate arata (hand placement) moves for *piece_hp* (a HandPiece)."""
    parsed = parse_fen(fen)
    board = parsed["board"]
    turn = parsed["turn"]
    mode = parsed["mode"]
    hand = parsed["hand"]
    drafting = parsed["drafting"]

    if piece_hp is None or turn != piece_hp["color"]:
        return []

    # If marshal is still in hand, only marshal may be placed
    is_marshal_placed = not any(
        p["type"] == MARSHAL and p["color"] == piece_hp["color"] for p in hand
    )
    if not is_marshal_placed and piece_hp["type"] != MARSHAL:
        return []

    is_draft = drafting["b"] or drafting["w"]
    max_tier = 3 if mode == "advanced" else 2

    ranks: list[int] = []
    maybe: list[int] = []

    if is_draft:
        ranks = [7, 8, 9] if piece_hp["color"] == "w" else [1, 2, 3]
    else:
        start = 1 if piece_hp["color"] == "b" else 9
        end = 9 if piece_hp["color"] == "b" else 1
        step = 1 if piece_hp["color"] == "b" else -1
        rank = start
        while True:
            if piece_hp["color"] == "b":
                if rank > end:
                    break
            else:
                if rank < end:
                    break
            tops = [get_top(f"{rank}-{i + 1}", board) for i in range(9)]
            if any(p is not None and p["color"] == piece_hp["color"] for p in tops):
                ranks.extend(maybe)
                ranks.append(rank)
                maybe = []
            else:
                maybe.append(rank)
            rank += step

    squares = [f"{rank}-{i + 1}" for rank in ranks for i in range(9)]

    moves: list[dict] = []
    for s in squares:
        p = get_top(s, board)
        arata_piece: dict[str, Any] = {
            "type": piece_hp["type"],
            "color": piece_hp["color"],
            "square": s,
            "tier": 0,
        }

        if (
            p is None
            or (
                p["color"] == arata_piece["color"]
                and p["tier"] < max_tier
                and p["type"] != MARSHAL
            )
        ):
            t = (p["tier"] if p is not None else 0) + 1
            tower = get(s, board)
            betrayal_combos = (
                _get_betrayal_combos(tower, hand, piece_hp["color"])
                if piece_hp["type"] == TACTICIAN and tower is not None
                else []
            )
            player_hand_count = sum(
                h["count"] for h in hand if h["color"] == piece_hp["color"]
            )
            is_last_piece = player_hand_count == 1

            if drafting[piece_hp["color"]]:
                if not is_last_piece:
                    moves.append(
                        _create_move(arata_piece, f"{s}-{t}", fen, "arata")
                    )
                    for combo in betrayal_combos:
                        moves.append(
                            _create_move(
                                arata_piece, f"{s}-{t}", fen, "arata", combo
                            )
                        )
                moves.append(
                    _create_move(
                        arata_piece, f"{s}-{t}", fen, "arata", [], True
                    )
                )
                for combo in betrayal_combos:
                    moves.append(
                        _create_move(
                            arata_piece, f"{s}-{t}", fen, "arata", combo, True
                        )
                    )
            else:
                moves.append(
                    _create_move(arata_piece, f"{s}-{t}", fen, "arata")
                )
                for combo in betrayal_combos:
                    moves.append(
                        _create_move(
                            arata_piece, f"{s}-{t}", fen, "arata", combo
                        )
                    )

    return moves


# ---------------------------------------------------------------------------
# createMove / makeMove / SAN
# ---------------------------------------------------------------------------

def _create_move(
    pc: dict,
    to: str,
    fen: str,
    move_type: str,
    captured: list[dict] | None = None,
    draft_finished: bool | None = None,
) -> dict:
    """Build a Move dict including SAN and after-FEN."""
    from_str = f"({pc['square']}-{pc['tier']})" if pc["tier"] != 0 else ""
    to_parts = to.split("-")
    to_tier = int(to_parts[-1])

    arata = "新" if move_type == "arata" else ""
    capture_marker = "取" if move_type == "capture" else ""
    tsuke_marker = (
        "付"
        if (
            move_type == "tsuke"
            or (to_tier != 1 and to_tier - pc["tier"] > 0)
        )
        else ""
    )
    betray_marker = ""
    if move_type == "betray" or (
        move_type == "arata" and captured and len(captured) > 0
    ):
        betray_marker = "返" + "".join(p["type"] for p in captured)
    draft_done = "終" if draft_finished else ""

    # betray suppresses tsuke
    san_suffix = betray_marker or tsuke_marker

    san = f"{arata}{pc['type']}{from_str}{capture_marker}({to}){san_suffix}{draft_done}"

    move: dict[str, Any] = {
        "piece": pc["type"],
        "color": pc["color"],
        "from": from_str[1:-1] if from_str else "",
        "to": to,
        "type": move_type,
        "san": san,
        "before": fen,
        "after": "",
        "draftFinished": draft_finished,
        "captured": captured,
    }

    after, game_over_san = _make_move(move, fen)
    move["after"] = after
    move["san"] += game_over_san

    return move


def _make_move(move: dict, fen: str) -> tuple[str, str]:
    """Apply *move* to *fen* and return (after_fen, game_over_san_suffix)."""
    parsed = parse_fen(fen)
    board = parsed["board"]
    hand = parsed["hand"]
    turn = parsed["turn"]
    drafting = parsed["drafting"]
    move_number = parsed["moveNumber"]
    mode = parsed["mode"]

    to_parts = move["to"].split("-")
    rank_str, file_str, tier_str = to_parts[0], to_parts[1], to_parts[2]
    to_piece: dict[str, Any] = {
        "type": move["piece"],
        "color": move["color"],
        "square": f"{rank_str}-{file_str}",
        "tier": int(tier_str),
    }

    if move["type"] in ("route", "tsuke"):
        remove_top(move["from"], board)
    elif move["type"] == "capture":
        remove_top(move["from"], board)
        remove(f"{rank_str}-{file_str}", move["captured"], board)
    elif move["type"] == "betray":
        remove_top(move["from"], board)
        convert(f"{rank_str}-{file_str}", move["captured"], board)
        update_hand(move["captured"], hand, True)
    elif move["type"] == "arata":
        update_hand([to_piece], hand)
        if move.get("captured") and len(move["captured"]) > 0:
            convert(f"{rank_str}-{file_str}", move["captured"], board)
            update_hand(move["captured"], hand, True)
        if move.get("draftFinished"):
            drafting[move["color"]] = False
            if move["color"] == "b":
                drafting["w"] = False

    put(to_piece, board)

    # moveNumber increment
    if turn == "b" and (drafting["w"] or not drafting["b"]):
        move_number += 1
    if (
        turn == "w"
        and not drafting["b"]
        and not drafting["w"]
        and move.get("draftFinished")
    ):
        move_number += 1

    # Turn flip
    if drafting["w"] == drafting["b"]:
        if not move.get("draftFinished") or turn != "w":
            turn = "b" if turn == "w" else "w"
    else:
        if not drafting["b"] and turn == "b":
            turn = "w"
        elif not drafting["w"] and turn == "w":
            turn = "b"

    game_over_san = ""
    if move.get("captured") and any(
        p["type"] == MARSHAL for p in move["captured"]
    ):
        game_over_san = "#"

    after = encode_fen(
        {
            "board": board,
            "hand": hand,
            "mode": mode,
            "turn": turn,
            "drafting": drafting,
            "moveNumber": move_number,
        }
    )
    return after, game_over_san


# ---------------------------------------------------------------------------
# Attack / check queries
# ---------------------------------------------------------------------------

def _get_attacked_squares(square: str, board: list) -> list[str]:
    """Return all squares attacked by the piece on top of *square*."""
    pc = get_top(square, board)
    if pc is None:
        return []

    parts = square.split("-")
    py, px = int(parts[0]), int(parts[1])
    probes = piece_probes[pc["type"]]

    attacked: list[str] = []
    for i, probe in enumerate(probes):
        if isinstance(probe, list):
            pval = probe[0]
            pcarry = probe[1]
        else:
            pval = probe
            pcarry = 1

        if pval < 1:
            continue

        dy, dx = dirs[i]
        if pc["color"] == "b":
            dy = -dy
            dx = -dx

        if pval == math.inf:
            sy, sx = py + dy, px + dx
            length: float = math.inf
        else:
            sy, sx = py + pval * dy, px + dx
            length = pc["tier"] + pcarry - 1

        attacked.extend(
            _get_available_squares([dy, dx], [sy, sx], [py, px], length, board)
        )
    return attacked


def is_square_attacked(square: str, by_color: str, board: list) -> bool:
    """Return True if *square* is attacked by any piece of *by_color*."""
    parts = square.split("-")
    target_rank, target_file = int(parts[0]), int(parts[1])
    target_square = f"{target_rank}-{target_file}"

    for sq in SQUARES:
        pc = get_top(sq, board)
        if pc is None or pc["color"] != by_color:
            continue
        attacked = _get_attacked_squares(sq, board)
        if any(s.startswith(target_square) for s in attacked):
            return True
    return False


def in_check(color: str, fen: str) -> bool:
    """Return True if *color*'s marshal is in check."""
    parsed = parse_fen(fen)
    board = parsed["board"]
    opposite = "b" if color == "w" else "w"

    for sq in SQUARES:
        pc = get_top(sq, board)
        if pc is not None and pc["type"] == MARSHAL and pc["color"] == color:
            return is_square_attacked(sq, opposite, board)

    return False


def would_be_in_check_after_move(move: dict) -> bool:
    """Return True if *move.color*'s marshal is in check after the move."""
    return in_check(move["color"], move["after"])


# ---------------------------------------------------------------------------
# generate_all_moves  (mirrors gungi.ts moves())
# ---------------------------------------------------------------------------

def generate_all_moves(fen: str) -> list[dict]:
    """Generate all pseudo-legal moves for the side to move.

    During draft: all arata moves for all hand pieces.
    Otherwise: all board moves + all arata (hand) moves.
    """
    parsed = parse_fen(fen)
    drafting = parsed["drafting"]
    hand = parsed["hand"]

    if drafting["b"] or drafting["w"]:
        # Draft phase: only arata
        return [m for hp in hand for m in generate_arata(hp, fen)]

    # Normal play
    board_moves = [
        m for sq in SQUARES for m in generate_moves_for_square(sq, fen)
    ]
    hand_moves = [m for hp in hand for m in generate_arata(hp, fen)]
    return board_moves + hand_moves
