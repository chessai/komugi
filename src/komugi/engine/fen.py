"""
gungi.fen — FEN parser, encoder, and validator for Gungi games.
"""

from __future__ import annotations

from komugi.engine.utils import (
    create_hand_piece_from_fen_code,
    create_piece_from_fen_code,
    piece_to_fen_code,
    setup_code_to_mode,
    setup_mode_to_code,
    symbol_to_name,
)

# ---------------------------------------------------------------------------
# Start-position constants
# ---------------------------------------------------------------------------

INTRO_POSITION = (
    "3img3/1s2n2s1/d1fwdwf1d/9/9/9/D1FWDWF1D/1S2N2S1/3GMI3"
    " J2N2R2D1/j2n2r2d1 w 0 - 1"
)
BEGINNER_POSITION = (
    "3img3/1ra1n1as1/d1fwdwf1d/9/9/9/D1FWDWF1D/1SA1N1AR1/3GMI3"
    " J2N2S1R1D1/j2n2s1r1d1 w 1 - 1"
)
INTERMEDIATE_POSITION = (
    "9/9/9/9/9/9/9/9/9"
    " M1G1I1J2W2N3R2S2F2D4C1A2K1T1/m1g1i1j2w2n3r2s2f2d4c1a2k1t1 w 2 wb 1"
)
ADVANCED_POSITION = (
    "9/9/9/9/9/9/9/9/9"
    " M1G1I1J2W2N3R2S2F2D4C1A2K1T1/m1g1i1j2w2n3r2s2f2d4c1a2k1t1 w 3 wb 1"
)

_VALID_PIECES = "mgijwnrsfdcaktMGIJWNRSFDCAKT"

# ---------------------------------------------------------------------------
# validateFen
# ---------------------------------------------------------------------------


def validate_fen(fen: str) -> dict:
    """Validate a FEN string.

    Returns ``{"ok": True}`` or ``{"ok": False, "error": "..."}``.
    """
    parts = fen.split(" ")
    if len(parts) != 6:
        return {"ok": False, "error": f"expected 6 fields, received {len(parts)}"}

    placement, handpieces, turn, mode, drafting, move_number = parts

    # --- Field 1: piece placement ---
    ranks = placement.split("/")
    if len(ranks) != 9:
        return {
            "ok": False,
            "error": (
                "1st field (piece positions) is invalid "
                f"[expected 9 ranks, received {len(ranks)}]"
            ),
        }

    for i, rank in enumerate(ranks):
        file_parts = rank.split("|")
        file_count = 0

        for file_part in file_parts:
            if ":" in file_part:
                file_count += 1
                tower_pieces = file_part.split(":")
                if len(tower_pieces) > 3:
                    return {
                        "ok": False,
                        "error": (
                            "1st field (piece positions) is invalid "
                            f"[tower size] @({i + 1}-{file_count})"
                        ),
                    }
                if not all(p in _VALID_PIECES for p in tower_pieces):
                    return {
                        "ok": False,
                        "error": (
                            "1st field (piece positions) is invalid "
                            f"[invalid piece] @({i + 1}-{file_count})"
                        ),
                    }
            else:
                for ch in file_part:
                    if ch.isdigit():
                        file_count += int(ch)
                    else:
                        file_count += 1
                        if ch not in _VALID_PIECES:
                            return {
                                "ok": False,
                                "error": (
                                    "1st field (piece positions) is invalid "
                                    f"[invalid piece] @({i + 1}-{file_count})"
                                ),
                            }

        if file_count != 9:
            return {
                "ok": False,
                "error": (
                    "1st field (piece positions) is invalid "
                    f"[expected 9 squares, received {file_count}] in rank: {i + 1}"
                ),
            }

    # --- Field 2: hand pieces ---
    hands = handpieces.split("/")
    if len(hands) != 2:
        return {
            "ok": False,
            "error": (
                "2nd field (hand pieces) is invalid "
                f"[expected 2 hands, received {len(hands)}]"
            ),
        }

    white_hand, black_hand = hands

    if (len(white_hand) == 1 and white_hand != "-") or (
        len(black_hand) == 1 and black_hand != "-"
    ):
        return {
            "ok": False,
            "error": "2nd field (hand pieces) is invalid [invalid piece]",
        }

    if (len(white_hand) > 1 and len(white_hand) % 2 != 0) or (
        len(black_hand) > 1 and len(black_hand) % 2 != 0
    ):
        return {
            "ok": False,
            "error": "2nd field (hand pieces) is invalid [invalid piece count]",
        }

    if len(white_hand) > 1:
        for j in range(0, len(white_hand), 2):
            if white_hand[j] not in "MGIJWNRSFDCAKT":
                return {
                    "ok": False,
                    "error": "2nd field (hand pieces) is invalid [invalid piece]",
                }
            if not white_hand[j + 1].isdigit():
                return {
                    "ok": False,
                    "error": "2nd field (hand pieces) is invalid [invalid count]",
                }

    if len(black_hand) > 1:
        for j in range(0, len(black_hand), 2):
            if black_hand[j] not in "mgijwnrsfdcakt":
                return {
                    "ok": False,
                    "error": "2nd field (hand pieces) is invalid [invalid piece]",
                }
            if not black_hand[j + 1].isdigit():
                return {
                    "ok": False,
                    "error": "2nd field (hand pieces) is invalid [invalid count]",
                }

    # --- Field 3: active player ---
    if turn not in ("w", "b"):
        return {
            "ok": False,
            "error": (
                "3rd field (active player) is invalid "
                f"[expected 'w' or 'b', received {turn}]"
            ),
        }

    # --- Field 4: setup mode ---
    if len(mode) != 1 or mode not in "0123":
        return {
            "ok": False,
            "error": (
                "4th field (setup mode) is invalid "
                f"[expected '0', '1', '2', or '3', received {mode}]"
            ),
        }

    # --- Field 5: drafting availability ---
    if drafting != "-" and drafting not in "wb":
        return {
            "ok": False,
            "error": (
                "5th field (drafting availability) is invalid "
                f"[expected '-', 'w', 'b', or 'wb', received {drafting}]"
            ),
        }

    # --- Field 6: full move number ---
    try:
        int(move_number)
    except ValueError:
        return {
            "ok": False,
            "error": (
                "6th field (full move number) is invalid "
                f"[expected an integer, received {move_number}]"
            ),
        }

    return {"ok": True}


# ---------------------------------------------------------------------------
# parseFEN
# ---------------------------------------------------------------------------


def parse_fen(fen: str) -> dict:
    """Parse a FEN string into a dict with board, hand, turn, mode, drafting, moveNumber."""
    result = validate_fen(fen)
    if not result["ok"]:
        raise ValueError(f"Invalid FEN: {result['error']}")

    placement, handpieces, turn, mode_str, drafting, move_number = fen.split(" ")

    # -- board --
    board: list[list[list]] = []
    for y, rank_str in enumerate(placement.split("/")):
        row: list[list] = []
        for partial in rank_str.split("|"):
            if ":" in partial:
                tower = []
                for z, pc in enumerate(partial.split(":")):
                    tower.append(
                        create_piece_from_fen_code(pc, (y + 1, 9 - len(row), z + 1))
                    )
                row.append(tower)
            else:
                partial_pieces: list = []
                for ch in partial:
                    if ch.isdigit():
                        partial_pieces.extend([None] * int(ch))
                    else:
                        partial_pieces.append(
                            create_piece_from_fen_code(
                                ch,
                                (y + 1, 9 - len(row) - len(partial_pieces), 1),
                            )
                        )
                row.extend([[sq] for sq in partial_pieces])
        board.append(row)

    # -- hand --
    hand: list[dict] = []
    for player_hand in handpieces.split("/"):
        if player_hand != "-":
            for j in range(0, len(player_hand), 2):
                hand.append(
                    create_hand_piece_from_fen_code(
                        player_hand[j], int(player_hand[j + 1])
                    )
                )

    return {
        "board": board,
        "hand": hand,
        "turn": turn,
        "mode": setup_code_to_mode[int(mode_str)],
        "drafting": {
            "w": "w" in drafting,
            "b": "b" in drafting,
        },
        "moveNumber": int(move_number),
    }


# ---------------------------------------------------------------------------
# encodeFEN
# ---------------------------------------------------------------------------


def encode_fen(parsed: dict) -> str:
    """Encode a parsed FEN dict back into a FEN string."""
    placement = ""
    empty_count = 0

    for file_idx in range(9):
        for rank_idx in range(9):
            square = parsed["board"][file_idx][rank_idx]
            if not square[0]:
                empty_count += 1
                continue

            if empty_count:
                placement += str(empty_count)
                empty_count = 0

            if len(square) > 1:
                tower_parts = []
                for p in square:
                    if p is None:
                        continue
                    fc = piece_to_fen_code[symbol_to_name[p["type"]]]
                    tower_parts.append(fc.upper() if p["color"] == "w" else fc)
                placement += "|" + ":".join(tower_parts) + "|"
            else:
                p = square[0]
                fc = piece_to_fen_code[symbol_to_name[p["type"]]]
                placement += fc.upper() if p["color"] == "w" else fc

        if empty_count:
            placement += f"{empty_count}/"
        else:
            placement += "/"
        empty_count = 0

    # Strip trailing '/'
    placement = placement[:-1]

    # -- hand --
    hand_str = ""
    white_hand = [p for p in parsed["hand"] if p["color"] == "w"]
    black_hand = [p for p in parsed["hand"] if p["color"] == "b"]

    for p in white_hand:
        fc = piece_to_fen_code[symbol_to_name[p["type"]]]
        hand_str += f"{fc.upper() if p['color'] == 'w' else fc}{p['count']}"
    hand_str += "/" if white_hand else "-/"

    for p in black_hand:
        fc = piece_to_fen_code[symbol_to_name[p["type"]]]
        hand_str += f"{fc.upper() if p['color'] == 'w' else fc}{p['count']}"
    if not black_hand:
        hand_str += "-"

    # -- drafting --
    draft_keys = [k for k in ("w", "b") if parsed["drafting"][k]]
    drafting_str = "".join(draft_keys) if draft_keys else "-"

    return (
        f"{placement} {hand_str} {parsed['turn']}"
        f" {setup_mode_to_code[parsed['mode']]} {drafting_str}"
        f" {parsed['moveNumber']}"
    )
