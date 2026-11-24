"""
gungi.api -- Top-level Gungi game class.

Port of gungi.js gungi.ts (skip UI/animation surface, piece-ids, pgn,
comments, ascii, undo).
"""

from __future__ import annotations

from komugi.engine.fen import INTRO_POSITION, encode_fen, parse_fen
from komugi.engine.move_gen import (
    generate_arata,
    generate_moves_for_square,
    in_check,
    would_be_in_check_after_move,
)
from komugi.engine.utils import (
    MARSHAL,
    SQUARES,
    get,
    get_top,
    is_game_over,
    piece,
)


class Gungi:
    """Top-level game object mirroring the gungi.js Gungi class."""

    def __init__(self, fen: str | None = None):
        self._init_position = fen if fen is not None else INTRO_POSITION
        self._history: list[dict] = []
        self._load_state(self._init_position)

    # ------------------------------------------------------------------
    # Internal state management
    # ------------------------------------------------------------------

    def _load_state(self, fen: str) -> None:
        parsed = parse_fen(fen)
        self._board = parsed["board"]
        self._hand = parsed["hand"]
        self._turn = parsed["turn"]
        self._move_number = parsed["moveNumber"]
        self._drafting = parsed["drafting"]
        self._mode = parsed["mode"]

    def load(self, fen: str) -> None:
        """Load a new position from *fen*."""
        self._load_state(fen)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def fen(self) -> str:
        return encode_fen({
            "board": self._board,
            "hand": self._hand,
            "turn": self._turn,
            "mode": self._mode,
            "drafting": self._drafting,
            "moveNumber": self._move_number,
        })

    def board(self):
        return self._board

    def get(self, square: str):
        return get(square, self._board)

    def get_top(self, square: str):
        return get_top(square, self._board)

    # JS compat alias
    getTop = get_top

    def hand(self, color: str | None = None):
        if color is not None:
            return [p for p in self._hand if p["color"] == color]
        return self._hand

    def turn(self) -> str:
        return self._turn

    def move_number(self) -> int:
        return self._move_number

    # JS compat alias
    moveNumber = move_number

    def get_drafting_rights(self, color: str | None = None):
        if color is not None:
            return self._drafting[color]
        return self._drafting

    # JS compat alias
    getDraftingRights = get_drafting_rights

    def in_draft(self) -> bool:
        return self._drafting["b"] or self._drafting["w"]

    # JS compat alias
    inDraft = in_draft

    # ------------------------------------------------------------------
    # Move generation
    # ------------------------------------------------------------------

    def moves(self, opts: dict | None = None):
        if self.is_game_over():
            return []

        verbose = opts.get("verbose", False) if opts else False
        fen = self.fen()

        if opts and "square" in opts:
            mvs = generate_moves_for_square(opts["square"], fen)
            return mvs if verbose else [m["san"] for m in mvs]
        elif opts and "arata" in opts:
            mvs = generate_arata(opts["arata"], fen)
            return mvs if verbose else [m["san"] for m in mvs]
        elif self.in_draft():
            mvs = []
            for hp in self.hand():
                mvs.extend(generate_arata(hp, fen))
            return mvs if verbose else [m["san"] for m in mvs]
        else:
            board_moves = []
            for sq in SQUARES:
                board_moves.extend(generate_moves_for_square(sq, fen))
            hand_moves = []
            for hp in self.hand():
                hand_moves.extend(generate_arata(hp, fen))
            mvs = board_moves + hand_moves
            return mvs if verbose else [m["san"] for m in mvs]

    # ------------------------------------------------------------------
    # Applying a move
    # ------------------------------------------------------------------

    def move(self, m):
        """Apply a move (SAN string or partial move dict).

        Returns the full Move dict.
        Raises ValueError on illegal move.
        """
        all_moves = self.moves({"verbose": True})

        if isinstance(m, str):
            normalized = m.rstrip("=#")
            found = None
            for mv in all_moves:
                if mv["san"] == normalized or mv["san"] == m:
                    found = mv
                    break
        else:
            found = None
            for mv in all_moves:
                if (
                    m.get("piece") == mv["piece"]
                    and m.get("type") == mv["type"]
                    and m.get("from") == mv["from"]
                    and m.get("to") == mv["to"]
                    and m.get("captured") == mv["captured"]
                    and m.get("draftFinished") == mv["draftFinished"]
                ):
                    found = mv
                    break

        if found is None:
            raise ValueError(f"Illegal move: {m}")

        self._history.append(found)
        self._load_state(found["after"])

        # Append game state symbols to SAN
        if self.is_checkmate():
            self._history[-1]["san"] += "#"
        elif (
            self.is_stalemate()
            or self.is_fourfold_repetition()
            or self.is_insufficient_material()
        ):
            self._history[-1]["san"] += "="

        return found

    # ------------------------------------------------------------------
    # Check / checkmate / stalemate helpers
    # ------------------------------------------------------------------

    def in_check(self, color: str | None = None) -> bool:
        c = color if color is not None else self.turn()
        return in_check(c, self.fen())

    # JS compat alias
    inCheck = in_check

    def _has_escaping_move(self) -> bool:
        fen = self.fen()

        for sq in SQUARES:
            for mv in generate_moves_for_square(sq, fen):
                if not would_be_in_check_after_move(mv):
                    return True

        for hp in self._hand:
            for mv in generate_arata(hp, fen):
                if not would_be_in_check_after_move(mv):
                    return True

        return False

    def is_checkmate(self) -> bool:
        if self.in_draft():
            return False
        if is_game_over(self._board):
            return False
        return self.in_check() and not self._has_escaping_move()

    # JS compat alias
    isCheckmate = is_checkmate

    def is_stalemate(self) -> bool:
        if self.in_draft():
            return False
        if is_game_over(self._board):
            return False
        return not self.in_check() and not self._has_escaping_move()

    # JS compat alias
    isStalemate = is_stalemate

    # ------------------------------------------------------------------
    # Draw detection
    # ------------------------------------------------------------------

    def is_insufficient_material(self) -> bool:
        if self.in_draft():
            return False

        # Hand must contain only marshals (or be empty)
        if any(p["type"] != piece["marshal"] for p in self._hand):
            return False

        # Board must have exactly 2 marshals and nothing else
        pieces = [
            p
            for row in self._board
            for cell in row
            for p in cell
            if p is not None
        ]
        non_marshals = [p for p in pieces if p["type"] != piece["marshal"]]
        if len(non_marshals) > 0:
            return False

        marshals = [p for p in pieces if p["type"] == piece["marshal"]]
        if len(marshals) != 2:
            return False

        m1, m2 = marshals
        r1, f1 = (int(x) for x in m1["square"].split("-"))
        r2, f2 = (int(x) for x in m2["square"].split("-"))

        rank_diff = abs(r1 - r2)
        file_diff = abs(f1 - f2)

        is_adjacent = rank_diff <= 1 and file_diff <= 1
        return not is_adjacent

    # JS compat alias
    isInsufficientMaterial = is_insufficient_material

    def is_fourfold_repetition(self) -> bool:
        states = [mv["before"].split(" ")[0] for mv in self._history]
        if states:
            states.append(self._history[-1]["after"].split(" ")[0])

        freq: dict[str, int] = {}
        for s in states:
            freq[s] = freq.get(s, 0) + 1

        if not freq:
            return False

        return max(freq.values()) == 4

    # JS compat alias
    isFourfoldRepetition = is_fourfold_repetition

    def is_draw(self) -> bool:
        return (
            self.is_stalemate()
            or self.is_fourfold_repetition()
            or self.is_insufficient_material()
        )

    # JS compat alias
    isDraw = is_draw

    def is_game_over(self) -> bool:
        if self.in_draft():
            return False
        if is_game_over(self._board):
            return True  # marshal captured
        if self.is_fourfold_repetition():
            return True
        if self.is_insufficient_material():
            return True
        return self.is_checkmate() or self.is_stalemate()

    # JS compat alias
    isGameOver = is_game_over
