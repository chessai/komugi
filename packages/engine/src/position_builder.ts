// KOM-1198 unit 2 (docs/design/kom1198-move-io.md section 4): plain-data
// position builder, host-neutral, in the section 2 rank/file/tier/piece/color
// convention (never a FEN letter or SAN kanji). `buildPosition`/`parsePosition`
// are the inverse pair; `fen.ts`'s `encodeFen`/`parseFen` do the actual FEN
// syntax, so this file only maps PlainPosition <-> ParsedFen plus legality.

import { MARSHAL, MODE_ADVANCED, MODE_BEGINNER, MODE_INTERMEDIATE, MODE_INTRO, NUM_PIECE_TYPES } from './constants.ts';
import { encodeFen, parseFen } from './fen.ts';
import type { FenHandPiece, FenPiece, FenTower, ParsedFen } from './fen.ts';

export interface PlainPiece {
  piece: number; // 0-13
  color: number; // 0-1
}

// Tower bottom to top; [] = empty square.
export type PlainCell = PlainPiece[];

export interface PlainHandEntry {
  piece: number; // 0-13
  color: number; // 0-1
  count: number; // 1-9
}

export interface PlainPosition {
  cells: PlainCell[][]; // cells[rank-1][file-1]; tower index = tier-1
  hand: PlainHandEntry[];
  turn: number; // 0 white, 1 black
  mode: number; // constants.ts MODE_INTRO..MODE_ADVANCED
  drafting: { w: boolean; b: boolean };
  moveNumber: number;
}

// Each check below throws `Invalid position: <reason>` the moment it finds a
// violation (section 4's rules checks), after the shape checks that make the
// rules checks safe to run (a tower's pieces are only inspected once their
// own shape is known-good).
export function validatePosition(position: PlainPosition): void {
  const { cells, hand, turn, mode, drafting, moveNumber } = position;

  if (cells.length !== 9) {
    throw new Error(`Invalid position: cells must have 9 ranks, got ${cells.length}`);
  }
  for (let r = 0; r < 9; r++) {
    const row = cells[r]!;
    if (row.length !== 9) {
      throw new Error(`Invalid position: rank ${r + 1} must have 9 files, got ${row.length}`);
    }
    for (let f = 0; f < 9; f++) {
      const tower = row[f]!;
      if (tower.length > 3) {
        throw new Error(`Invalid position: tower at rank ${r + 1} file ${f + 1} has more than 3 pieces`);
      }
      for (const p of tower) {
        if (!Number.isInteger(p.piece) || p.piece < 0 || p.piece >= NUM_PIECE_TYPES) {
          throw new Error(`Invalid position: piece type out of range at rank ${r + 1} file ${f + 1}`);
        }
        if (p.color !== 0 && p.color !== 1) {
          throw new Error(`Invalid position: color out of range at rank ${r + 1} file ${f + 1}`);
        }
      }
    }
  }

  if (turn !== 0 && turn !== 1) {
    throw new Error(`Invalid position: turn must be 0 or 1, got ${turn}`);
  }

  if (mode !== MODE_INTRO && mode !== MODE_BEGINNER && mode !== MODE_INTERMEDIATE && mode !== MODE_ADVANCED) {
    throw new Error(`Invalid position: mode must be 0-3, got ${mode}`);
  }

  if (!Number.isInteger(moveNumber) || moveNumber < 1) {
    throw new Error(`Invalid position: moveNumber must be an integer >= 1, got ${moveNumber}`);
  }

  for (const h of hand) {
    if (!Number.isInteger(h.piece) || h.piece < 0 || h.piece >= NUM_PIECE_TYPES) {
      throw new Error('Invalid position: hand entry piece type out of range');
    }
    if (h.color !== 0 && h.color !== 1) {
      throw new Error('Invalid position: hand entry color out of range');
    }
    if (!Number.isInteger(h.count) || h.count < 1 || h.count > 9) {
      throw new Error(`Invalid position: hand entry count must be 1-9, got ${h.count}`);
    }
  }
  const handKeys = new Set<number>();
  for (const h of hand) {
    const key = h.piece * 2 + h.color;
    if (handKeys.has(key)) {
      throw new Error('Invalid position: duplicate hand entry for the same piece and color');
    }
    handKeys.add(key);
  }

  // Exactly one marshal per color, counting board plus hand.
  let marshalW = 0;
  let marshalB = 0;
  for (let r = 0; r < 9; r++) {
    const row = cells[r]!;
    for (let f = 0; f < 9; f++) {
      for (const p of row[f]!) {
        if (p.piece === MARSHAL) {
          if (p.color === 0) marshalW++;
          else marshalB++;
        }
      }
    }
  }
  for (const h of hand) {
    if (h.piece === MARSHAL) {
      if (h.color === 0) marshalW += h.count;
      else marshalB += h.count;
    }
  }
  if (marshalW !== 1) {
    throw new Error(`Invalid position: white must have exactly one marshal, got ${marshalW}`);
  }
  if (marshalB !== 1) {
    throw new Error(`Invalid position: black must have exactly one marshal, got ${marshalB}`);
  }

  // A marshal is never covered (nothing stacks on a marshal, any mode); a
  // marshal on top of other pieces requires mode >= 2; tower height > 2
  // requires mode 3.
  for (let r = 0; r < 9; r++) {
    const row = cells[r]!;
    for (let f = 0; f < 9; f++) {
      const tower = row[f]!;
      if (tower.length > 2 && mode !== MODE_ADVANCED) {
        throw new Error(`Invalid position: tower height > 2 requires mode 3, at rank ${r + 1} file ${f + 1}`);
      }
      for (let t = 0; t < tower.length - 1; t++) {
        if (tower[t]!.piece === MARSHAL) {
          throw new Error(
            `Invalid position: a marshal cannot be covered by another piece, at rank ${r + 1} file ${f + 1}`,
          );
        }
      }
      if (tower.length > 1 && tower[tower.length - 1]!.piece === MARSHAL && mode < MODE_INTERMEDIATE) {
        throw new Error(
          `Invalid position: a marshal over other pieces requires mode >= 2, at rank ${r + 1} file ${f + 1}`,
        );
      }
    }
  }

  // Drafting: both, Black only, or neither (White-only is impossible); only
  // in modes 2/3; when exactly one side drafts, turn is that side; while
  // either side drafts, every piece sits in its color's home ranks.
  if (drafting.w && !drafting.b) {
    throw new Error('Invalid position: white-only drafting is impossible');
  }
  if ((drafting.w || drafting.b) && mode < MODE_INTERMEDIATE) {
    throw new Error('Invalid position: drafting is only allowed in mode 2 or 3');
  }
  const draftingSides = (drafting.w ? 1 : 0) + (drafting.b ? 1 : 0);
  if (draftingSides === 1) {
    const draftingColor = drafting.w ? 0 : 1;
    if (turn !== draftingColor) {
      throw new Error('Invalid position: turn must be the drafting side when only one side is drafting');
    }
  }
  if (drafting.w || drafting.b) {
    for (let r = 0; r < 9; r++) {
      const row = cells[r]!;
      for (let f = 0; f < 9; f++) {
        for (const p of row[f]!) {
          if (p.color === 0 && r < 6) {
            throw new Error(
              `Invalid position: while drafting, every piece must sit in its color's home ranks (white piece at rank ${r + 1})`,
            );
          }
          if (p.color === 1 && r > 2) {
            throw new Error(
              `Invalid position: while drafting, every piece must sit in its color's home ranks (black piece at rank ${r + 1})`,
            );
          }
        }
      }
    }
  }
}

// -> canonical FEN. Section 2's file reversal (`f = 8 - j`) is applied once,
// here, going from `cells[r][f]` to `board[r][j]`.
export function buildPosition(position: PlainPosition): string {
  validatePosition(position);

  const board: FenTower[][] = [];
  for (let r = 0; r < 9; r++) {
    const row: FenTower[] = new Array(9);
    const cellRow = position.cells[r]!;
    for (let j = 0; j < 9; j++) {
      const f = 8 - j;
      const tower = cellRow[f]!;
      row[j] = tower.length === 0 ? [null] : tower.map((p): FenPiece => ({ type: p.piece, color: p.color }));
    }
    board.push(row);
  }

  const hand: FenHandPiece[] = position.hand.map((h) => ({ type: h.piece, color: h.color, count: h.count }));

  const parsed: ParsedFen = {
    board,
    hand,
    turn: position.turn === 0 ? 'w' : 'b',
    mode: position.mode,
    drafting: { w: position.drafting.w, b: position.drafting.b },
    moveNumber: position.moveNumber,
  };
  return encodeFen(parsed);
}

// Inverse of buildPosition. No legality check: parseFen validates FEN syntax
// only (validateFen), exactly like canonicalFen.
export function parsePosition(fen: string): PlainPosition {
  const parsed = parseFen(fen);

  const cells: PlainCell[][] = [];
  for (let r = 0; r < 9; r++) {
    const row: PlainCell[] = new Array(9);
    const parsedRow = parsed.board[r]!;
    for (let j = 0; j < 9; j++) {
      const f = 8 - j;
      const tower = parsedRow[j]!;
      row[f] = tower[0] === null ? [] : tower.map((p) => ({ piece: p!.type, color: p!.color }));
    }
    cells.push(row);
  }

  const hand: PlainHandEntry[] = parsed.hand.map((h) => ({ piece: h.type, color: h.color, count: h.count }));

  return {
    cells,
    hand,
    turn: parsed.turn === 'w' ? 0 : 1,
    mode: parsed.mode,
    drafting: { w: parsed.drafting.w, b: parsed.drafting.b },
    moveNumber: parsed.moveNumber,
  };
}
