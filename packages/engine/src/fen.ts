// FEN validate / parse / encode (src/komugi/engine/fen.py:44-345) and the typed
// board conversions board_from_fen / board_to_fen (src/komugi/engine/cyboard.pyx:180-339).

import { Board, colorOf, makeCode, typeIndex } from './board.ts';
import { FEN_LETTERS, FEN_LETTER_TO_INDEX } from './constants.ts';

export type FenPiece = { type: number; color: number };
// A square: [null] when empty, else its pieces bottom to top.
export type FenTower = (FenPiece | null)[];
export type FenHandPiece = { type: number; color: number; count: number };
export type ParsedFen = {
  // board[rank-1][9-file]
  board: FenTower[][];
  hand: FenHandPiece[];
  turn: string;
  mode: number;
  drafting: { w: boolean; b: boolean };
  moveNumber: number;
};
export type FenValidation = { ok: true } | { ok: false; error: string };

const VALID_PIECES = 'mgijwnrsfdcaktMGIJWNRSFDCAKT';
const WHITE_HAND_LETTERS = 'MGIJWNRSFDCAKT';
const BLACK_HAND_LETTERS = 'mgijwnrsfdcakt';
const DIGIT = /^[0-9]$/;
// Python int(): optional surrounding whitespace, sign, digits with single underscores.
const PY_INT = /^\s*[+-]?[0-9]+(?:_[0-9]+)*\s*$/;

function isDigit(ch: string | undefined): boolean {
  return ch !== undefined && DIGIT.test(ch);
}

function pyInt(s: string): number {
  return Number(s.trim().replace(/_/g, '')) + 0; // + 0 folds -0 into 0
}

// Python's `a in b` for strings is a substring test, which String.includes matches.
function pyIn(needle: string, hay: string): boolean {
  return hay.includes(needle);
}

export function validateFen(fen: string): FenValidation {
  const parts = fen.split(' ');
  if (parts.length !== 6) {
    return { ok: false, error: `expected 6 fields, received ${parts.length}` };
  }
  const [placement, handpieces, turn, mode, drafting, moveNumber] = parts as [
    string,
    string,
    string,
    string,
    string,
    string,
  ];

  const ranks = placement.split('/');
  if (ranks.length !== 9) {
    return {
      ok: false,
      error: `1st field (piece positions) is invalid [expected 9 ranks, received ${ranks.length}]`,
    };
  }
  for (let i = 0; i < ranks.length; i++) {
    let fileCount = 0;
    for (const filePart of ranks[i]!.split('|')) {
      if (filePart.includes(':')) {
        fileCount += 1;
        const towerPieces = filePart.split(':');
        if (towerPieces.length > 3) {
          return {
            ok: false,
            error: `1st field (piece positions) is invalid [tower size] @(${i + 1}-${fileCount})`,
          };
        }
        if (!towerPieces.every((p) => pyIn(p, VALID_PIECES))) {
          return {
            ok: false,
            error: `1st field (piece positions) is invalid [invalid piece] @(${i + 1}-${fileCount})`,
          };
        }
      } else {
        for (const ch of filePart) {
          if (isDigit(ch)) {
            fileCount += Number(ch);
          } else {
            fileCount += 1;
            if (!pyIn(ch, VALID_PIECES)) {
              return {
                ok: false,
                error: `1st field (piece positions) is invalid [invalid piece] @(${i + 1}-${fileCount})`,
              };
            }
          }
        }
      }
    }
    if (fileCount !== 9) {
      return {
        ok: false,
        error:
          `1st field (piece positions) is invalid [expected 9 squares, received ${fileCount}]` +
          ` in rank: ${i + 1}`,
      };
    }
  }

  const hands = handpieces.split('/');
  if (hands.length !== 2) {
    return {
      ok: false,
      error: `2nd field (hand pieces) is invalid [expected 2 hands, received ${hands.length}]`,
    };
  }
  const [whiteHand, blackHand] = hands as [string, string];
  if ((whiteHand.length === 1 && whiteHand !== '-') || (blackHand.length === 1 && blackHand !== '-')) {
    return { ok: false, error: '2nd field (hand pieces) is invalid [invalid piece]' };
  }
  if ((whiteHand.length > 1 && whiteHand.length % 2 !== 0) || (blackHand.length > 1 && blackHand.length % 2 !== 0)) {
    return { ok: false, error: '2nd field (hand pieces) is invalid [invalid piece count]' };
  }
  for (const [hand, letters] of [[whiteHand, WHITE_HAND_LETTERS], [blackHand, BLACK_HAND_LETTERS]] as [
    string,
    string,
  ][]) {
    if (hand.length > 1) {
      for (let j = 0; j < hand.length; j += 2) {
        if (!pyIn(hand[j]!, letters)) {
          return { ok: false, error: '2nd field (hand pieces) is invalid [invalid piece]' };
        }
        if (!isDigit(hand[j + 1])) {
          return { ok: false, error: '2nd field (hand pieces) is invalid [invalid count]' };
        }
      }
    }
  }

  if (turn !== 'w' && turn !== 'b') {
    return {
      ok: false,
      error: `3rd field (active player) is invalid [expected 'w' or 'b', received ${turn}]`,
    };
  }
  if (mode.length !== 1 || !pyIn(mode, '0123')) {
    return {
      ok: false,
      error: `4th field (setup mode) is invalid [expected '0', '1', '2', or '3', received ${mode}]`,
    };
  }
  // Python's `drafting not in "wb"` also admits the empty string.
  if (drafting !== '-' && !pyIn(drafting, 'wb')) {
    return {
      ok: false,
      error:
        "5th field (drafting availability) is invalid [expected '-', 'w', 'b', or 'wb', " +
        `received ${drafting}]`,
    };
  }
  if (!PY_INT.test(moveNumber)) {
    return {
      ok: false,
      error: `6th field (full move number) is invalid [expected an integer, received ${moveNumber}]`,
    };
  }
  return { ok: true };
}

// utils.create_piece_from_fen_code: lowercase is black; an unknown code raises.
function pieceFromCode(code: string): FenPiece {
  const ti = FEN_LETTER_TO_INDEX[code.toLowerCase()];
  if (ti === undefined || code.length !== 1) throw new Error(`Invalid FEN piece code: ${code}`);
  return { type: ti, color: code === code.toLowerCase() ? 1 : 0 };
}

export function parseFen(fen: string): ParsedFen {
  const result = validateFen(fen);
  if (!result.ok) throw new Error(`Invalid FEN: ${result.error}`);
  const [placement, handpieces, turn, modeStr, drafting, moveNumber] = fen.split(' ') as [
    string,
    string,
    string,
    string,
    string,
    string,
  ];

  const board: FenTower[][] = [];
  for (const rankStr of placement.split('/')) {
    const row: FenTower[] = [];
    for (const partial of rankStr.split('|')) {
      if (partial.includes(':')) {
        row.push(partial.split(':').map(pieceFromCode));
      } else {
        for (const ch of partial) {
          if (isDigit(ch)) {
            for (let k = Number(ch); k > 0; k--) row.push([null]);
          } else {
            row.push([pieceFromCode(ch)]);
          }
        }
      }
    }
    board.push(row);
  }

  const hand: FenHandPiece[] = [];
  for (const playerHand of handpieces.split('/')) {
    if (playerHand === '-') continue;
    for (let j = 0; j < playerHand.length; j += 2) {
      const p = pieceFromCode(playerHand[j]!);
      hand.push({ type: p.type, color: p.color, count: Number(playerHand[j + 1]) });
    }
  }

  return {
    board,
    hand,
    turn,
    mode: Number(modeStr),
    drafting: { w: drafting.includes('w'), b: drafting.includes('b') },
    moveNumber: pyInt(moveNumber),
  };
}

function letterFor(ti: number, color: number): string {
  const l = FEN_LETTERS[ti]!;
  return color === 0 ? l.toUpperCase() : l;
}

export function encodeFen(parsed: ParsedFen): string {
  let placement = '';
  let empty = 0;
  for (let y = 0; y < 9; y++) {
    for (let x = 0; x < 9; x++) {
      const square = parsed.board[y]![x]!;
      if (!square[0]) {
        empty += 1;
        continue;
      }
      if (empty) {
        placement += String(empty);
        empty = 0;
      }
      if (square.length > 1) {
        const parts: string[] = [];
        for (const p of square) if (p !== null) parts.push(letterFor(p.type, p.color));
        placement += '|' + parts.join(':') + '|';
      } else {
        placement += letterFor(square[0].type, square[0].color);
      }
    }
    placement += empty ? `${empty}/` : '/';
    empty = 0;
  }
  placement = placement.slice(0, -1);

  const white = parsed.hand.filter((p) => p.color === 0);
  const black = parsed.hand.filter((p) => p.color === 1);
  let handStr = '';
  for (const p of white) handStr += `${letterFor(p.type, p.color)}${p.count}`;
  handStr += white.length ? '/' : '-/';
  for (const p of black) handStr += `${letterFor(p.type, p.color)}${p.count}`;
  if (!black.length) handStr += '-';

  let draftStr = (parsed.drafting.w ? 'w' : '') + (parsed.drafting.b ? 'b' : '');
  if (!draftStr) draftStr = '-';

  return `${placement} ${handStr} ${parsed.turn} ${parsed.mode} ${draftStr} ${parsed.moveNumber}`;
}

// The server's canonical_fen = Gungi(x).fen(): keeps the hand order, throws on a malformed FEN.
export function canonicalFen(fen: string): string {
  return encodeFen(parseFen(fen));
}

export function boardFromFen(fen: string): Board {
  const parsed = parseFen(fen);
  const b = new Board();
  for (let r = 0; r < 9; r++) {
    const row = parsed.board[r]!;
    for (let j = 0; j < 9; j++) {
      const f = 8 - j;
      const tower = row[j]!;
      if (tower[0] === null) continue;
      for (let t = 0; t < tower.length; t++) {
        const p = tower[t]!;
        if (p === null) continue;
        b.cells[(r * 9 + f) * 3 + t] = makeCode(p.type, p.color);
      }
    }
  }
  for (const hp of parsed.hand) {
    b.hand[hp.type * 2 + hp.color] = hp.count;
    b.pushHandOrder(hp.type * 2 + hp.color);
  }
  b.turn = parsed.turn === 'w' ? 0 : 1;
  b.mode = parsed.mode;
  b.draftW = parsed.drafting.w ? 1 : 0;
  b.draftB = parsed.drafting.b ? 1 : 0;
  b.moveNumber = parsed.moveNumber;
  return b;
}

export function boardToFen(b: Board): string {
  const ranks: string[] = [];
  for (let r = 0; r < 9; r++) {
    let rankStr = '';
    let empty = 0;
    for (let j = 0; j < 9; j++) {
      const f = 8 - j;
      const h = b.height(r, f);
      if (h === 0) {
        empty += 1;
        continue;
      }
      if (empty > 0) {
        rankStr += String(empty);
        empty = 0;
      }
      const base = (r * 9 + f) * 3;
      if (h === 1) {
        const code = b.cells[base]!;
        rankStr += letterFor(typeIndex(code), colorOf(code));
      } else {
        const parts: string[] = [];
        for (let t = 0; t < h; t++) {
          const code = b.cells[base + t]!;
          parts.push(letterFor(typeIndex(code), colorOf(code)));
        }
        rankStr += '|' + parts.join(':') + '|';
      }
    }
    if (empty > 0) rankStr += String(empty);
    ranks.push(rankStr);
  }

  // Hand order as listed in the source FEN; type-index order for a board built by hand.
  let whiteHand = '';
  let blackHand = '';
  if (b.handOrderLen > 0) {
    for (let k = 0; k < b.handOrderLen; k++) {
      const e = b.handOrder[k]!;
      const cnt = b.hand[e]!;
      if (cnt <= 0) continue;
      if ((e & 1) === 0) whiteHand += letterFor(e >> 1, 0) + cnt;
      else blackHand += letterFor(e >> 1, 1) + cnt;
    }
  } else {
    for (let ti = 0; ti < 14; ti++) {
      const cw = b.hand[ti * 2]!;
      const cb = b.hand[ti * 2 + 1]!;
      if (cw > 0) whiteHand += letterFor(ti, 0) + cw;
      if (cb > 0) blackHand += letterFor(ti, 1) + cb;
    }
  }
  const handStr = (whiteHand || '-') + '/' + (blackHand || '-');
  let draftStr = (b.draftW ? 'w' : '') + (b.draftB ? 'b' : '');
  if (!draftStr) draftStr = '-';
  return `${ranks.join('/')} ${handStr} ${b.turn === 0 ? 'w' : 'b'} ${b.mode} ${draftStr} ${b.moveNumber}`;
}
