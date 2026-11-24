// Typed board, the state cyboard.Board holds (src/komugi/engine/cyboard.pyx:112-156).

import { HAND_SIZE, NUM_CELLS } from './constants.ts';

export function makeCode(ti: number, color: number): number {
  return ti * 2 + color + 1;
}

export function typeIndex(code: number): number {
  return (code - 1) >> 1;
}

export function colorOf(code: number): number {
  return (code - 1) & 1;
}

export function cellIndex(r: number, f: number, t: number): number {
  return (r * 9 + f) * 3 + t;
}

export class Board {
  // cells[(r*9 + f)*3 + t]; a tower of height h fills t = 0..h-1.
  cells: Int8Array = new Int8Array(NUM_CELLS);
  // hand[ti*2 + color] = count.
  hand: Uint8Array = new Uint8Array(HAND_SIZE);
  turn = 0;
  mode = 0;
  draftW = 0;
  draftB = 0;
  moveNumber = 1;
  // Hand entries (ti*2 + color) in the order the FEN listed them, duplicates
  // included (cyboard._hand_order). Grows past 28 only for a FEN that repeats
  // an entry.
  handOrder: Uint8Array = new Uint8Array(HAND_SIZE);
  handOrderLen = 0;

  top(r: number, f: number): number {
    const base = (r * 9 + f) * 3;
    for (let t = 2; t >= 0; t--) {
      const c = this.cells[base + t]!;
      if (c !== 0) return c;
    }
    return 0;
  }

  height(r: number, f: number): number {
    const base = (r * 9 + f) * 3;
    for (let t = 2; t >= 0; t--) {
      if (this.cells[base + t] !== 0) return t + 1;
    }
    return 0;
  }

  pushHandOrder(entry: number): void {
    if (this.handOrderLen === this.handOrder.length) {
      const grown = new Uint8Array(this.handOrder.length * 2);
      grown.set(this.handOrder);
      this.handOrder = grown;
    }
    this.handOrder[this.handOrderLen++] = entry;
  }

  copy(): Board {
    const b = new Board();
    b.cells.set(this.cells);
    b.hand.set(this.hand);
    b.turn = this.turn;
    b.mode = this.mode;
    b.draftW = this.draftW;
    b.draftB = this.draftB;
    b.moveNumber = this.moveNumber;
    b.handOrder = this.handOrder.slice();
    b.handOrderLen = this.handOrderLen;
    return b;
  }
}
