// Move application: a faithful port of cyengine.apply_move
// (src/komugi/engine/cyengine.pyx:179-292), consuming the Int32Array(16)
// struct emitted by movegen_board.ts / movegen_arata.ts. Mutates the Board in
// place; captured-piece lists are read off the board before any mutation,
// via the same mask decode apply.ts shares with san.ts (actions.ts).

import { Board, cellIndex, colorOf, makeCode, typeIndex } from './board.ts';
import { MT_ARATA, MT_BETRAY, MT_CAPTURE, MT_ROUTE, MT_TSUKE } from './constants.ts';
import { capturedFromMask } from './actions.ts';

// Port of cyengine._remove_top (cyengine.pyx:87-91).
function removeTop(board: Board, r: number, f: number): void {
  const h = board.height(r, f);
  if (h > 0) board.cells[cellIndex(r, f, h - 1)] = 0;
}

// Port of cyengine._put (cyengine.pyx:94-97): places on top of the current
// tower height. The move's own to_tier field is not consulted here, exactly
// as in apply_move (which declares it but never reads it).
function put(board: Board, r: number, f: number, code: number): void {
  const h = board.height(r, f);
  board.cells[cellIndex(r, f, h)] = code;
}

// Port of cyengine._remove_captured (cyengine.pyx:100-140): drop the matching
// enemies and rebuild the tower from the surviving pieces, bottom to top.
function removeCaptured(
  board: Board,
  r: number,
  f: number,
  captured: ReadonlyArray<readonly [number, number]>,
  moverColor: number,
): void {
  const h = board.height(r, f);
  if (h === 0) return;
  const toRemove = new Set<number>(captured.map(([ti, tier]) => ti * 8 + tier));
  const survivors: number[] = [];
  for (let t = 0; t < h; t++) {
    const code = board.cells[cellIndex(r, f, t)]!;
    if (code === 0) continue;
    const key = typeIndex(code) * 8 + (t + 1);
    if (colorOf(code) !== moverColor && toRemove.has(key)) {
      toRemove.delete(key);
      continue;
    }
    survivors.push(code);
  }
  for (let t = 0; t < 3; t++) board.cells[cellIndex(r, f, t)] = 0;
  for (let t = 0; t < survivors.length; t++) board.cells[cellIndex(r, f, t)] = survivors[t]!;
}

// Port of cyengine._convert_captured (cyengine.pyx:143-170): flip the color
// of the matching enemies in place. `col !== moverColor` already holds at the
// match, so the new color is always moverColor (the two colors are binary).
function convertCaptured(
  board: Board,
  r: number,
  f: number,
  captured: ReadonlyArray<readonly [number, number]>,
  moverColor: number,
): void {
  const h = board.height(r, f);
  if (h === 0) return;
  const toConvert = new Set<number>(captured.map(([ti, tier]) => ti * 8 + tier));
  for (let t = 0; t < h; t++) {
    const code = board.cells[cellIndex(r, f, t)]!;
    if (code === 0) continue;
    const ti = typeIndex(code);
    const key = ti * 8 + (t + 1);
    if (colorOf(code) !== moverColor && toConvert.has(key)) {
      board.cells[cellIndex(r, f, t)] = makeCode(ti, moverColor);
    }
  }
}

// Port of cyengine.apply_move (cyengine.pyx:179-292).
export function applyMove(board: Board, struct: Int32Array): void {
  const moveType = struct[0]!;
  const pieceTi = struct[1]!;
  const pieceColor = struct[2]!;
  const fromR = struct[3]!;
  const fromF = struct[4]!;
  const toR = struct[6]!;
  const toF = struct[7]!;
  const draftFinished = struct[9] !== 0;
  const mask = struct[10]!;
  const pieceCode = makeCode(pieceTi, pieceColor);

  if (moveType === MT_ROUTE || moveType === MT_TSUKE) {
    removeTop(board, fromR, fromF);
  } else if (moveType === MT_CAPTURE) {
    const captured = capturedFromMask(board, toR, toF, mask);
    removeTop(board, fromR, fromF);
    removeCaptured(board, toR, toF, captured, pieceColor);
  } else if (moveType === MT_BETRAY) {
    const captured = capturedFromMask(board, toR, toF, mask);
    removeTop(board, fromR, fromF);
    convertCaptured(board, toR, toF, captured, pieceColor);
    for (const [ti] of captured) board.hand[ti * 2 + pieceColor]! -= 1;
  } else if (moveType === MT_ARATA) {
    board.hand[pieceTi * 2 + pieceColor]! -= 1;
    if (mask !== 0) {
      const captured = capturedFromMask(board, toR, toF, mask);
      convertCaptured(board, toR, toF, captured, pieceColor);
      for (const [ti] of captured) board.hand[ti * 2 + pieceColor]! -= 1;
    }
    if (draftFinished) {
      if (pieceColor === 0) {
        board.draftW = 0;
      } else {
        board.draftB = 0;
        board.draftW = 0;
      }
    }
  }

  put(board, toR, toF, pieceCode);

  // moveNumber: pre-flip turn, post-draft-update draft bits (cyengine.pyx:257-273).
  const turn = board.turn;
  if (turn === 1 && (board.draftW !== 0 || board.draftB === 0)) board.moveNumber += 1;
  if (turn === 0 && board.draftB === 0 && board.draftW === 0 && draftFinished) board.moveNumber += 1;

  // Turn flip (cyengine.pyx:275-292).
  if (board.draftW === board.draftB) {
    if (!draftFinished || turn !== 0) board.turn = 1 - turn;
  } else if (board.draftB === 0 && turn === 1) {
    board.turn = 0;
  } else if (board.draftW === 0 && turn === 0) {
    board.turn = 1;
  }
}
