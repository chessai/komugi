// KOM-1139: ligungi-only rule (1), the purpose-built self-exposure filter
// (TS port of the training engine's cymovegen.pyx rule-(1) filter, KOM-1128).
// CLEAN ROOM: rule text and the lemma structure come from the ticket's /
// design doc's prose only, never from ligungi source.
//
// Every call site below is reached only behind an `if (rules === 'ligungi')`
// branch at the call site (same convention as rules_ligungi.ts's rule (2)/(3)
// helpers) -- komugi/gungijs never call into this module.
//
// This module deliberately duplicates the DIRS/PROBE_PVAL/PROBE_PCARRY attack
// tables and the reach-walk already present in movegen_board.ts and
// terminal.ts (each of those files carries the same duplication, with the
// same rationale in their own header comments: no cimport-equivalent exists
// for a hot-path internal, and importing forEachBoardMove/forEachArataMove
// here (as terminal.ts does) would create a import cycle, since
// movegen_board.ts/movegen_arata.ts need to call INTO this module to apply
// the filter during generation. This module is therefore a leaf: it only
// imports board.ts, constants.ts and rules_ligungi.ts (plus repetition.ts's
// RulesProfile as a type-only import, erased at compile time).
//
// Algorithm (design doc section 4):
//   - P0: every marshal on the board must be the top of its own tower
//     (nothing tsukes onto a marshal, no drop lands on one); violated only by
//     a synthetic/FEN-built position, never a reachable one. Checked once per
//     buildFilterContext call (same scan that locates the mover's own marshal
//     and the opponent's tops costs nothing extra to also check this).
//   - Geometric prefilter (geometricAdmit): for each opponent top, is there a
//     probe direction whose ray could reach the mover's marshal at all
//     (pure arithmetic, no board read)? At most one direction ever admits
//     (L1): the probe rows are pairwise disjoint.
//   - walkForReadset: the SAME board-aware ray walk movegen_board.ts /
//     terminal.ts already use (reverse-probe wing check, ligungi archer-elbow
//     blocking), but also recording every square the walk reads (p's own
//     square, the wing square, every reverse-probe square, every ligungi
//     elbow square, every forward square up to the target or the first
//     blocker inclusive) into a caller-supplied readset. A blocked candidate
//     still contributes its readset (L1's certificate: unless a move touches
//     one of these squares, the walk's result cannot change).
//   - Per move: L6 fast-accept when not in check and neither the origin nor
//     (for a capture) the destination touches any candidate's readset, and
//     no buried enemy piece is revealed at the origin; L5 fast-reject when in
//     check and some checker's readset excludes the destination; otherwise
//     EXACT -- apply the move to a cells-only scratch board and re-check.
//     EXACT is always correct (it IS the naive oracle), so every fast path
//     above is a pure optimization: getting it wrong would only cost
//     performance, not correctness, EXCEPT the fast-reject/fast-accept
//     booleans themselves, which must only fire when the lemmas guarantee
//     the answer.

import { Board, cellIndex, colorOf, typeIndex } from './board.ts';
import {
  ARCHER,
  CANNON,
  MARSHAL,
  MT_ARATA,
  MT_BETRAY,
  MT_CAPTURE,
  MT_ROUTE,
  MT_TSUKE,
  MUSKETEER,
} from './constants.ts';
import { archerElbowBlocked } from './rules_ligungi.ts';
import { toRulesConfig } from './rules_config.ts';
import type { RulesConfig, RulesProfile } from './rules_config.ts';

// Direction table and probe tables: verbatim copies of movegen_board.ts's /
// terminal.ts's own (cymovegen.pyx:104-148 / cyengine.pyx:325-338).
const DIRS: ReadonlyArray<readonly [number, number]> = [
  [-1, 1], [-1, 0], [-1, -1], [0, 1], [0, -1], [1, 1], [1, 0], [1, -1],
];

const PROBE_PVAL: readonly number[][] = [
  [1, 1, 1, 1, 1, 1, 1, 1],
  [1, 99, 1, 99, 99, 1, 99, 1],
  [99, 1, 99, 1, 1, 99, 1, 99],
  [1, 1, 1, 1, 1, 0, 1, 0],
  [1, 1, 1, 0, 0, 0, 1, 0],
  [1, 1, 1, 0, 0, 0, 1, 0],
  [0, 1, 0, 1, 1, 0, 1, 0],
  [1, 0, 1, 0, 0, 1, 0, 1],
  [0, 1, 0, 1, 1, 1, 0, 1],
  [0, 1, 0, 0, 0, 0, 1, 0],
  [0, 3, 0, 1, 1, 0, 1, 0],
  [2, 2, 2, 0, 0, 0, 1, 0],
  [0, 2, 0, 0, 0, 1, 0, 1],
  [1, 0, 1, 0, 0, 0, 1, 0],
];

const PROBE_PCARRY: readonly number[][] = [
  [1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1],
  [1, 2, 1, 1, 1, 1, 1, 1],
  [1, 2, 1, 1, 1, 1, 2, 1],
  [2, 1, 2, 1, 1, 2, 1, 2],
  [1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1],
];

function topAt(board: Board, rank: number, file: number): number {
  return board.top(rank - 1, file - 1);
}

function heightAt(board: Board, rank: number, file: number): number {
  return board.height(rank - 1, file - 1);
}

// 1-based (rank, file) -> a square id in [0, 80], for readset membership.
function sqId(rank: number, file: number): number {
  return (rank - 1) * 9 + (file - 1);
}

// Pure-arithmetic admissibility test (design doc 4.2 step 2): is there a
// probe direction of (pcTi, pcColor, pcTier) at (py, px) whose ray could,
// geometrically, pass through (kr, kf)? Independent of board content. At
// most one direction ever admits (L1: the probe rows are pairwise disjoint).
function stepsToTarget(sy: number, sx: number, dy: number, dx: number, kr: number, kf: number, maxSteps: number): number {
  // `null` (not -1) is the "not yet computed" sentinel: a real step count of
  // -1 (reject, below) must never be confused with "unset" (KOM-1139 review
  // caught this exact collision -- a non-slider's dy-axis step can
  // legitimately compute to -1, which a -1 sentinel would silently treat as
  // unset and let the dx-axis branch overwrite with an unrelated value).
  let t: number | null = null;
  if (dy !== 0) {
    const diff = kr - sy;
    if (diff % dy !== 0) return -1;
    t = diff / dy;
  } else if (kr !== sy) {
    return -1;
  }
  if (dx !== 0) {
    const diff = kf - sx;
    if (diff % dx !== 0) return -1;
    const tx = diff / dx;
    if (t === null) t = tx;
    else if (t !== tx) return -1;
  } else if (kf !== sx) {
    return -1;
  }
  if (t === null || t < 0 || t > maxSteps) return -1;
  return t;
}

type Ray = { dy: number; dx: number; sy: number; sx: number; length: number };

function geometricAdmit(
  pcTi: number, pcColor: number, pcTier: number, py: number, px: number, kr: number, kf: number,
): Ray | null {
  for (let i = 0; i < 8; i++) {
    const pval = PROBE_PVAL[pcTi]![i]!;
    const pcarry = PROBE_PCARRY[pcTi]![i]!;
    if (pval < 1) continue;

    let dy = DIRS[i]![0];
    let dx = DIRS[i]![1];
    if (pcColor === 1) {
      dy = -dy;
      dx = -dx;
    }

    let sy: number;
    let sx: number;
    let length: number;
    if (pval === 99) {
      sy = py + dy;
      sx = px + dx;
      length = 99;
    } else {
      sy = py + pval * dy;
      sx = px + dx;
      length = pcTier + pcarry - 1;
    }

    const maxSteps = length === 99 ? 16 : length - 1;
    if (stepsToTarget(sy, sx, dy, dx, kr, kf, maxSteps) !== -1) {
      return { dy, dx, sy, sx, length };
    }
  }
  return null;
}

// Board-aware ray walk (faithful twin of movegen_board.ts's
// getAvailableSquares / terminal.ts's squareInAvailableSquares), but also
// recording every square it reads into `readset` (the "readset definition",
// design doc 4.1): p's own square, the wing square, every reverse-probe
// square, every ligungi elbow square, and every forward square visited up to
// the target or the first blocker inclusive.
function walkForReadset(
  board: Board,
  dy: number,
  dx: number,
  sy: number,
  sx: number,
  py: number,
  px: number,
  originTi: number,
  originColor: number,
  originTier: number,
  length: number,
  targetRank: number,
  targetFile: number,
  config: RulesConfig,
  readset: Set<number>,
): boolean {
  readset.add(sqId(py, px));

  if (originTi === ARCHER && Math.abs(dy) === 1 && Math.abs(dx) === 1) {
    const wingRank = py + dy;
    const wingFile = px + dx;
    if (wingRank >= 1 && wingRank <= 9 && wingFile >= 1 && wingFile <= 9) {
      readset.add(sqId(wingRank, wingFile));
      const wingCode = topAt(board, wingRank, wingFile);
      if (wingCode !== 0) {
        const wingTier = heightAt(board, wingRank, wingFile);
        if (wingTier > originTier) return false;
      }
    }
  }

  let rx = sx;
  let ry = sy;
  while (rx !== px || ry !== py) {
    rx -= dx;
    ry -= dy;
    if (!(ry >= 1 && ry <= 9 && rx >= 1 && rx <= 9)) break;
    readset.add(sqId(ry, rx));
    const code = topAt(board, ry, rx);
    if (code !== 0) {
      const tier = heightAt(board, ry, rx);
      if (tier > originTier) return false;
    }
    const side = originColor === 1 ? -1 : 1;
    if (ry + side >= 1 && ry + side <= 9 && rx >= 1 && rx <= 9) {
      if (ry + side === py && rx === px) break;
    }
  }

  let fx = sx;
  let fy = sy;
  let step = 0;
  while (step < length) {
    if (fx < 1 || fx > 9 || fy < 1 || fy > 9) break;
    if (config.archerElbowBlocking && step >= 1 && originTi === ARCHER && Math.abs(dy) === 1 && Math.abs(dx) === 1) {
      const k = step + 1;
      const elbowRank = py + k * dy;
      const elbowFile = px + k * dx;
      if (elbowRank >= 1 && elbowRank <= 9 && elbowFile >= 1 && elbowFile <= 9) {
        readset.add(sqId(elbowRank, elbowFile));
      }
      if (archerElbowBlocked(board, py, px, dy, dx, step, originTier)) break;
    }
    readset.add(sqId(fy, fx));
    const code = topAt(board, fy, fx);
    if (code !== 0) {
      const tier = heightAt(board, fy, fx);
      if (tier > originTier) break;
    }
    if (fy === targetRank && fx === targetFile) return true;
    if (code !== 0) {
      if (originTi !== CANNON && originTi !== MUSKETEER && originTi !== ARCHER) break;
      const movingForward = originColor === 0 ? dy < 0 : dy > 0;
      if (!movingForward) break;
    }
    fx += dx;
    fy += dy;
    step += 1;
  }

  return false;
}

type Checker = { readset: Set<number> };

// null = no filter needed (mover has no marshal on the board -- a
// marshal-captured position, which is already terminal: design doc 4.2
// step 1, "apply no filter").
export type FilterContext = { checkers: Checker[]; rAll: Set<number> } | null;

// Context, built once per fill / has_legal_move / generate call (design doc
// 4.2): locates the mover's own marshal (P0: must be the top of its tower,
// else throw -- the SAME scan also checks the opponent's marshal, since it
// costs nothing extra), collects the opponent's tops, geometrically
// prefilters them (P_K), and walks each admitted candidate for its
// reach/readset. `checkers` is the subset that currently reaches the mover's
// marshal (in_check iff non-empty); `rAll` is the union of every P_K
// candidate's readset (reaching or not), used by the not-in-check fast
// accept (L6).
export function buildFilterContext(board: Board, mover: number, rules: RulesProfile | RulesConfig): FilterContext {
  const config = toRulesConfig(rules);
  const opp = 1 - mover;
  let moverMarshal: { r: number; f: number } | null = null;
  const oppTops: Array<{ r: number; f: number; ti: number; tier: number }> = [];
  // KOM-1148 F2: P0 also requires at most one top marshal per colour (a
  // duplicate is never reachable from real play, only from a synthetic/FEN
  // position -- mirrors cymovegen's _check_p0 dup_marshal check).
  const marshalTopCount = [0, 0];

  for (let r = 0; r < 9; r++) {
    for (let f = 0; f < 9; f++) {
      const h = board.height(r, f);
      for (let t = 0; t < h; t++) {
        const code = board.cells[cellIndex(r, f, t)]!;
        if (code === 0) continue;
        if (typeIndex(code) === MARSHAL) {
          const isTop = t === h - 1;
          if (!isTop) {
            throw new Error(`buried marshal: color=${colorOf(code)} at r=${r + 1} f=${f + 1} tier=${t + 1}`);
          }
          marshalTopCount[colorOf(code)]!++;
          if (colorOf(code) === mover) moverMarshal = { r: r + 1, f: f + 1 };
        }
      }
      if (h > 0) {
        const topCode = board.cells[cellIndex(r, f, h - 1)]!;
        if (colorOf(topCode) === opp) {
          oppTops.push({ r: r + 1, f: f + 1, ti: typeIndex(topCode), tier: h });
        }
      }
    }
  }

  if (marshalTopCount[0]! > 1 || marshalTopCount[1]! > 1) {
    throw new Error(`duplicate top marshal: white=${marshalTopCount[0]} black=${marshalTopCount[1]}`);
  }
  // KOM-1148 F2: a marshal still in hand once the draft is over is likewise
  // only reachable from a synthetic position (marshals are never recaptured
  // into hand, and the draft always places both before finishing).
  if (board.draftW === 0 && board.draftB === 0) {
    if (board.hand[MARSHAL * 2]! > 0 || board.hand[MARSHAL * 2 + 1]! > 0) {
      throw new Error('marshal in hand outside the draft');
    }
  }

  if (moverMarshal === null) return null;
  const kr = moverMarshal.r;
  const kf = moverMarshal.f;

  const checkers: Checker[] = [];
  const rAll = new Set<number>();
  for (const p of oppTops) {
    const admit = geometricAdmit(p.ti, opp, p.tier, p.r, p.f, kr, kf);
    if (admit === null) continue;
    const readset = new Set<number>();
    const reaches = walkForReadset(
      board, admit.dy, admit.dx, admit.sy, admit.sx, p.r, p.f, p.ti, opp, p.tier, admit.length, kr, kf, config, readset,
    );
    for (const sq of readset) rAll.add(sq);
    if (reaches) checkers.push({ readset });
  }
  return { checkers, rAll };
}

// EXACT fallback: apply the move's effect on cells only (hand/turn/draft/
// moveNumber never affect check) to a cells-only scratch board, then
// re-check. Always correct -- the naive oracle itself, just reused as the
// path the fast accept/reject rules above could not resolve. The marshal-
// capture exception is handled by the caller before ever reaching here (a
// move that captures the opponent's marshal is legal unconditionally), so a
// move that reaches EXACT never removes the opponent's marshal.
const scratchBoard = new Board();

function applyCellsOnly(cells: Int8Array, struct: Int32Array): void {
  const moveType = struct[0];
  const pieceTi = struct[1]!;
  const pieceColor = struct[2]!;
  const fromR = struct[3]!;
  const fromF = struct[4]!;
  const toR = struct[6]!;
  const toF = struct[7]!;
  const mask = struct[10]!;
  const pieceCode = pieceTi * 2 + pieceColor + 1;

  function h(r: number, f: number): number {
    for (let t = 2; t >= 0; t--) if (cells[(r * 9 + f) * 3 + t] !== 0) return t + 1;
    return 0;
  }
  function removeTop(r: number, f: number): void {
    const height = h(r, f);
    if (height > 0) cells[(r * 9 + f) * 3 + height - 1] = 0;
  }
  function removeCaptured(r: number, f: number, m: number): void {
    const height = h(r, f);
    const survivors: number[] = [];
    for (let t = 0; t < height; t++) {
      if (m & (1 << t)) continue;
      const code = cells[(r * 9 + f) * 3 + t]!;
      if (code !== 0) survivors.push(code);
    }
    for (let t = 0; t < 3; t++) cells[(r * 9 + f) * 3 + t] = 0;
    for (let t = 0; t < survivors.length; t++) cells[(r * 9 + f) * 3 + t] = survivors[t]!;
  }
  function convertCaptured(r: number, f: number, m: number, color: number): void {
    const height = h(r, f);
    for (let t = 0; t < height; t++) {
      if (!(m & (1 << t))) continue;
      const code = cells[(r * 9 + f) * 3 + t]!;
      if (code === 0) continue;
      const ti = (code - 1) >> 1;
      cells[(r * 9 + f) * 3 + t] = ti * 2 + color + 1;
    }
  }
  function put(r: number, f: number, code: number): void {
    const height = h(r, f);
    cells[(r * 9 + f) * 3 + height] = code;
  }

  if (moveType === MT_ROUTE || moveType === MT_TSUKE) {
    removeTop(fromR, fromF);
  } else if (moveType === MT_CAPTURE) {
    removeTop(fromR, fromF);
    removeCaptured(toR, toF, mask);
  } else if (moveType === MT_BETRAY) {
    removeTop(fromR, fromF);
    convertCaptured(toR, toF, mask, pieceColor);
  } else if (moveType === MT_ARATA) {
    if (mask !== 0) convertCaptured(toR, toF, mask, pieceColor);
  }
  put(toR, toF, pieceCode);
}

function isAttackedLocal(board: Board, targetRank: number, targetFile: number, byColor: number, config: RulesConfig): boolean {
  for (let r = 0; r < 9; r++) {
    for (let f = 0; f < 9; f++) {
      const top = board.top(r, f);
      if (top === 0 || colorOf(top) !== byColor) continue;
      const tier = board.height(r, f);
      const ti = typeIndex(top);
      const admit = geometricAdmit(ti, byColor, tier, r + 1, f + 1, targetRank, targetFile);
      if (admit === null) continue;
      const discard = new Set<number>();
      if (
        walkForReadset(board, admit.dy, admit.dx, admit.sy, admit.sx, r + 1, f + 1, ti, byColor, tier, admit.length, targetRank, targetFile, config, discard)
      ) {
        return true;
      }
    }
  }
  return false;
}

function localInCheck(board: Board, color: number, config: RulesConfig): boolean {
  const opp = 1 - color;
  for (let r = 0; r < 9; r++) {
    for (let f = 0; f < 9; f++) {
      const top = board.top(r, f);
      if (top !== 0 && typeIndex(top) === MARSHAL && colorOf(top) === color) {
        return isAttackedLocal(board, r + 1, f + 1, opp, config);
      }
    }
  }
  return false;
}

function exactLegal(board: Board, mover: number, config: RulesConfig, struct: Int32Array): boolean {
  scratchBoard.cells.set(board.cells);
  applyCellsOnly(scratchBoard.cells, struct);
  return !localInCheck(scratchBoard, mover, config);
}

// KOM-1148 F1 (test-only): with this off, isMoveLegalUnderRule1 always
// accepts, i.e. rules (2)/(3) movegen with no rule-(1) filtering at all --
// the differential tests' pseudo-ligungi oracle input. buildFilterContext's
// own P0 checks (buried/duplicate marshal, marshal in hand) are unaffected.
// Defaults to enabled (real behaviour); only ever flipped by test code.
let rule1FilterEnabledForTest = true;

export function setRule1FilterEnabledForTest(enabled: boolean): void {
  rule1FilterEnabledForTest = enabled;
}

// Per-move decision (design doc 4.2 "Per emitted move"). `board` must still
// be in the move's PRE-move state (the same precondition every other
// movegen/apply helper in this codebase already carries). `struct` is the
// Int32Array(16) record movegen_board.ts/movegen_arata.ts build, reused
// as-is (its captured tier-mask IS the bitmask applyCellsOnly needs, with no
// intermediate (type, tier) decoding).
export function isMoveLegalUnderRule1(
  ctx: FilterContext, board: Board, mover: number, rules: RulesProfile | RulesConfig, struct: Int32Array,
): boolean {
  const config = toRulesConfig(rules);
  const moveType = struct[0];
  const pieceTi = struct[1]!;
  const fromR = struct[3]!;
  const fromF = struct[4]!;
  const fromTier = struct[5]!;
  const toR = struct[6]!;
  const toF = struct[7]!;

  if (moveType === MT_CAPTURE) {
    const topBefore = board.top(toR, toF);
    if (topBefore !== 0 && typeIndex(topBefore) === MARSHAL) return true;
  }

  if (!rule1FilterEnabledForTest) return true;
  if (ctx === null) return true;
  if (pieceTi === MARSHAL) return exactLegal(board, mover, config, struct);

  const destSq = toR * 9 + toF;

  if (ctx.checkers.length === 0) {
    const originSq = fromR * 9 + fromF;
    if (ctx.rAll.has(originSq)) return exactLegal(board, mover, config, struct);
    if (moveType === MT_CAPTURE && ctx.rAll.has(destSq)) return exactLegal(board, mover, config, struct);
    if (fromTier >= 2) {
      const belowCode = board.cells[cellIndex(fromR, fromF, fromTier - 2)]!;
      if (belowCode !== 0 && colorOf(belowCode) !== mover) return exactLegal(board, mover, config, struct);
    }
    return true;
  }

  for (const c of ctx.checkers) {
    if (!c.readset.has(destSq)) return false;
  }
  return exactLegal(board, mover, config, struct);
}
