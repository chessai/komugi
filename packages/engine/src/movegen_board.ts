// Board-move (A0/A1) mask fill and move-record emitter.
// Port of src/komugi/engine/cymovegen.pyx:
//   probe tables (97-148), _get_available_squares (203-283),
//   _get_betrayal_combos / _generate_combinations (291-363),
//   _qualifying_tier_mask (366-430),
//   _generate_moves_for_square (541-677), _fill_moves_for_square (855-953).
// Arata (hand-drop) moves are U2b's movegen_arata.ts; this file only ever
// touches A0_START/A1_START ids (block A0/A1, ids < B_START).

import { Board, cellIndex, colorOf, typeIndex } from './board.ts';
import {
  A0_START,
  A1_START,
  ARCHER,
  CANNON,
  MARSHAL,
  MT_BETRAY,
  MT_CAPTURE,
  MT_ROUTE,
  MT_TSUKE,
  MUSKETEER,
  TACTICIAN,
} from './constants.ts';
import { toRulesConfig } from './rules_config.ts';
import type { RulesConfig, RulesProfile } from './rules_config.ts';
import { archerElbowBlocked, fullBetrayalCombo } from './rules_ligungi.ts';
import { buildFilterContext, isMoveLegalUnderRule1, type FilterContext } from './ligungi_filter.ts';

// Direction table: [NW, N, NE, E, W, SE, S, SW] as (dy, dx), white's frame.
const DIRS: ReadonlyArray<readonly [number, number]> = [
  [-1, 1], [-1, 0], [-1, -1], [0, 1], [0, -1], [1, 1], [1, 0], [1, -1],
];

// probe_pval[type_index][dir]; 0 = inactive, 99 = slider (cymovegen.pyx:104-148).
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

// probe_pcarry[type_index][dir] (cymovegen.pyx:104-148).
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

// sq_index for 0-based (r, f): matches cymovegen._sq_index(rank, file) with
// rank = r+1, file = f+1 — (rank-1)*9 + (9-file) = r*9 + (8-f).
function sqIndex(r: number, f: number): number {
  return r * 9 + (8 - f);
}

function topAt(board: Board, rank: number, file: number): number {
  return board.top(rank - 1, file - 1);
}

function heightAt(board: Board, rank: number, file: number): number {
  return board.height(rank - 1, file - 1);
}

// Scratch outputs of readTopTier (see visitAvailableSquares): topAt and
// heightAt both scan the same tower top-down to the same stop condition, so
// the hot-path twin reads it once instead of twice. Module-scoped to avoid a
// per-call tuple allocation; read immediately after the call, never across a
// nested call into readTopTier itself.
let scratchTopCode = 0;
let scratchTopTier = 0;

function readTopTier(board: Board, rank: number, file: number): void {
  const base = ((rank - 1) * 9 + (file - 1)) * 3;
  for (let t = 2; t >= 0; t--) {
    const c = board.cells[base + t]!;
    if (c !== 0) {
      scratchTopCode = c;
      scratchTopTier = t + 1;
      return;
    }
  }
  scratchTopCode = 0;
  scratchTopTier = 0;
}

// Faithful port of cymovegen._get_available_squares (cymovegen.pyx:203-283).
// Coordinates are 1-based throughout, matching the Cython source exactly.
function getAvailableSquares(
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
  config: RulesConfig,
): Array<[number, number]> {
  const available: Array<[number, number]> = [];

  if (originTi === ARCHER && Math.abs(dy) === 1 && Math.abs(dx) === 1) {
    const wingRank = py + dy;
    const wingFile = px + dx;
    if (wingRank >= 1 && wingRank <= 9 && wingFile >= 1 && wingFile <= 9) {
      const wingCode = topAt(board, wingRank, wingFile);
      if (wingCode !== 0) {
        const wingTier = heightAt(board, wingRank, wingFile);
        if (wingTier > originTier) return [];
      }
    }
  }

  let rx = sx;
  let ry = sy;
  while (rx !== px || ry !== py) {
    rx -= dx;
    ry -= dy;
    if (!(ry >= 1 && ry <= 9 && rx >= 1 && rx <= 9)) break;
    const code = topAt(board, ry, rx);
    if (code !== 0) {
      const tier = heightAt(board, ry, rx);
      if (tier > originTier) return [];
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
    if (
      config.archerElbowBlocking &&
      step >= 1 &&
      originTi === ARCHER &&
      Math.abs(dy) === 1 &&
      Math.abs(dx) === 1 &&
      archerElbowBlocked(board, py, px, dy, dx, step, originTier)
    ) {
      break;
    }
    const code = topAt(board, fy, fx);
    if (code !== 0) {
      const tier = heightAt(board, fy, fx);
      if (tier > originTier) break;
    }
    available.push([fy, fx]);
    if (code !== 0) {
      if (originTi !== CANNON && originTi !== MUSKETEER && originTi !== ARCHER) break;
      const movingForward = originColor === 0 ? dy < 0 : dy > 0;
      if (!movingForward) break;
    }
    fx += dx;
    fy += dy;
    step += 1;
  }

  return available;
}

// Faithful port of cymovegen._generate_combinations (cymovegen.pyx:291-304):
// all non-empty subsets of items, in recursive pre-order.
function generateCombinations<T>(items: readonly T[]): T[][] {
  const result: T[][] = [];
  const current: T[] = [];
  function helper(start: number): void {
    if (current.length > 0) result.push(current.slice());
    for (let i = start; i < items.length; i++) {
      current.push(items[i]!);
      helper(i + 1);
      current.pop();
    }
  }
  helper(0);
  return result;
}

// Faithful port of cymovegen._get_betrayal_combos (cymovegen.pyx:307-363).
// dropTi is -1 for a board move: the tactician comes off the board, so
// nothing is spent from hand first.
function getBetrayalCombos(
  board: Board,
  rank: number,
  file: number,
  color: number,
  dropTi: number,
): Array<Array<[number, number]>> {
  const h = heightAt(board, rank, file);
  if (h === 0) return [];
  const r = rank - 1;
  const f = file - 1;

  const enemies: Array<[number, number]> = [];
  for (let t = 0; t < h; t++) {
    const code = board.cells[cellIndex(r, f, t)]!;
    if (code === 0) continue;
    const ti = typeIndex(code);
    const col = colorOf(code);
    if (col !== color) enemies.push([ti, t + 1]);
  }
  if (enemies.length === 0) return [];

  const enemyCount = new Map<number, number>();
  for (const [eti] of enemies) enemyCount.set(eti, (enemyCount.get(eti) ?? 0) + 1);

  const betrayalOptions: Array<[number, number]> = [];
  for (const [eti, ecount] of enemyCount) {
    const need = ecount + (eti === dropTi ? 1 : 0);
    if (board.hand[eti * 2 + color]! >= need) {
      for (const e of enemies) if (e[0] === eti) betrayalOptions.push(e);
    }
  }
  if (betrayalOptions.length === 0) return [];

  return generateCombinations(betrayalOptions);
}

// Faithful port of cymovegen._qualifying_tier_mask (cymovegen.pyx:366-430):
// the allocation-free twin of getBetrayalCombos' betrayalOptions, as a
// per-tier bit instead of a list of combinations.
function qualifyingTierMask(
  board: Board,
  rank: number,
  file: number,
  color: number,
  dropTi: number,
): number {
  const r = rank - 1;
  const f = file - 1;
  let mask = 0;
  for (let t = 0; t < 3; t++) {
    const code = board.cells[cellIndex(r, f, t)]!;
    if (code === 0 || colorOf(code) === color) continue;
    const ti = typeIndex(code);
    let count = 0;
    for (let u = 0; u < 3; u++) {
      const code2 = board.cells[cellIndex(r, f, u)]!;
      if (code2 !== 0 && colorOf(code2) !== color && typeIndex(code2) === ti) count++;
    }
    if (ti === dropTi) count++;
    if (board.hand[ti * 2 + color]! >= count) mask |= 1 << t;
  }
  return mask;
}

// Port of cymovegen's per-piece reachable-square scan, the shared first half
// of _generate_moves_for_square / _fill_moves_for_square (cymovegen.pyx:556-588,
// 870-895).
function collectReachable(
  board: Board,
  py: number,
  px: number,
  pcTi: number,
  pcColor: number,
  pcTier: number,
  config: RulesConfig,
): Array<[number, number]> {
  const reachable: Array<[number, number]> = [];
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

    reachable.push(
      ...getAvailableSquares(board, dy, dx, sy, sx, py, px, pcTi, pcColor, pcTier, length, config),
    );
  }
  return reachable;
}

// Allocation-free twin of getAvailableSquares/collectReachable, for
// forEachBoardMove's hot path (Game.winState -> isGameOver -> hasEscapingMove
// runs this on every non-terminal ply, same rationale as terminal.ts's
// squareInAvailableSquares): visits each reachable square via callback
// instead of building per-direction arrays and concatenating them.
function visitAvailableSquares(
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
  config: RulesConfig,
  visit: (ry: number, rx: number, topCode: number, tier: number) => void,
): void {
  if (originTi === ARCHER && Math.abs(dy) === 1 && Math.abs(dx) === 1) {
    const wingRank = py + dy;
    const wingFile = px + dx;
    if (wingRank >= 1 && wingRank <= 9 && wingFile >= 1 && wingFile <= 9) {
      readTopTier(board, wingRank, wingFile);
      if (scratchTopCode !== 0 && scratchTopTier > originTier) return;
    }
  }

  let rx = sx;
  let ry = sy;
  while (rx !== px || ry !== py) {
    rx -= dx;
    ry -= dy;
    if (!(ry >= 1 && ry <= 9 && rx >= 1 && rx <= 9)) break;
    readTopTier(board, ry, rx);
    if (scratchTopCode !== 0) {
      if (scratchTopTier > originTier) return;
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
    if (
      config.archerElbowBlocking &&
      step >= 1 &&
      originTi === ARCHER &&
      Math.abs(dy) === 1 &&
      Math.abs(dx) === 1 &&
      archerElbowBlocked(board, py, px, dy, dx, step, originTier)
    ) {
      break;
    }
    readTopTier(board, fy, fx);
    const code = scratchTopCode;
    const tier = scratchTopTier;
    if (code !== 0) {
      if (tier > originTier) break;
    }
    visit(fy, fx, code, tier);
    if (code !== 0) {
      if (originTi !== CANNON && originTi !== MUSKETEER && originTi !== ARCHER) break;
      const movingForward = originColor === 0 ? dy < 0 : dy > 0;
      if (!movingForward) break;
    }
    fx += dx;
    fy += dy;
    step += 1;
  }
}

// Allocation-free twin of collectReachable (see visitAvailableSquares).
function forEachReachable(
  board: Board,
  py: number,
  px: number,
  pcTi: number,
  pcColor: number,
  pcTier: number,
  config: RulesConfig,
  visit: (ry: number, rx: number, topCode: number, tier: number) => void,
): void {
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

    visitAvailableSquares(board, dy, dx, sy, sx, py, px, pcTi, pcColor, pcTier, length, config, visit);
  }
}

// struct layout (Int32Array(16), reused across calls within one run):
//   [0] move_type  [1] piece (type index)  [2] color
//   [3] from_r     [4] from_f              [5] from_tier
//   [6] to_r       [7] to_f                [8] to_tier
//   [9] draft_finished (0/1, always 0 here)
//   [10] captured tier-mask: bit (tier-1) set iff the piece at that tier of
//        the destination tower is captured/converted by this move
//        (action_encoding.sel_from_captured's mask; the type at each set
//        tier is read off the board, which generation never mutates)
//   [11..15] reserved, always 0
export type BoardMoveCallback = (id: number, struct: Int32Array) => void;

// Module-scoped state for forEachBoardMove's per-square reachable-squares
// visitor (fbmVisit below), same rationale as readTopTier's scratch* fields:
// forEachBoardMove is not reentrant from within its own callback (none of its
// callers -- terminal.ts's hasEscapingMove, movegen.ts's generateRecords,
// actions.ts's decodeAction -- re-enter it while a visit is in flight), so one
// shared struct/closure per module replaces a fresh Int32Array(16) and a fresh
// per-square arrow closure that forEachReachable used to get on every call
// (profiled: the single largest closure-allocation cost on this file's hot
// path, alongside visitAvailableSquares/forEachBoardMove's own self time).
const fbmStruct = new Int32Array(16);
let fbmBoard: Board;
let fbmConfig: RulesConfig = toRulesConfig('komugi');
let fbmMaxTier = 2;
let fbmMarshalCanStack = false;
let fbmPcTi = 0;
let fbmPcColor = 0;
let fbmPcTier = 0;
let fbmR = 0;
let fbmF = 0;
let fbmFromSq = 0;
let fbmTryEmit: (id: number, s: Int32Array) => void;

function fbmVisit(sr: number, sf: number, targetCode: number, targetH: number): void {
  const board = fbmBoard;
  const config = fbmConfig;
  const struct = fbmStruct;
  const pcTi = fbmPcTi;
  const pcColor = fbmPcColor;
  const pcTier = fbmPcTier;
  const r = fbmR;
  const f = fbmF;
  const fromSq = fbmFromSq;
  const maxTier = fbmMaxTier;
  const marshalCanStack = fbmMarshalCanStack;
  const tryEmit = fbmTryEmit;

  const tr = sr - 1;
  const tf = sf - 1;
  const toSq = sqIndex(tr, tf);

  if (targetCode === 0) {
    struct[0] = MT_ROUTE;
    struct[1] = pcTi;
    struct[2] = pcColor;
    struct[3] = r;
    struct[4] = f;
    struct[5] = pcTier;
    struct[6] = tr;
    struct[7] = tf;
    struct[8] = 1;
    struct[9] = 0;
    struct[10] = 0;
    tryEmit(A0_START + fromSq * 81 + toSq, struct);
    return;
  }

  const targetTi = typeIndex(targetCode);
  const targetColor = colorOf(targetCode);
  const targetTier = targetH;

  if (targetTier < maxTier && targetTi !== MARSHAL) {
    if (pcTi !== MARSHAL || marshalCanStack) {
      struct[0] = MT_TSUKE;
      struct[1] = pcTi;
      struct[2] = pcColor;
      struct[3] = r;
      struct[4] = f;
      struct[5] = pcTier;
      struct[6] = tr;
      struct[7] = tf;
      struct[8] = targetTier + 1;
      struct[9] = 0;
      struct[10] = 0;
      tryEmit(A1_START + (fromSq * 4 + 0) * 81 + toSq, struct);

      if (pcTi === TACTICIAN) {
        let enemyCount = 0;
        for (let t = 0; t < targetH; t++) {
          const code2 = board.cells[cellIndex(tr, tf, t)]!;
          if (code2 !== 0 && colorOf(code2) !== pcColor) enemyCount++;
        }
        if (enemyCount > 0) {
          const allCombos = getBetrayalCombos(board, sr, sf, pcColor, -1);
          const combos = config.tacticianBetrayalMode === 'all-or-nothing' ? fullBetrayalCombo(allCombos, enemyCount) : allCombos;
          for (const combo of combos) {
            let sel = 0;
            for (const [, tier] of combo) sel |= 1 << (tier - 1);
            struct[0] = MT_BETRAY;
            struct[1] = pcTi;
            struct[2] = pcColor;
            struct[3] = r;
            struct[4] = f;
            struct[5] = pcTier;
            struct[6] = tr;
            struct[7] = tf;
            struct[8] = targetTier + 1;
            struct[9] = 0;
            struct[10] = sel;
            tryEmit(A1_START + (fromSq * 4 + sel) * 81 + toSq, struct);
          }
        }
      }
    }
  }

  if (targetColor !== pcColor) {
    let newTier = 0;
    for (let t = 0; t < targetH; t++) {
      const code2 = board.cells[cellIndex(tr, tf, t)]!;
      if (code2 !== 0 && colorOf(code2) === pcColor) newTier++;
    }
    newTier += 1;

    let capturedSel = 0;
    for (let t = 0; t < targetH; t++) {
      const code2 = board.cells[cellIndex(tr, tf, t)]!;
      if (code2 !== 0 && colorOf(code2) !== pcColor) capturedSel |= 1 << t;
    }

    struct[0] = MT_CAPTURE;
    struct[1] = pcTi;
    struct[2] = pcColor;
    struct[3] = r;
    struct[4] = f;
    struct[5] = pcTier;
    struct[6] = tr;
    struct[7] = tf;
    struct[8] = newTier;
    struct[9] = 0;
    struct[10] = capturedSel;
    tryEmit(A0_START + fromSq * 81 + toSq, struct);
  }
}

// Port of cymovegen._generate_moves_for_square, over every square in SQUARES
// order (cymovegen.pyx:541-677, 1098-1101). Board moves do not exist during
// the draft (cymovegen.pyx:1090-1106: the draft branch emits arata only).
export function forEachBoardMove(board: Board, cb: BoardMoveCallback, rules: RulesProfile | RulesConfig = 'komugi'): void {
  if (board.draftW || board.draftB) return;
  const config = toRulesConfig(rules);

  const maxTier = board.mode === 3 ? 3 : 2;
  const marshalCanStack = board.mode >= 2;
  // KOM-1139: rule (1) self-exposure filter, context built once per call
  // (design doc 4.2) -- every emission point below routes through tryEmit.
  const filterCtx: FilterContext = config.selfExposureFilter ? buildFilterContext(board, board.turn, config) : null;

  function tryEmit(id: number, s: Int32Array): void {
    if (config.selfExposureFilter && !isMoveLegalUnderRule1(filterCtx, board, board.turn, config, s)) return;
    cb(id, s);
  }

  fbmBoard = board;
  fbmConfig = config;
  fbmMaxTier = maxTier;
  fbmMarshalCanStack = marshalCanStack;
  fbmTryEmit = tryEmit;

  for (let r = 0; r < 9; r++) {
    for (let f = 8; f >= 0; f--) {
      const topCode = board.top(r, f);
      if (topCode === 0) continue;
      const pcTi = typeIndex(topCode);
      const pcColor = colorOf(topCode);
      if (pcColor !== board.turn) continue;
      const pcTier = board.height(r, f);
      const py = r + 1;
      const px = f + 1;
      const fromSq = sqIndex(r, f);

      fbmPcTi = pcTi;
      fbmPcColor = pcColor;
      fbmPcTier = pcTier;
      fbmR = r;
      fbmF = f;
      fbmFromSq = fromSq;

      forEachReachable(board, py, px, pcTi, pcColor, pcTier, config, fbmVisit);
    }
  }
}

// Port of cymovegen._fill_moves_for_square, over every square in SQUARES
// order (cymovegen.pyx:855-953, 1133-1136). Mirrors forEachBoardMove's move
// set exactly, writing mask bits only (no per-move allocation).
export function fillBoardMask(board: Board, mask: Uint8Array, rules: RulesProfile | RulesConfig = 'komugi'): void {
  if (board.draftW || board.draftB) return;
  const config = toRulesConfig(rules);

  const maxTier = board.mode === 3 ? 3 : 2;
  const marshalCanStack = board.mode >= 2;
  // KOM-1139: rule (1) self-exposure filter, context built once per call
  // (design doc 4.2), applied via a scratch struct reused across the loop
  // below (same struct shape forEachBoardMove builds, movegen.ts's layout).
  const filterCtx: FilterContext = config.selfExposureFilter ? buildFilterContext(board, board.turn, config) : null;
  const struct = new Int32Array(16);

  function legal(
    moveType: number, pcTi: number, pcColor: number, r: number, f: number, pcTier: number,
    tr: number, tf: number, sel: number,
  ): boolean {
    if (!config.selfExposureFilter) return true;
    struct[0] = moveType;
    struct[1] = pcTi;
    struct[2] = pcColor;
    struct[3] = r;
    struct[4] = f;
    struct[5] = pcTier;
    struct[6] = tr;
    struct[7] = tf;
    struct[8] = 0;
    struct[9] = 0;
    struct[10] = sel;
    return isMoveLegalUnderRule1(filterCtx, board, board.turn, config, struct);
  }

  for (let r = 0; r < 9; r++) {
    for (let f = 8; f >= 0; f--) {
      const topCode = board.top(r, f);
      if (topCode === 0) continue;
      const pcTi = typeIndex(topCode);
      const pcColor = colorOf(topCode);
      if (pcColor !== board.turn) continue;
      const pcTier = board.height(r, f);
      const py = r + 1;
      const px = f + 1;
      const fromSq = sqIndex(r, f);

      const reachable = collectReachable(board, py, px, pcTi, pcColor, pcTier, config);

      for (const [sr, sf] of reachable) {
        const tr = sr - 1;
        const tf = sf - 1;
        const targetCode = board.top(tr, tf);
        const targetH = board.height(tr, tf);
        const toSq = sqIndex(tr, tf);

        if (targetCode === 0) {
          if (legal(MT_ROUTE, pcTi, pcColor, r, f, pcTier, tr, tf, 0)) {
            mask[A0_START + fromSq * 81 + toSq] = 1;
          }
          continue;
        }

        const targetTi = typeIndex(targetCode);
        const targetColor = colorOf(targetCode);
        const targetTier = targetH;

        if (targetTier < maxTier && targetTi !== MARSHAL) {
          if (pcTi !== MARSHAL || marshalCanStack) {
            if (legal(MT_TSUKE, pcTi, pcColor, r, f, pcTier, tr, tf, 0)) {
              mask[A1_START + (fromSq * 4 + 0) * 81 + toSq] = 1;
            }

            if (pcTi === TACTICIAN) {
              let enemyMask = 0;
              for (let t = 0; t < targetH; t++) {
                const code2 = board.cells[cellIndex(tr, tf, t)]!;
                if (code2 !== 0 && colorOf(code2) !== pcColor) enemyMask |= 1 << t;
              }
              if (enemyMask !== 0) {
                const qualifying = qualifyingTierMask(board, sr, sf, pcColor, -1);
                if (config.tacticianBetrayalMode === 'all-or-nothing') {
                  if (qualifying === enemyMask && legal(MT_BETRAY, pcTi, pcColor, r, f, pcTier, tr, tf, qualifying)) {
                    mask[A1_START + (fromSq * 4 + qualifying) * 81 + toSq] = 1;
                  }
                } else {
                  for (let sel = 1; sel < 4; sel++) {
                    // F3 fix (docs/design/kom1203-rules-config.md section 5):
                    // this branch used to set the mask bit unconditionally,
                    // skipping the self-exposure filter for any-subset
                    // betrayal (harmless while the two knobs were coupled to
                    // one profile string, unsound once selfExposureFilter and
                    // tacticianBetrayalMode are independently settable).
                    if ((sel & ~qualifying) === 0 && legal(MT_BETRAY, pcTi, pcColor, r, f, pcTier, tr, tf, sel)) {
                      mask[A1_START + (fromSq * 4 + sel) * 81 + toSq] = 1;
                    }
                  }
                }
              }
            }
          }
        }

        if (targetColor !== pcColor) {
          let capturedSel = 0;
          for (let t = 0; t < targetH; t++) {
            const code2 = board.cells[cellIndex(tr, tf, t)]!;
            if (code2 !== 0 && colorOf(code2) !== pcColor) capturedSel |= 1 << t;
          }
          if (legal(MT_CAPTURE, pcTi, pcColor, r, f, pcTier, tr, tf, capturedSel)) {
            mask[A0_START + fromSq * 81 + toSq] = 1;
          }
        }
      }
    }
  }
}
