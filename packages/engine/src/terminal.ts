// Attack and terminal predicates. Faithful port of src/komugi/engine/cyengine.pyx:
//   direction + probe tables (cyengine.pyx:299-338), SQUARES order (340-348),
//   _get_available_squares (355-426), _get_attacked_squares (433-472),
//   _is_square_attacked (475-493), _in_check / in_check (496-513),
//   _marshal_captured (520-530), _has_escaping_move (540-554),
//   is_checkmate / is_stalemate (557-572), is_insufficient_material (579-623),
//   is_repetition_draw (706-714), is_terminal_nonmate (721-736),
//   is_game_over (754-767).
//
// cyengine.pyx keeps its own copy of the direction/probe tables and
// _get_available_squares rather than cimporting cymovegen's (its own comment:
// "duplicated from cymovegen — cdef functions there are not accessible via
// cimport since cymovegen.pxd doesn't export them"). This file does the same
// thing on the TS side for the same reason: movegen_board.ts (U2a) does not
// export its copies, and a unit never edits a file it does not own. The two
// copies are pinned equal by the exported in_check_w/in_check_b of every
// move-record position (docs/design/clientside-engine-port.md section 2.3, S2).

import { Board, cellIndex, colorOf, typeIndex } from './board.ts';
import { applyMove } from './apply.ts';
import { forEachArataMove } from './movegen_arata.ts';
import { forEachBoardMove } from './movegen_board.ts';
import { ARCHER, BLACK, CANNON, MARSHAL, MUSKETEER, NUM_PIECE_TYPES, WHITE } from './constants.ts';
import { DEFAULT_RULES_PROFILE, RULES_PRESETS, toRulesConfig } from './rules_config.ts';
import type { RulesConfig, RulesProfile } from './rules_config.ts';
import { archerElbowBlocked } from './rules_ligungi.ts';

// Direction table: [NW, N, NE, E, W, SE, S, SW] as (dy, dx), white's frame.
const DIRS: ReadonlyArray<readonly [number, number]> = [
  [-1, 1], [-1, 0], [-1, -1], [0, 1], [0, -1], [1, 1], [1, 0], [1, -1],
];

// probe_pval[type_index][dir]; 0 = inactive, 99 = slider (cyengine.pyx:325-338).
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

// probe_pcarry[type_index][dir] (cyengine.pyx:325-338).
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

// Scratch outputs of readTopTier (module-scoped to avoid a per-call tuple
// allocation; read immediately after the call, never across a nested call
// into readTopTier itself -- see its one call site below).
let scratchTopCode = 0;
let scratchTopTier = 0;

// Fused topAt+heightAt: both scan the same tower top-down to the same stop
// condition, so this reads it once instead of twice (same rationale as
// isSquareAttacked's inline scan below).
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

// Faithful port of cyengine._get_available_squares (cyengine.pyx:355-426),
// itself a duplicate of cymovegen._get_available_squares. Coordinates are
// 1-based throughout, matching the Cython source exactly.
//
// Hot-path note (section 3, "no per-call allocation in hot paths"):
// isSquareAttacked only ever asks "is (targetRank, targetFile) in the
// available set", so this checks membership inline during the same scan the
// original built a list from, instead of allocating `Array<[number,
// number]>` per direction per candidate piece. inCheck runs up to three
// times per winState() call (isCheckmate, isStalemate, and once per
// hasEscapingMove trial), so this was the single largest allocation source
// on the replay path (profiled: ~33% of self time pre-fix).
function squareInAvailableSquares(
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
): boolean {
  if (originTi === ARCHER && Math.abs(dy) === 1 && Math.abs(dx) === 1) {
    const wingRank = py + dy;
    const wingFile = px + dx;
    if (wingRank >= 1 && wingRank <= 9 && wingFile >= 1 && wingFile <= 9) {
      readTopTier(board, wingRank, wingFile);
      if (scratchTopCode !== 0 && scratchTopTier > originTier) return false;
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
      if (scratchTopTier > originTier) return false;
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
    if (code !== 0) {
      if (scratchTopTier > originTier) break;
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

// Port of cyengine._get_attacked_squares (cyengine.pyx:433-472), as a
// membership test against one target square rather than the full list (see
// squareInAvailableSquares). topCode/pcTier are the caller's own top()/
// height() reading of (py, px) -- isSquareAttacked already finds the
// occupying piece to call this, so this takes it rather than re-scanning
// the same tower twice more (once for topAt, once for heightAt).
function pieceAttacksSquare(
  board: Board,
  py: number,
  px: number,
  topCode: number,
  pcTier: number,
  targetRank: number,
  targetFile: number,
  config: RulesConfig,
): boolean {
  const pcTi = typeIndex(topCode);
  const pcColor = colorOf(topCode);

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

    if (
      squareInAvailableSquares(
        board, dy, dx, sy, sx, py, px, pcTi, pcColor, pcTier, length, targetRank, targetFile, config,
      )
    ) {
      return true;
    }
  }
  return false;
}

// Port of cyengine._is_square_attacked (cyengine.pyx:475-493). Scans every
// square in SQUARES order (rank 1->9, file 9->1), matching the per-square
// loop order movegen_board.ts uses for the same iteration.
function isSquareAttacked(
  board: Board,
  targetRank: number,
  targetFile: number,
  byColor: number,
  config: RulesConfig,
): boolean {
  for (let r = 0; r < 9; r++) {
    for (let f = 8; f >= 0; f--) {
      const base = (r * 9 + f) * 3;
      let topCode = 0;
      let tier = 0;
      for (let t = 2; t >= 0; t--) {
        const c = board.cells[base + t]!;
        if (c !== 0) {
          topCode = c;
          tier = t + 1;
          break;
        }
      }
      if (topCode === 0) continue;
      if (colorOf(topCode) !== byColor) continue;
      const rank = r + 1;
      const file = f + 1;
      if (pieceAttacksSquare(board, rank, file, topCode, tier, targetRank, targetFile, config)) return true;
    }
  }
  return false;
}

// Port of cyengine._in_check / the public in_check (cyengine.pyx:496-513).
// KOM-1106 (SPLIT 2026-11-14): `config.archerElbowBlocking` reaches the attack
// scan so an archer's ligungi tier-2/3 blocking (rules_ligungi.ts) applies to
// check detection too.
// KOM-1203: split into this RulesConfig-only core and the exported wrapper
// below, so hot-path internal callers (tryEscape, isCheckmate/isStalemate/
// isGameOver below) resolve `rules` once per request and call this directly,
// rather than re-resolving via toRulesConfig on every one of inCheck's up-to-
// three calls per winState().
function inCheckConfig(board: Board, color: number, config: RulesConfig): boolean {
  const opposite = 1 - color;
  for (let r = 0; r < 9; r++) {
    for (let f = 8; f >= 0; f--) {
      const rank = r + 1;
      const file = f + 1;
      const topCode = topAt(board, rank, file);
      if (topCode !== 0 && typeIndex(topCode) === MARSHAL && colorOf(topCode) === color) {
        return isSquareAttacked(board, rank, file, opposite, config);
      }
    }
  }
  return false;
}

export function inCheck(
  board: Board,
  color: number,
  rules: RulesProfile | RulesConfig = DEFAULT_RULES_PROFILE,
): boolean {
  return inCheckConfig(board, color, toRulesConfig(rules));
}

// Port of cyengine._marshal_captured (cyengine.pyx:520-530): true when the
// board does not hold exactly 2 marshals.
function isMarshalCaptured(board: Board): boolean {
  let count = 0;
  for (let r = 0; r < 9; r++) {
    for (let f = 0; f < 9; f++) {
      for (let t = 0; t < 3; t++) {
        const code = board.cells[cellIndex(r, f, t)]!;
        if (code !== 0 && typeIndex(code) === MARSHAL) count++;
      }
    }
  }
  return count !== 2;
}

// Port of cyengine._has_escaping_move (cyengine.pyx:540-554): true if the
// side to move has any pseudo-legal move that does not leave its own marshal
// in check. Section 3 ("Terminal checks... runs with early exit over the
// mask filler"): tries candidates straight off the board/arata emitters and
// stops at the first escape, instead of materializing every move record
// (movegen.ts's generateRecords) and cloning a fresh Board (board.copy())
// per candidate. The trial board is one reused scratch instance reset by
// byte-copy before every candidate -- cheaper than a Board allocation and
// exactly as disposable, since every trial here is discarded regardless of
// its outcome (M9's own undo stack is Game's, not Board's).
//
// KOM-1110/KOM-1124: under every profile except 'gungijs' this also
// matches the fixed cyengine _has_escaping_move (KOM-1109) -- a reply that
// captures the OPPOSING marshal counts as an escape too, since it ends the
// game in the mover's favour (marshal_captured) rather than leaving the
// mover nominally "in check" by the piece it just captured. 'komugi' matches
// the fixed cyengine directly; 'ligungi' gets the same escape too, per
// KOM-1106's UPDATE 2026-11-13 (87cd3a3): capturing the opponent's marshal
// is ALWAYS legal under ligungi and wins, the same escape semantics as
// KOM-1109. 'gungijs' keeps the pre-fix gungi.js 3.4.11 behaviour (a false
// mate there still scores as mate, until jwyce/gungi.js#53 lands).
const ESCAPE_FOUND = Symbol('escape-found');
const scratchBoard = new Board();

function tryEscape(board: Board, mover: number, config: RulesConfig, struct: Int32Array): void {
  scratchBoard.cells.set(board.cells);
  scratchBoard.hand.set(board.hand);
  scratchBoard.turn = board.turn;
  scratchBoard.draftW = board.draftW;
  scratchBoard.draftB = board.draftB;
  applyMove(scratchBoard, struct);
  if (!inCheckConfig(scratchBoard, mover, config)) throw ESCAPE_FOUND;
  if (config.marshalCaptureIsEscape && isMarshalCaptured(scratchBoard)) throw ESCAPE_FOUND;
}

// Module-scoped callback for hasEscapingMove's forEachBoardMove/forEachArataMove
// calls (same rationale as scratchBoard above): hasEscapingMove runs on every
// non-terminal ply and is not reentrant (tryEscape never triggers another
// hasEscapingMove call), so these three hold the current call's (board, mover,
// rules) instead of a fresh closure being allocated per candidate move.
let hemBoard: Board;
let hemMover = 0;
let hemConfig: RulesConfig = RULES_PRESETS[DEFAULT_RULES_PROFILE];

function hemTryEscape(_id: number, struct: Int32Array): void {
  tryEscape(hemBoard, hemMover, hemConfig, struct);
}

function hasEscapingMove(board: Board, config: RulesConfig): boolean {
  hemBoard = board;
  hemMover = board.turn;
  hemConfig = config;
  try {
    forEachBoardMove(board, hemTryEscape, config);
    forEachArataMove(board, hemTryEscape, config);
  } catch (e) {
    if (e === ESCAPE_FOUND) return true;
    throw e;
  }
  return false;
}

// Port of cyengine.is_checkmate (cyengine.pyx:557-563).
// KOM-1110: `rules` now selects hasEscapingMove's profile-dependent
// marshal-capture-is-an-escape fix -- see hasEscapingMove's own note.
export function isCheckmate(board: Board, rules: RulesProfile | RulesConfig = DEFAULT_RULES_PROFILE): boolean {
  if (board.draftW !== 0 || board.draftB !== 0) return false;
  if (isMarshalCaptured(board)) return false;
  const config = toRulesConfig(rules);
  return inCheckConfig(board, board.turn, config) && !hasEscapingMove(board, config);
}

// Port of cyengine.is_stalemate (cyengine.pyx:566-572).
// KOM-1110: see isCheckmate's own note on `rules`.
export function isStalemate(board: Board, rules: RulesProfile | RulesConfig = DEFAULT_RULES_PROFILE): boolean {
  if (board.draftW !== 0 || board.draftB !== 0) return false;
  if (isMarshalCaptured(board)) return false;
  const config = toRulesConfig(rules);
  return !inCheckConfig(board, board.turn, config) && !hasEscapingMove(board, config);
}

// Port of cyengine.is_insufficient_material (cyengine.pyx:579-623): true if
// only two non-adjacent marshals remain, with no non-marshal hand pieces.
export function isInsufficientMaterial(board: Board): boolean {
  if (board.draftW !== 0 || board.draftB !== 0) return false;

  for (let ti = 0; ti < NUM_PIECE_TYPES; ti++) {
    if (ti === MARSHAL) continue;
    if (board.hand[ti * 2 + WHITE]! > 0 || board.hand[ti * 2 + BLACK]! > 0) return false;
  }

  let marshalCount = 0;
  let nonMarshalCount = 0;
  let m1Rank = 0;
  let m1File = 0;
  let m2Rank = 0;
  let m2File = 0;

  for (let r = 0; r < 9; r++) {
    for (let f = 0; f < 9; f++) {
      for (let t = 0; t < 3; t++) {
        const code = board.cells[cellIndex(r, f, t)]!;
        if (code === 0) continue;
        if (typeIndex(code) === MARSHAL) {
          marshalCount++;
          if (marshalCount === 1) {
            m1Rank = r + 1;
            m1File = f + 1;
          } else if (marshalCount === 2) {
            m2Rank = r + 1;
            m2File = f + 1;
          }
        } else {
          nonMarshalCount++;
        }
      }
    }
  }

  if (nonMarshalCount > 0) return false;
  if (marshalCount !== 2) return false;

  const rankDiff = Math.abs(m1Rank - m2Rank);
  const fileDiff = Math.abs(m1File - m2File);
  const isAdjacent = rankDiff <= 1 && fileDiff <= 1;
  return !isAdjacent;
}

// Port of cyengine._REPETITION_THRESHOLD / is_repetition_draw (cyengine.pyx:640,
// 706-714): >= 4, never == 4 (monotone, never switched off by a later ply).
const REPETITION_THRESHOLD = 4;

export function isRepetitionDraw(count: number): boolean {
  return count >= REPETITION_THRESHOLD;
}

// Port of cyengine.is_terminal_nonmate (cyengine.pyx:721-736): terminal for a
// non-checkmate/stalemate reason, with no move generation. repetitionCount is
// the current position's occurrence count (0 when no history is tracked).
// KOM-1105 (plumbing only): `rules` is accepted only for call-site
// uniformity with isGameOver/isCheckmate/isStalemate -- it is genuinely
// unused here, since none of marshal-captured/repetition/insufficient-
// material is profile-dependent. Only hasEscapingMove's marshal-capture-is-
// an-escape fix reads `rules` (isCheckmate's own note), and this function
// never calls hasEscapingMove.
export function isTerminalNonmate(
  board: Board,
  repetitionCount: number,
  rules: RulesProfile | RulesConfig = DEFAULT_RULES_PROFILE,
): boolean {
  void rules;
  if (board.draftW !== 0 || board.draftB !== 0) return false;
  if (isMarshalCaptured(board)) return true;
  if (isRepetitionDraw(repetitionCount)) return true;
  if (isInsufficientMaterial(board)) return true;
  return false;
}

// Port of cyengine.is_game_over (cyengine.pyx:754-767). The last line reads
// `isCheckmate(board) || isStalemate(board)` in the Cython source; by the
// time it runs here, draft is already ruled out and isMarshalCaptured is
// already false, which is exactly the common precondition both predicates
// share (cyengine.pyx:557-572). Under that precondition
// `isCheckmate(b) = inCheck(b) && !hasEscapingMove(b)` and
// `isStalemate(b) = !inCheck(b) && !hasEscapingMove(b)` partition on
// inCheck, so their disjunction is exactly `!hasEscapingMove(b)` regardless
// of check status -- an algebraic identity, not an approximation. This
// drops the two redundant inCheck/isMarshalCaptured recomputations
// isCheckmate and isStalemate each do on this path (profiled: the single
// largest reducible cost on the replay path, inCheck being dominated by
// squareInAvailableSquares' attack scan).
// KOM-1110: see isCheckmate's own note on `rules`. The algebraic identity
// above holds per-profile too: both isCheckmate and isStalemate call
// hasEscapingMove(board, rules) with the SAME rules, so their disjunction
// is still exactly `!hasEscapingMove(board, rules)`.
export function isGameOver(
  board: Board,
  repetitionCount = 0,
  rules: RulesProfile | RulesConfig = DEFAULT_RULES_PROFILE,
): boolean {
  if (board.draftW !== 0 || board.draftB !== 0) return false;
  if (isMarshalCaptured(board)) return true;
  if (isRepetitionDraw(repetitionCount)) return true;
  if (isInsufficientMaterial(board)) return true;
  return !hasEscapingMove(board, toRulesConfig(rules));
}
