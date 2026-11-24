// Arata (hand-drop) mask fill and move-record emitter, blocks B/C.
// Port of src/komugi/engine/cymovegen.pyx:
//   _qualifying_tier_mask (366-430), _generate_arata (687-848),
//   _fill_arata (955-1066), the _hand_order loops of generate_all_moves /
//   fill_legal_mask (1090-1106, 1127-1141).
// Board moves are U2a's movegen_board.ts; this file only ever touches ids
// >= B_START.

import { Board, colorOf, typeIndex } from './board.ts';
import { B_START, C_START, MARSHAL, MT_ARATA, NUM_PIECE_TYPES, TACTICIAN } from './constants.ts';
import { buildFilterContext, isMoveLegalUnderRule1, type FilterContext } from './ligungi_filter.ts';
import { toRulesConfig } from './rules_config.ts';
import type { RulesConfig, RulesProfile } from './rules_config.ts';

// struct layout: identical to movegen_board.ts (Int32Array(16), reused):
//   [0] move_type (MT_ARATA)  [1] piece (type index)  [2] color
//   [3] from_r = -1           [4] from_f = -1         [5] from_tier = 0
//   [6] to_r                  [7] to_f                [8] to_tier
//   [9] draft_finished (0/1)
//   [10] captured tier-mask: bit (tier-1) set iff the piece at that tier of
//        the destination tower is converted (an arata can only flip tier 1)
//   [11..15] reserved, always 0
export type ArataMoveCallback = (id: number, struct: Int32Array) => void;

function sqIndex(r: number, f: number): number {
  return r * 9 + (8 - f);
}

// Port of cymovegen._qualifying_tier_mask with 0-based (r, f): the tiers whose
// enemy the mover may convert. drop_ti is the hand type spent first (KOM-170).
function qualifyingTierMask(board: Board, r: number, f: number, color: number, dropTi: number): number {
  const base = (r * 9 + f) * 3;
  let mask = 0;
  for (let t = 0; t < 3; t++) {
    const code = board.cells[base + t]!;
    if (code === 0 || colorOf(code) === color) continue;
    const ti = typeIndex(code);
    let count = 0;
    for (let u = 0; u < 3; u++) {
      const code2 = board.cells[base + u]!;
      if (code2 !== 0 && colorOf(code2) !== color && typeIndex(code2) === ti) count++;
    }
    if (ti === dropTi) count++;
    if (board.hand[ti * 2 + color]! >= count) mask |= 1 << t;
  }
  return mask;
}

// Eligible 1-based ranks for a drop, in cymovegen's order (cymovegen.pyx:703-745).
function eligibleRanks(board: Board, color: number): number[] {
  if (board.draftB !== 0 || board.draftW !== 0) {
    return color === 0 ? [7, 8, 9] : [1, 2, 3];
  }
  const ranks: number[] = [];
  let maybe: number[] = [];
  const step = color === 1 ? 1 : -1;
  for (let rank = color === 1 ? 1 : 9; rank >= 1 && rank <= 9; rank += step) {
    let hasFriendly = false;
    for (let f = 0; f < 9; f++) {
      const top = board.top(rank - 1, f);
      if (top !== 0 && colorOf(top) === color) {
        hasFriendly = true;
        break;
      }
    }
    if (hasFriendly) {
      for (const m of maybe) ranks.push(m);
      ranks.push(rank);
      maybe = [];
    } else {
      maybe.push(rank);
    }
  }
  return ranks;
}

// Drives one piece's drops: calls emit(r, f, toTier, bsel, finish)
// in cymovegen's order. Shared by the record emitter and the mask filler,
// which cymovegen keeps as two copies (_generate_arata / _fill_arata) of the
// same enumeration.
function forEachDrop(
  board: Board,
  ti: number,
  color: number,
  emit: (r: number, f: number, toTier: number, bsel: number, finish: number) => void,
): void {
  if (color !== board.turn) return;
  if (board.hand[ti * 2 + color]! <= 0) return;
  if (board.hand[MARSHAL * 2 + color]! > 0 && ti !== MARSHAL) return;

  const maxTier = board.mode === 3 ? 3 : 2;
  const ranks = eligibleRanks(board, color);

  let handCount = 0;
  for (let t = 0; t < NUM_PIECE_TYPES; t++) handCount += board.hand[t * 2 + color]!;
  const isLastPiece = handCount === 1;
  const drafting = color === 0 ? board.draftW !== 0 : board.draftB !== 0;

  for (const rank of ranks) {
    const r = rank - 1;
    for (let f = 0; f < 9; f++) {
      const top = board.top(r, f);
      const h = board.height(r, f);
      let toTier: number;
      if (top === 0) {
        toTier = 1;
      } else if (colorOf(top) === color && h < maxTier && typeIndex(top) !== MARSHAL) {
        toTier = h + 1;
      } else {
        continue;
      }

      // A legal drop target is empty or friendly on top with h < 3, so tier 1
      // is the only enemy _get_betrayal_combos can ever offer: its combos are
      // [] or [[(ti, 1)]], i.e. the one-bit selector (cymovegen.pyx:1042-1051).
      let bsel = 0;
      if (ti === TACTICIAN && h > 0) {
        const q = qualifyingTierMask(board, r, f, color, ti);
        if (q & ~1) throw new Error(`arata conversion beyond tier 1 at r=${r} f=${f}`);
        bsel = q & 1;
      }

      if (drafting) {
        if (!isLastPiece) {
          emit(r, f, toTier, 0, 0);
          if (bsel) emit(r, f, toTier, 1, 0);
        }
        emit(r, f, toTier, 0, 1);
        if (bsel) emit(r, f, toTier, 1, 1);
      } else {
        emit(r, f, toTier, 0, 0);
        if (bsel) emit(r, f, toTier, 1, 0);
      }
    }
  }
}

function arataId(ti: number, bsel: number, r: number, f: number, finish: number): number {
  return (finish ? C_START : B_START) + (ti * 2 + bsel) * 81 + sqIndex(r, f);
}

// KOM-1139: rule (1) self-exposure filter never applies during the draft
// (design doc 4.1/4.2 -- drops only ever fall under rule 1 post-draft); the
// context is built once per call, same as movegen_board.ts.
function arataFilterContext(board: Board, config: RulesConfig): FilterContext {
  if (!config.selfExposureFilter) return null;
  if (board.draftW !== 0 || board.draftB !== 0) return null;
  return buildFilterContext(board, board.turn, config);
}

// Module-scoped state for forEachArataMove's forEachDrop callback (famEmit
// below), same rationale as movegen_board.ts's fbmVisit: forEachArataMove is
// not reentrant from within its own callback (none of its callers --
// terminal.ts's hasEscapingMove, movegen.ts's generateRecords, actions.ts's
// decodeAction -- re-enter it while a drop is in flight), so one shared
// struct/closure per module replaces a fresh Int32Array(16) and a fresh
// per-hand-entry arrow closure forEachDrop used to get on every call
// (profiled: the largest single closure-allocation cost in this file).
const famStruct = new Int32Array(16);
let famBoard: Board;
let famConfig: RulesConfig = toRulesConfig('komugi');
let famFilterCtx: FilterContext = null;
let famFiltering = false;
let famCb: ArataMoveCallback;
let famTi = 0;
let famColor = 0;

function famEmit(r: number, f: number, toTier: number, bsel: number, finish: number): void {
  const struct = famStruct;
  struct[0] = MT_ARATA;
  struct[1] = famTi;
  struct[2] = famColor;
  struct[3] = -1;
  struct[4] = -1;
  struct[5] = 0;
  struct[6] = r;
  struct[7] = f;
  struct[8] = toTier;
  struct[9] = finish;
  struct[10] = bsel;
  if (famFiltering && !isMoveLegalUnderRule1(famFilterCtx, famBoard, famBoard.turn, famConfig, struct)) return;
  famCb(arataId(famTi, bsel, r, f, finish), struct);
}

// Port of the _hand_order loop over _generate_arata (cymovegen.pyx:1092-1095,
// 1103-1106): every arata record in generate_all_moves order, draft or not.
export function forEachArataMove(board: Board, cb: ArataMoveCallback, rules: RulesProfile | RulesConfig = 'komugi'): void {
  const config = toRulesConfig(rules);
  famBoard = board;
  famConfig = config;
  famFilterCtx = arataFilterContext(board, config);
  famFiltering = config.selfExposureFilter && (board.draftW === 0 && board.draftB === 0);
  famCb = cb;
  for (let i = 0; i < board.handOrderLen; i++) {
    const entry = board.handOrder[i]!;
    famTi = entry >> 1;
    famColor = entry & 1;
    forEachDrop(board, famTi, famColor, famEmit);
  }
}

// Module-scoped state for fillArataMask's forEachDrop callback (famFillEmit
// below), same rationale as famEmit above (a separate struct/state from
// forEachArataMove's since the two are never nested, but kept distinct for
// clarity).
const famFillStruct = new Int32Array(16);
let famFillBoard: Board;
let famFillConfig: RulesConfig = toRulesConfig('komugi');
let famFillFilterCtx: FilterContext = null;
let famFillFiltering = false;
let famFillMask: Uint8Array;
let famFillTi = 0;
let famFillColor = 0;

function famFillEmit(r: number, f: number, toTier: number, bsel: number, finish: number): void {
  if (famFillFiltering) {
    const struct = famFillStruct;
    struct[0] = MT_ARATA;
    struct[1] = famFillTi;
    struct[2] = famFillColor;
    struct[3] = -1;
    struct[4] = -1;
    struct[5] = 0;
    struct[6] = r;
    struct[7] = f;
    struct[8] = toTier;
    struct[9] = finish;
    struct[10] = bsel;
    if (!isMoveLegalUnderRule1(famFillFilterCtx, famFillBoard, famFillBoard.turn, famFillConfig, struct)) return;
  }
  famFillMask[arataId(famFillTi, bsel, r, f, finish)] = 1;
}

// Port of the _hand_order loop over _fill_arata (cymovegen.pyx:1128-1131,
// 1138-1141): sets mask[id] = 1 for every arata id; writes nothing else.
export function fillArataMask(board: Board, mask: Uint8Array, rules: RulesProfile | RulesConfig = 'komugi'): void {
  const config = toRulesConfig(rules);
  famFillBoard = board;
  famFillConfig = config;
  famFillFilterCtx = arataFilterContext(board, config);
  famFillFiltering = config.selfExposureFilter && (board.draftW === 0 && board.draftB === 0);
  famFillMask = mask;
  for (let i = 0; i < board.handOrderLen; i++) {
    const entry = board.handOrder[i]!;
    famFillTi = entry >> 1;
    famFillColor = entry & 1;
    forEachDrop(board, famFillTi, famFillColor, famFillEmit);
  }
}
