// Repetition key builders (src/komugi/engine/cyengine.pyx:643-703) and the
// RepetitionRule the key is built under (src/komugi/engine/repetition.py).

import { Board } from './board.ts';
import { HAND_SIZE, MT_ARATA, MT_CAPTURE, MT_ROUTE, NUM_CELLS } from './constants.ts';
import { generateRecords } from './movegen.ts';

// A string enum (RepetitionRule(str, Enum) on the Python side) so a rule
// round-trips through JSON as its own value with no translation table.
export const RepetitionRule = { FULL: 'full', PLACEMENT: 'placement' } as const;
export type RepetitionRule = (typeof RepetitionRule)[keyof typeof RepetitionRule];

// KOM-1105 (plumbing): 'komugi' is the default/training rules, 'gungijs' is
// strict gungi.js 3.4.11 parity, 'ligungi' is the ligungi referee -- see
// tickets/KOM-1105.md and
// docs/reviews/2026-11-13-kom1105-1107-rules-profile-design-review.md.
// Carried immutably by Game (game.ts) and threaded through the
// request/worker/root/ponder-key plumbing; the three profiles diverge
// starting with KOM-1106/1107/1110/1124 (terminal.ts's hasEscapingMove now
// reads `rules` for all three). KOM-1203 (F6): these four now live in the
// leaf module rules_config.ts (which also owns RulesConfig/toRulesConfig),
// re-exported here unchanged so every existing caller of this module keeps
// compiling.
export type { RulesProfile } from './rules_config.ts';
export { RULES_PROFILES, DEFAULT_RULES_PROFILE, parseRulesProfile } from './rules_config.ts';
import { DEFAULT_RULES_PROFILE, toRulesConfig } from './rules_config.ts';
import type { RulesConfig, RulesProfile } from './rules_config.ts';

// KOM-1111: the ligungi oracle fixture's canonical move names, in Komugi's
// own type-index order (constants.ts MARSHAL..TACTICIAN) -- see
// web/engine/test/fixtures/ligungi_oracle.json's `canonicalMoveFormat`.
const LIGUNGI_ROLE_NAMES: readonly string[] = [
  'marshal', 'general', 'lieutenant', 'major', 'samurai', 'lancer', 'rider',
  'spy', 'fortress', 'soldier', 'cannon', 'archer', 'musketeer', 'tactician',
];

// The move-record capture mask (actions.ts's capturedFromMask bit layout:
// bit t -> the piece on tier t+1) reread as the fixture's own `^tiers`
// suffix: the ascending tiers a betrayal/tsuke-with-capture removes, with no
// separator (e.g. mask 0b11 -> '12'). Never set on a plain 'move'/'capture'
// (capture always clears the whole destination tower, so the fixture omits
// the suffix there) -- only a 'stack' (MT_TSUKE/MT_BETRAY) carries it, and
// only when it actually captures something.
function captureTierSuffix(mask: number): string {
  if (!mask) return '';
  const tiers: number[] = [];
  for (let t = 0; t < 3; t++) if (mask & (1 << t)) tiers.push(t + 1);
  return `^${tiers.join('')}`;
}

// KOM-1111: the fixture's canonical move string for one of generateRecords'
// move structs (movegen.ts's MoveRecord layout, shared with san.ts's
// buildSan): 'D:<role>@R-F[^tiers][+done]' for an arata placement,
// 'M:R-F>R-F:<move|stack|capture>[^tiers]' for a board move. R-F is Komugi's
// own 1-based rank-file (struct's 0-based fromR/fromF/toR/toF plus one),
// exactly as buildSan already renders it.
export function canonicalMove(struct: Int32Array): string {
  const moveType = struct[0];
  const pieceTi = struct[1]!;
  const fromR = struct[3]!;
  const fromF = struct[4]!;
  const toR = struct[6]!;
  const toF = struct[7]!;
  const draftFinished = struct[9] !== 0;
  const mask = struct[10]!;

  if (moveType === MT_ARATA) {
    const done = draftFinished ? '+done' : '';
    return `D:${LIGUNGI_ROLE_NAMES[pieceTi]}@${toR + 1}-${toF + 1}${captureTierSuffix(mask)}${done}`;
  }

  const kind = moveType === MT_ROUTE ? 'move' : moveType === MT_CAPTURE ? 'capture' : 'stack';
  const suffix = kind === 'stack' ? captureTierSuffix(mask) : '';
  return `M:${fromR + 1}-${fromF + 1}>${toR + 1}-${toF + 1}:${kind}${suffix}`;
}

// KOM-1111 (M5): the canonical legal move set BEFORE terminal checks (unlike
// Game.validMoves(), which empties at checkmate/stalemate) -- the ligungi
// oracle fixture's own legal sets are taken at the same point, before either
// engine asks whether the position is terminal.
// KOM-1106 (SPLIT 2026-11-14): `rules` now reaches generateRecords, which
// filters archer_blocking/betrayal_partial by profile (movegen_board.ts).
export function legalSet(board: Board, rules: RulesProfile | RulesConfig = DEFAULT_RULES_PROFILE): Set<string> {
  const config = toRulesConfig(rules);
  const out = new Set<string>();
  for (const { struct } of generateRecords(board, config)) out.add(canonicalMove(struct));
  return out;
}

// repetition.py:50-54.
export const FULL_KEY_BYTES = NUM_CELLS + 3 + HAND_SIZE; // 243 + 3 + 28 = 274
export const PLACEMENT_KEY_BYTES = NUM_CELLS; // 243

export function keyBytesFor(rule: RepetitionRule): number {
  if (rule === RepetitionRule.FULL) return FULL_KEY_BYTES;
  if (rule === RepetitionRule.PLACEMENT) return PLACEMENT_KEY_BYTES;
  throw new Error(`not a RepetitionRule: ${rule}`);
}

// Port of cyengine.placement_key (cyengine.pyx:679-703): the leading 243-byte
// cells block, gungi.js@3.4.11's cells-only key.
export function placementKey(board: Board): Uint8Array {
  return new Uint8Array(board.cells);
}

// Port of cyengine.repetition_key (cyengine.pyx:643-676): cells, then
// [turn, draft_w, draft_b], then the hand narrowed to uint8[28]. board.hand
// is already a Uint8Array (board.ts), so no narrowing step is needed here.
export function fullKey(board: Board): Uint8Array {
  const out = new Uint8Array(FULL_KEY_BYTES);
  out.set(board.cells, 0);
  out[NUM_CELLS] = board.turn;
  out[NUM_CELLS + 1] = board.draftW;
  out[NUM_CELLS + 2] = board.draftB;
  out.set(board.hand, NUM_CELLS + 3);
  return out;
}

export function keyFor(board: Board, rule: RepetitionRule): Uint8Array {
  if (rule === RepetitionRule.FULL) return fullKey(board);
  if (rule === RepetitionRule.PLACEMENT) return placementKey(board);
  throw new Error(`not a RepetitionRule: ${rule}`);
}

// FNV-1a prime (section 3 "Repetition counts": "a 2x32-bit hash of the key
// bytes (FNV-1a over the bytes, recomputed per ply)"). Two independent
// 32-bit accumulators are folded over the bytes in one pass -- allocation
// free, unlike the previous per-byte `String.fromCharCode` + concatenation,
// which forced a 243+-char string (and its V8 hash) to be rebuilt on every
// play()/undo()/repetitionCount().
const FNV_PRIME = 0x01000193;

function hashKeyBytes(bytes: Uint8Array): { hi: number; lo: number } {
  let hi = 0x811c9dc5;
  let lo = (0x9e3779b9 ^ bytes.length) >>> 0;
  for (let i = 0; i < bytes.length; i++) {
    const b = bytes[i]!;
    hi = Math.imul(hi ^ b, FNV_PRIME);
    lo = Math.imul(lo ^ b, FNV_PRIME);
  }
  return { hi: hi >>> 0, lo: lo >>> 0 };
}

function bytesEqual(a: Uint8Array, b: Uint8Array): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

// A JS number cannot hold the 2x32-bit hash exactly; this is the lossy fold
// into one safe-integer bucket id (N8): the high half times 2^21 plus the
// low 21 bits of the low half. Folding is allowed to collide -- every bucket
// is a chain, and `bytesEqual` below is the actual exactness check, so a
// fold (or even a full hash) collision never changes a count.
function foldKey(hi: number, lo: number): number {
  return hi * 0x200000 + (lo & 0x1fffff);
}

type Slot = { hi: number; lo: number; bytes: Uint8Array; count: number };

// Occurrence counts for repetition keys (section 3 "Repetition counts"):
// a Map<number, slot[]> bucketed by the lossy hash fold, each bucket a short
// chain of byte-exact slots, instead of Python's dict keyed by the exact
// bytes. On the common ply -- a position never seen before -- `increment`
// does one hash pass and a bucket miss, no byte comparison at all; only a
// real repeat (or a fold collision) walks the chain and compares bytes.
export class RepetitionCounts {
  private buckets = new Map<number, Slot[]>();

  private findSlot(hi: number, lo: number, bytes: Uint8Array): Slot | undefined {
    const chain = this.buckets.get(foldKey(hi, lo));
    if (!chain) return undefined;
    for (const slot of chain) {
      if (slot.hi === hi && slot.lo === lo && bytesEqual(slot.bytes, bytes)) return slot;
    }
    return undefined;
  }

  // Increments the count for `bytes` (the caller hands over ownership: the
  // array must never be mutated afterwards, exactly as Game already treats
  // its own `key`). Returns the hash so the caller can look the same slot up
  // again later (repetitionCount(), undo()) without rehashing.
  increment(bytes: Uint8Array): { hi: number; lo: number } {
    const { hi, lo } = hashKeyBytes(bytes);
    const fold = foldKey(hi, lo);
    let chain = this.buckets.get(fold);
    if (!chain) {
      chain = [];
      this.buckets.set(fold, chain);
    }
    for (const slot of chain) {
      if (slot.hi === hi && slot.lo === lo && bytesEqual(slot.bytes, bytes)) {
        slot.count += 1;
        return { hi, lo };
      }
    }
    chain.push({ hi, lo, bytes, count: 1 });
    return { hi, lo };
  }

  // Count of the key already hashed to (hi, lo) -- Game.repetitionCount().
  count(hi: number, lo: number, bytes: Uint8Array): number {
    return this.findSlot(hi, lo, bytes)?.count ?? 0;
  }

  // Sets the count directly, creating the slot if absent (Game's
  // constructor seed, and the bench's skip-decrement mutant restoring a
  // count undo() just correctly decremented). Returns the hash.
  setCount(bytes: Uint8Array, count: number): { hi: number; lo: number } {
    const { hi, lo } = hashKeyBytes(bytes);
    const fold = foldKey(hi, lo);
    let chain = this.buckets.get(fold);
    if (!chain) {
      chain = [];
      this.buckets.set(fold, chain);
    }
    for (const slot of chain) {
      if (slot.hi === hi && slot.lo === lo && bytesEqual(slot.bytes, bytes)) {
        slot.count = count;
        return { hi, lo };
      }
    }
    chain.push({ hi, lo, bytes, count });
    return { hi, lo };
  }

  // Decrements the count for the key already hashed to (hi, lo), deleting
  // the slot at count 1 rather than leaving a 0 entry (Game.undo()).
  decrement(hi: number, lo: number, bytes: Uint8Array): void {
    const fold = foldKey(hi, lo);
    const chain = this.buckets.get(fold);
    if (!chain) return;
    for (let i = 0; i < chain.length; i++) {
      const slot = chain[i]!;
      if (slot.hi === hi && slot.lo === lo && bytesEqual(slot.bytes, bytes)) {
        if (slot.count <= 1) {
          chain.splice(i, 1);
          if (chain.length === 0) this.buckets.delete(fold);
        } else {
          slot.count -= 1;
        }
        return;
      }
    }
  }

  // An independent copy: every bucket's slots are copied as new objects
  // (so incrementing/decrementing one side never touches the other), but
  // `bytes` is shared since slots never mutate it (increment/setCount hand
  // over ownership once, per the comment above).
  copy(): RepetitionCounts {
    const out = new RepetitionCounts();
    for (const [fold, chain] of this.buckets) {
      out.buckets.set(fold, chain.map((slot) => ({ ...slot })));
    }
    return out;
  }

  // A full, order-independent dump of every live count, for the bench's
  // byte-exact undo check (section 3, M9) only -- never on the hot path.
  // (hi, lo, count) alone is enough to catch a missed decrement/increment:
  // this is a test-comparison fingerprint, not the identity check that
  // guards a real count (that one is always the full `bytesEqual`, above).
  snapshot(): string {
    const rows: [number, number, number][] = [];
    for (const chain of this.buckets.values()) {
      for (const slot of chain) rows.push([slot.hi, slot.lo, slot.count]);
    }
    rows.sort((a, b) => a[0] - b[0] || a[1] - b[1] || a[2] - b[2]);
    return rows.map((r) => r.join(':')).join('|');
  }
}
