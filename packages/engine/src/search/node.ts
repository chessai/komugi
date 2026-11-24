// Ports src/komugi/search/match/node.py: one expanded match-search position
// and the read-time arithmetic over it (docs/design/clientside-engine-port.md
// section 2.6).

import { pairwiseSum, sumRowsSequential } from './pairwise.ts';

// Length of a WDL triple, and of the packed second moment's diagonal
// (node.py:63).
export const WDL = 3;

// Unique entries of a symmetric 3x3 matrix: (ww, wd, wl, dd, dl, ll)
// (node.py:69).
export const MOMENT_SIZE = 6;

// design A2's "no previous move" bias-key slot (node.py:73).
export const NO_PREVIOUS_ACTION = -1;

// "no mover is known yet" (node.py:80).
export const NO_MOVER = -1;

// FNV-1a 64 offset basis 0xCBF29CE484222325, split into 32-bit halves.
const _OFFSET_HI = 0xcbf29ce4;
const _OFFSET_LO = 0x84222325;
// FNV-1a 64 prime 0x100000001B3 = 0x100 * 2**32 + 0x1B3, split the same way.
// Both halves are small (256 and 435), so multiplying by them stays well
// inside float64's exact-integer range (53 bits) and plain `*`/`%` give an
// exact 64-bit result without Math.imul or BigInt on this path.
const _PRIME_HI = 0x100;
const _PRIME_LO = 0x1b3;
const _TWO_32 = 4294967296;

export type LegalHash = { hi: number; lo: number };

// legal_hash_of (node.py:88-117): FNV-1a 64 over the int32 LE bytes of
// `actions`, kept as its two unsigned 32-bit halves rather than widened to a
// signed int64 -- every caller only ever compares two hashes for equality,
// which (hi, lo) pairwise equality already gives without BigInt.
export function legalHashOf(actions: Int32Array): LegalHash {
  const bytes = new Uint8Array(actions.buffer, actions.byteOffset, actions.byteLength);
  let hi = _OFFSET_HI;
  let lo = _OFFSET_LO;
  for (let i = 0; i < bytes.length; i++) {
    lo = (lo ^ bytes[i]!) >>> 0;
    // (hi*2**32 + lo) * (_PRIME_HI*2**32 + _PRIME_LO) mod 2**64
    //   = (hi*_PRIME_LO + lo*_PRIME_HI) * 2**32 + lo*_PRIME_LO, mod 2**64
    const p0 = lo * _PRIME_LO;
    const p1 = hi * _PRIME_LO + lo * _PRIME_HI;
    const p0Hi = Math.floor(p0 / _TWO_32);
    const p0Lo = p0 - p0Hi * _TWO_32;
    const sum = p1 + p0Hi;
    hi = sum % _TWO_32;
    lo = p0Lo;
  }
  return { hi: hi >>> 0, lo: lo >>> 0 };
}

export function legalHashEqual(a: LegalHash, b: LegalHash): boolean {
  return a.hi === b.hi && a.lo === b.lo;
}

// Test/export-only: the unsigned 64-bit hash reinterpreted as Python's
// signed int64 (node.py's declared storage dtype). Never on the hot path.
export function legalHashToBigInt(h: LegalHash): bigint {
  const unsigned = (BigInt(h.hi) << 32n) | BigInt(h.lo);
  return unsigned >= 1n << 63n ? unsigned - (1n << 64n) : unsigned;
}

// bias_key_of (node.py:120-143): (mover, action, previous action) packed in
// mixed radix A+1. The max key is about 2.8e9, well inside a JS number's
// exact integer range.
export function biasKeyOf(
  toPlay: number,
  action: number,
  previousAction: number,
  numActions: number,
): number {
  const radix = numActions + 1;
  return (toPlay * radix + (action + 1)) * radix + (previousAction + 1);
}

// edge_action_of (node.py:146-173): the inverse of bias_key_of's middle slot.
export function edgeActionOf(biasKey: number, numActions: number): number {
  const radix = numActions + 1;
  return (Math.floor(biasKey / radix) % radix) - 1;
}

// utility (node.py:176-197): w - l + contempt * d.
export function utility(wdl: Float64Array, contempt: number): number {
  return wdl[0]! - wdl[2]! + contempt * wdl[1]!;
}

// pack_outer (node.py:219-246): x x' packed as (ww, wd, wl, dd, dl, ll).
export function packOuter(wdl: Float64Array): Float64Array {
  const win = wdl[0]!;
  const draw = wdl[1]!;
  const loss = wdl[2]!;
  return Float64Array.from([
    win * win,
    win * draw,
    win * loss,
    draw * draw,
    draw * loss,
    loss * loss,
  ]);
}

// quadratic_form (node.py:249-278): q'Mq for each packed row of `moment`,
// q = (1, c, -1). `moment` is flattened (L, MOMENT_SIZE) row-major. The
// expansion keeps Python's left-to-right term order exactly (design S11).
export function quadraticForm(moment: Float64Array, contempt: number): Float64Array {
  const rows = moment.length / MOMENT_SIZE;
  const out = new Float64Array(rows);
  for (let i = 0; i < rows; i++) {
    const base = i * MOMENT_SIZE;
    const m0 = moment[base + 0]!;
    const m1 = moment[base + 1]!;
    const m2 = moment[base + 2]!;
    const m3 = moment[base + 3]!;
    const m4 = moment[base + 4]!;
    const m5 = moment[base + 5]!;
    let v = m0;
    v = v + contempt * contempt * m3;
    v = v + m5;
    v = v + 2.0 * contempt * m1;
    v = v - 2.0 * m2;
    v = v - 2.0 * contempt * m4;
    out[i] = v;
  }
  return out;
}

// masked_prior (node.py:281-300): softmax of `logits` over the node's legal
// actions, max-subtracted in float64, stored float32.
export function maskedPrior(logits: Float32Array): Float32Array {
  const n = logits.length;
  if (n === 0) return new Float32Array(0);
  let max = logits[0]!;
  for (let i = 1; i < n; i++) if (logits[i]! > max) max = logits[i]!;
  const exp = new Float64Array(n);
  for (let i = 0; i < n; i++) exp[i] = Math.exp(logits[i]! - max);
  const sum = pairwiseSum(exp);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = exp[i]! / sum;
  return out;
}

// What child_bias_correction (select.ts) needs from a child node: whether
// design §A2's provenance fraction can be read at all (bias.py's "an opponent
// reply with no provenance" skip), without requiring the full MatchNode shape
// a reconstructed-from-fixture child does not have (only its rollup stats are
// exported, not its own edges).
export interface ChildStats {
  isTerminal: boolean;
  totalWeight(): number;
}

// MatchNode (node.py:303-412): design §S1's field list and nothing else.
// Arrays are sparse over the node's own legal set: `actions` holds the
// legal ids ascending, and every per-child array is indexed by position in
// `actions`, not by the 37341-wide action space.
export class MatchNode {
  toPlay: number;
  isTerminal: boolean;
  isRoot: boolean;
  actions: Int32Array;
  legalHash: LegalHash;
  logits: Float32Array;
  prior: Float32Array;
  wdlNet: Float64Array;
  visits: Int32Array;
  weight: Float64Array;
  // Flattened (width, WDL) row-major.
  wdlSum: Float64Array;
  // Flattened (width, MOMENT_SIZE) row-major.
  moment: Float64Array;
  biasKey: number;
  biasContribution: Float64Array;
  biasWeight: number;
  biasGen: number;
  children: Map<number, ChildStats>;

  constructor(opts: {
    toPlay: number;
    isTerminal: boolean;
    actions: Int32Array;
    logits: Float32Array;
    wdlNet: Float64Array;
    biasKey: number;
    biasGen?: number;
    isRoot?: boolean;
  }) {
    const ids = Int32Array.from(opts.actions);
    const width = ids.length;
    this.toPlay = opts.toPlay;
    this.isTerminal = opts.isTerminal;
    this.isRoot = opts.isRoot ?? false;
    this.actions = ids;
    this.legalHash = legalHashOf(ids);
    this.logits = Float32Array.from(opts.logits);
    this.prior = maskedPrior(this.logits);
    this.wdlNet = Float64Array.from(opts.wdlNet);
    this.visits = new Int32Array(width);
    this.weight = new Float64Array(width);
    this.wdlSum = new Float64Array(width * WDL);
    this.moment = new Float64Array(width * MOMENT_SIZE);
    this.biasKey = opts.biasKey;
    this.biasContribution = new Float64Array(WDL);
    this.biasWeight = 0.0;
    this.biasGen = opts.biasGen ?? 0;
    this.children = new Map();
  }

  get width(): number {
    return this.actions.length;
  }

  totalWeight(): number {
    return pairwiseSum(this.weight);
  }

  // Binary search: `actions` is ascending by construction.
  childIndex(action: number): number | null {
    let lo = 0;
    let hi = this.actions.length;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if (this.actions[mid]! < action) lo = mid + 1;
      else hi = mid;
    }
    if (lo >= this.width || this.actions[lo] !== action) return null;
    return lo;
  }

  subtreeWdlSum(): Float64Array {
    return sumRowsSequential(this.wdlSum, this.width, WDL);
  }
}
