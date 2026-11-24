// Fast-mode U7 (docs/design/clientside-fast-mode.md section 5; ticket
// KOM-1100): pondering. This module holds the pure pieces the design pins
// by name -- the free prediction (F9/S-5), the cache key (C2) and its
// store, and the cancellable driver over the public `searchBatchedGen`
// generator (C4) -- so that neither `tree.ts` nor `search.ts` needs an
// edit. `worker.ts` owns the wiring: cloning the game forward by two
// plies, building each speculative search's scratch tree/bias copy, and
// deciding when pondering is allowed to start at all.

import { ADVANCED_POSITION } from './constants.ts';
import { DEFAULT_PLANE_MAP_VERSION, type PlaneMapVersion } from './rule_features.ts';
import { DEFAULT_RULES_PROFILE, RULES_PRESETS, rulesConfigKey } from './rules_config.ts';
import type { MatchNode } from './search/node.ts';
import { searchBatchedGen } from './search/search.ts';
import type { MatchResult, SearchConfig, SearchOptions } from './search/search.ts';
import type { MatchState, WdlEval } from './search/state.ts';
import type { MatchTree } from './search/tree.ts';

// ---- F9/S-5: the free prediction, zero evaluator calls ---------------------

// predictReplies: the design's "argmax of visits (top K, default 1, max 3)
// under root.children.get(chosen) after the engine's own move" (ticket F9),
// read off the standing tree with no search and no evaluator call. Ties go
// to the lowest action id, the same convention selectChild/selectAction use
// (lowest index; `child.actions` is ascending, so lowest index == lowest
// action id). `root` must be the tree's root exactly as the just-finished
// search left it (before the next search's `tree.clear()`); `chosenAction`
// is that search's own `MatchResult.action`.
export function predictReplies(root: MatchNode, chosenAction: number, k: number): number[] {
  const chosenIndex = root.childIndex(chosenAction);
  if (chosenIndex === null) return [];
  // `children` is typed as `Map<number, ChildStats>` (node.ts): the search
  // package's own cast-back-to-MatchNode convention (e.g. tree.ts:326,
  // select.ts), since a retained child is always a real MatchNode, never a
  // reconstructed-from-fixture stand-in, inside this worker.
  const child = root.children.get(chosenIndex) as MatchNode | undefined;
  if (child === undefined || child.isTerminal || child.width === 0) return [];
  const indices = Array.from({ length: child.width }, (_, i) => i);
  indices.sort((a, b) => {
    const byVisits = child.visits[b]! - child.visits[a]!;
    if (byVisits !== 0) return byVisits;
    return child.actions[a]! - child.actions[b]!;
  });
  const top = Math.max(0, Math.min(k, child.width));
  return indices.slice(0, top).map((i) => child.actions[i]!);
}

// ---- C2: the ponder cache key -----------------------------------------------

const SHA256_K = Uint32Array.from([
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
  0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
  0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
  0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
  0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
  0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
  0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
  0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
]);

function rotr(x: number, n: number): number {
  return ((x >>> n) | (x << (32 - n))) >>> 0;
}

// A from-scratch SHA-256 (FIPS 180-4), synchronous and built only from
// `TextEncoder`/`DataView` (no `node:crypto`, no `crypto.subtle`): the cache
// key's history digest (C2) must be computable identically inside a browser
// Worker and under Node's `--test`, and WebCrypto's `subtle.digest` is
// async-only, which a synchronous key-equality check (PonderCache.get)
// cannot wait on. Not used for anything security-sensitive -- only to
// collapse an arbitrarily long history array into one key field.
export function sha256Hex(message: string): string {
  const bytes = new TextEncoder().encode(message);
  const bitLen = bytes.length * 8;
  const withOne = bytes.length + 1;
  const padded = Math.ceil((withOne + 8) / 64) * 64;
  const buf = new Uint8Array(padded);
  buf.set(bytes);
  buf[bytes.length] = 0x80;
  const view = new DataView(buf.buffer);
  // `bitLen` fits in 32 bits for any history this module ever hashes (a
  // JSON-stringified SAN list), so only the length field's low word is
  // ever written; the high word stays the zero `Uint8Array` already gave.
  view.setUint32(padded - 4, bitLen >>> 0, false);

  let h0 = 0x6a09e667;
  let h1 = 0xbb67ae85;
  let h2 = 0x3c6ef372;
  let h3 = 0xa54ff53a;
  let h4 = 0x510e527f;
  let h5 = 0x9b05688c;
  let h6 = 0x1f83d9ab;
  let h7 = 0x5be0cd19;

  const w = new Uint32Array(64);
  for (let offset = 0; offset < padded; offset += 64) {
    for (let i = 0; i < 16; i++) w[i] = view.getUint32(offset + i * 4, false);
    for (let i = 16; i < 64; i++) {
      const s0 = rotr(w[i - 15]!, 7) ^ rotr(w[i - 15]!, 18) ^ (w[i - 15]! >>> 3);
      const s1 = rotr(w[i - 2]!, 17) ^ rotr(w[i - 2]!, 19) ^ (w[i - 2]! >>> 10);
      w[i] = (w[i - 16]! + s0 + w[i - 7]! + s1) >>> 0;
    }

    let a = h0;
    let b = h1;
    let c = h2;
    let d = h3;
    let e = h4;
    let f = h5;
    let g = h6;
    let h = h7;
    for (let i = 0; i < 64; i++) {
      const bigS1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
      const ch = (e & f) ^ (~e & g);
      const temp1 = (h + bigS1 + ch + SHA256_K[i]! + w[i]!) >>> 0;
      const bigS0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
      const maj = (a & b) ^ (a & c) ^ (b & c);
      const temp2 = (bigS0 + maj) >>> 0;
      h = g;
      g = f;
      f = e;
      e = (d + temp1) >>> 0;
      d = c;
      c = b;
      b = a;
      a = (temp2 + temp1) >>> 0;
    }
    h0 = (h0 + a) >>> 0;
    h1 = (h1 + b) >>> 0;
    h2 = (h2 + c) >>> 0;
    h3 = (h3 + d) >>> 0;
    h4 = (h4 + e) >>> 0;
    h5 = (h5 + f) >>> 0;
    h6 = (h6 + g) >>> 0;
    h7 = (h7 + h) >>> 0;
  }

  return [h0, h1, h2, h3, h4, h5, h6, h7].map((x) => x.toString(16).padStart(8, '0')).join('');
}

// C2: key = (sha256(JSON.stringify(history)), fen, start, sims, batch,
// config identity, backend, modelSha256, maxMs, mainTableEpoch) -- every
// field the fix lists, folded into one comparable string. Folding
// `mainTableEpoch` into the key itself (rather than a side re-check at
// adoption) means "any main-tree search between ponder start and lookup" --
// which bumps the main tree's own `evaluations` counter -- is already a key
// mismatch, not an extra check a caller could forget. `start` (KOM-1104) is
// the EFFECTIVE start (the caller's own `request.start ?? ADVANCED_POSITION`
// normalization, root.ts's own default) so a ponder hit can never cross
// modes: two requests whose histories and fens agree but whose starts
// differ reach different positions once Beginner/Intro/Intermediate's
// smaller piece sets are in play, so they must never share a cache entry.
export interface PonderKeyInput {
  history: readonly string[];
  fen: string;
  // KOM-1104: the effective start the history replayed from. Optional so
  // every caller that predates this field (root.ts's own default) keys
  // exactly as if it had passed `ADVANCED_POSITION` explicitly.
  start?: string;
  // KOM-1105/KOM-1203 F2(c): the effective `rulesConfigKey(resolvedConfig)`
  // string the ponder root/request was built under -- never the bare
  // `RulesProfile`/`RulesConfig` itself (a plain `JSON.stringify` over an
  // arbitrary-order object would be key-order-dependent). Optional so every
  // caller that predates this field keys exactly as if it had passed
  // `rulesConfigKey(RULES_PRESETS[DEFAULT_RULES_PROFILE])` explicitly.
  rules?: string;
  // KOM-1162: the effective observation plane-map version. Optional so
  // every caller that predates this field keys exactly as if it had passed
  // DEFAULT_PLANE_MAP_VERSION explicitly.
  planeMapVersion?: PlaneMapVersion;
  sims: number;
  batch: number;
  configIdentity: string;
  backend: string | null;
  modelSha256: string | null;
  maxMs: number | null;
  mainTableEpoch: number;
  // KOM-1195 design section 6 (re-review N2): true iff the request/ctx this
  // ponder search ran under opted in (`target.stoppable`/`isOptedIn`). An
  // opted-in history play runs the chunked loop (a never-firing `shouldStop`
  // with no `maxMs`), whose reply can differ from the same request unpondered
  // on the continuous path -- folded in so a chunked entry is never matched
  // by an un-opted lookup, or vice versa. Optional so every caller that
  // predates this field keys exactly as if un-opted.
  chunked?: boolean;
}

export function ponderKey(input: PonderKeyInput): string {
  const parts: unknown[] = [
    sha256Hex(JSON.stringify(input.history)),
    input.fen,
    input.start ?? ADVANCED_POSITION,
    input.rules ?? rulesConfigKey(RULES_PRESETS[DEFAULT_RULES_PROFILE]),
    input.planeMapVersion ?? DEFAULT_PLANE_MAP_VERSION,
    input.sims,
    input.batch,
    input.configIdentity,
    input.backend,
    input.modelSha256,
    input.maxMs,
    input.mainTableEpoch,
  ];
  if (input.chunked === true) parts.push('chunked');
  return JSON.stringify(parts);
}

// ---- The cache itself --------------------------------------------------------

export interface PonderCacheEntry<TContext> {
  result: MatchResult;
  tree: MatchTree;
  context: TContext;
}

// A small keyed store, generic over whatever extra context the caller (here,
// worker.ts) needs to rebuild a reply and the next ponder session from a hit.
// `get` consumes (deletes) whatever is stored under `key`, hit or miss alike
// -- design 5.1's "a key match is necessary but not sufficient" means a used
// prediction never lingers to be matched twice.
export class PonderCache<TContext> {
  private entries = new Map<string, PonderCacheEntry<TContext>>();

  set(key: string, entry: PonderCacheEntry<TContext>): void {
    this.entries.set(key, entry);
  }

  get(key: string): PonderCacheEntry<TContext> | null {
    const entry = this.entries.get(key);
    this.entries.delete(key);
    return entry ?? null;
  }

  clear(): void {
    this.entries.clear();
  }

  get size(): number {
    return this.entries.size;
  }
}

// ---- C4: the abortable driver over the public generator --------------------

export class PonderAborted extends Error {}

export interface PonderSignal {
  aborted: boolean;
}

// C4 fix: U7's own abortable driver over `searchBatchedGen`, checking
// `signal.aborted` before every `gen.next()` call -- including the very
// first -- never mid-batch: an already-yielded batch's `Promise` is always
// awaited to completion first (design 5.3, "an in-flight session.run cannot
// be cancelled and is awaited regardless"). No edit to tree.ts/search.ts;
// this is the same discipline `runBatchGenAsync` (tree.ts) uses, plus the
// one check.
export async function runBatchedSearchAbortable(
  gen: Generator<Promise<WdlEval[]>, MatchResult, WdlEval[]>,
  signal: PonderSignal,
): Promise<MatchResult> {
  if (signal.aborted) throw new PonderAborted();
  let step = gen.next();
  while (!step.done) {
    const value = await step.value;
    if (signal.aborted) throw new PonderAborted();
    step = gen.next(value);
  }
  return step.value;
}

// One speculative full search (design 5.1 item 2, 5.2): `tree`/`state` are a
// scratch `MatchTree`/root state the caller built (its own deep-copied bias
// table, design 5.2), driven over the public `searchBatchedGen` and aborted
// per `signal`.
export function runPonderSearch(
  tree: MatchTree,
  state: MatchState,
  config: SearchConfig,
  opts: SearchOptions,
  batch: number,
  signal: PonderSignal,
): Promise<MatchResult> {
  return runBatchedSearchAbortable(searchBatchedGen(tree, state, config, opts, null, batch), signal);
}

// ---- M-6/F7: settings and gating --------------------------------------------

export interface PonderSettings {
  enabled: boolean;
  k: number;
}

export interface PonderGating {
  hidden: boolean;
  discharging: boolean;
}

// Ticket KOM-1100 F7 (overriding design 5.4's "default off" text): pondering
// is on by default, K = 1.
export const DEFAULT_PONDER_SETTINGS: PonderSettings = { enabled: true, k: 1 };
export const DEFAULT_PONDER_GATING: PonderGating = { hidden: false, discharging: false };

// Design 5.4: "up to 3" predicted replies.
export function clampPonderK(k: number): number {
  if (!Number.isFinite(k)) return DEFAULT_PONDER_SETTINGS.k;
  return Math.min(3, Math.max(1, Math.round(k)));
}

// Design 5.4: pondering is suspended while the tab is hidden or the page has
// told us the battery is discharging; a missing battery API leaves
// `discharging` at whatever it was last explicitly set to (never inferred).
export function shouldPonder(settings: PonderSettings, gating: PonderGating): boolean {
  return settings.enabled && !gating.hidden && !gating.discharging;
}
