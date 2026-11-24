// The deterministic fake evaluator both languages implement from spec
// (docs/design/clientside-engine-port.md section 4.3), ported from
// the fixture exporter's fake_logits/fake_wdl/
// fake_moves_left/fnv1a32. Shared with Python by specification, not by
// import: every output is dyadic and exact in float32 and float64.

import type { MlhLeafEval, MlhLeafEvaluator, MlhLeafEvaluatorAsync } from './search/mlh.ts';
import type { MatchState, WdlEval } from './search/state.ts';

const FNV32_OFFSET = 0x811c9dc5;
const FNV32_PRIME = 0x01000193;

// fnv1a32 (the fixture exporter's version): FNV-1a 32 over `data`, kept
// unsigned throughout (>>> 0 after every step) so later shifts and mod see
// the same nonnegative integer Python's unbounded int already is.
export function fnv1a32(data: Uint8Array): number {
  let h = FNV32_OFFSET;
  for (let i = 0; i < data.length; i++) {
    h = (h ^ data[i]!) >>> 0;
    h = Math.imul(h, FNV32_PRIME) >>> 0;
  }
  return h;
}

// fake_logits' per-action term (the fixture exporter's version, design
// 4.3): 0 on the selected subset, else -1e4. `a * 40503` stays below 2**31
// for every action id (design's own note), so the XOR sees it as the
// nonnegative int32 it already is; `>>> 0` before `%` is S3's unsigned rule.
export function fakeLogitAt(h: number, action: number): number {
  return (((action * 40503) ^ h) >>> 0) % 7 === 0 ? 0 : -1e4;
}

export function fakeLogits(h: number, numActions: number): Float32Array {
  const out = new Float32Array(numActions);
  for (let a = 0; a < numActions; a++) out[a] = fakeLogitAt(h, a);
  return out;
}

// fake_wdl (the fixture exporter's version, design 4.3): dyadic (w, d, l),
// at most 16 significant bits, exact in float32 and float64.
export function fakeWdl(h: number): Float64Array {
  const w = ((h >>> 3) & 255) / 256;
  const l = (((h >>> 11) & 255) / 256) * (1 - w);
  const d = 1 - w - l;
  return Float64Array.from([w, d, l]);
}

// fake_moves_left (the fixture exporter's version, design 4.3).
export function fakeMovesLeft(h: number): number {
  return ((h >>> 19) & 255) / 256;
}

// The WdlEvaluator seam (state.ts's WdlEvaluator): `h` is computed directly
// from the position's placement-key bytes (MatchState.placementKeyBytes),
// the same 243 cell bytes Python's exporter recovers by inverting the
// observation planes (the fixture exporter's module docstring) -- that
// inversion is an artifact of the real seam's (obs, legal) signature, not
// part of design 4.3's definition of `h`, so this port reads the bytes
// directly rather than depending on observation.ts (U5, not a U7b1
// dependency).
export function makeFakeEvaluator(numActions: number): (state: MatchState) => WdlEval {
  return (state: MatchState): WdlEval => {
    const h = fnv1a32(state.placementKeyBytes());
    return { policyLogits: fakeLogits(h, numActions), wdl: fakeWdl(h) };
  };
}

// The fake evaluator extended with the moves-left head search/mlh.ts's
// wrapper needs -- the same combinator test/search.test.ts already builds
// locally (makeFakeMlhLeaf) for U7b2's own fixture test; given a home here
// (moved from evaluator.ts, design doc ts-engine-library.md section 4.1a)
// since worker.ts's default/test leaf evaluator is this exact shape.
export function makeFakeMlhEvaluator(numActions: number): MlhLeafEvaluator {
  return (state: MatchState): MlhLeafEval => {
    const h = fnv1a32(state.placementKeyBytes());
    return { policyLogits: fakeLogits(h, numActions), wdl: fakeWdl(h), movesLeft: fakeMovesLeft(h) };
  };
}

// KOM-1078: the same fake leaf, but genuinely asynchronous (a resolved
// microtask away, never synchronous) -- what async.test.ts drives through
// searchAsync/handleAsync to pin that a real suspend-and-resume evaluator
// reaches exactly the fake evaluator's statistics the sync path already
// does, with no replay and no shortcuts.
export function makeFakeMlhEvaluatorAsync(numActions: number): MlhLeafEvaluatorAsync {
  const sync = makeFakeMlhEvaluator(numActions);
  return (state: MatchState): Promise<MlhLeafEval> => Promise.resolve(sync(state));
}
