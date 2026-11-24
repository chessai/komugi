// KOM-1168 (review F3, docs/reviews/2026-11-15-kom1162-ts-observation-v4-review.md):
// the hosts outside worker.ts that build their own ORT batches straight off
// a MatchState (tools/e2e_driver.mjs, tools/fast_match.mjs, bench/fast.ts)
// each hand-rolled `observation(state.game)`, dropping the state's own
// planeMapVersion and always encoding v3. The worker's own evaluators
// (evaluator.ts, ort_browser.ts) already encode with
// `observation(state.game, { planeMapVersion: state.planeMapVersion })`;
// this file is that one-liner, factored out so every host outside the
// worker calls it the same way.

import { observation } from './observation.ts';
import type { MatchState } from './search/state.ts';

export function encodeState(state: MatchState): Float32Array {
  return observation(state.game, { planeMapVersion: state.planeMapVersion });
}

// Concatenates `encodeState(states[i])` into row i of a
// `states.length * rowSize` buffer, in input order -- the shape every
// batched ORT call among these hosts already builds by hand.
export function encodeStatesBatch(states: readonly MatchState[], rowSize: number): Float32Array {
  const out = new Float32Array(states.length * rowSize);
  states.forEach((state, i) => out.set(encodeState(state), i * rowSize));
  return out;
}
