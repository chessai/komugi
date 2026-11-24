// Ports the MLH leaf seam (docs/design/clientside-engine-port.md section 2.6,
// section 6 U7b2):
//   - MlhUtility / mlh_utility_term (src/komugi/selfplay/adapter.py:540-681);
//   - shift_wdl and the per-search evaluator wrapper's _Ctx / make_evaluator_mlh
//     (the Python MLH search wrapper), as reparameterised by
//     the fixture exporter's _MlhCtx / _make_mlh_evaluator
//     (a plain argument instead of an environment read, same arithmetic).
// MAX_TURNS (observation.py:70) is game.ts's MaxTurns, reused rather than
// redeclared.

import { MaxTurns } from '../game.ts';
import type { MaybeAsyncWdlEvaluator } from './tree.ts';
import type { MatchState, WdlEval, WdlEvaluator } from './state.ts';

const WDL = 3;

// MlhUtility's four numbers (adapter.py:540-567). `slope` here is
// `slope_plies`; renamed to match the fixture exporter's `config.mlh`
// JSON object (`{lam, tau0, tau1, slope}`).
export interface MlhConfig {
  lam: number;
  tau0: number;
  tau1: number;
  slope: number;
}

// mlh_utility_term (adapter.py:625-681): lambda * g(|v|) * sign(v) * f(m),
// for one leaf. `wdl` is (w, d, l) in the mover's frame; only w and l are
// read (the gate is contempt-free, v = w - l).
export function mlhUtilityTerm(
  wdl: Float64Array,
  movesLeft: number,
  mRootPlies: number,
  config: MlhConfig,
): number {
  const v = wdl[0]! - wdl[2]!;
  const magnitude = Math.abs(v);
  const ramp = (magnitude - config.tau0) / (config.tau1 - config.tau0);
  const gate = magnitude * Math.min(Math.max(ramp, 0.0), 1.0);
  if (gate === 0.0) {
    // Exactly zero, returned before moves_left is ever read (adapter.py:667-671):
    // "below tau0 the term does not exist" is an identity, not a product that
    // happens to round to nothing.
    return 0.0;
  }
  const plies = MaxTurns * Math.min(Math.max(movesLeft, 0.0), 1.0);
  const centred = Math.min(Math.max((mRootPlies - plies) / config.slope, -1.0), 1.0);
  // copysign(1.0, v), written as a branch (adapter.py:677-680): v === 0.0 is
  // unreachable once gate !== 0.0 (magnitude > 0 implies v !== 0), kept
  // anyway for the identical association the Python source keeps.
  const sign = v === 0.0 ? 0.0 : Math.sign(v);
  return config.lam * gate * sign * centred;
}

// shift_wdl (the Python MLH search wrapper): move mass from l to w (or w to l)
// first, then d, so that w - l becomes clip(w - l + term, -1, 1) while the
// triple stays a distribution.
export function shiftWdl(wdl: Float64Array, term: number): Float64Array {
  let w = wdl[0]!;
  let d = wdl[1]!;
  let l = wdl[2]!;
  const v = w - l;
  const target = Math.min(Math.max(v + term, -1.0), 1.0);
  const delta = target - v;
  if (delta > 0.0) {
    const take = Math.min(delta / 2.0, l);
    l -= take;
    w += take;
    const rest = Math.min(delta - 2.0 * take, d);
    d -= rest;
    w += rest;
  } else if (delta < 0.0) {
    const need = -delta;
    const take = Math.min(need / 2.0, w);
    w -= take;
    l += take;
    const rest = Math.min(need - 2.0 * take, d);
    d -= rest;
    l += rest;
  }
  const out = Float64Array.from([Math.max(w, 0.0), Math.max(d, 0.0), Math.max(l, 0.0)]);
  // out.sum(): a 3-element numpy sum, left-to-right (design 4.3's reduction
  // table: "3 elements: plain left-to-right loop").
  let total = 0.0;
  for (let i = 0; i < WDL; i++) total += out[i]!;
  for (let i = 0; i < WDL; i++) out[i]! /= total;
  return out;
}

// _Ctx / _MlhCtx (the Python MLH search wrapper and the fixture exporter):
// the per-search moves-left reference and the gate count. One of these is
// created per search and reset immediately before it, by the caller -- the
// same lifecycle `ctx.reset()` has in both Python callers, and not this
// module's to impose (search.ts calls no evaluator directly).
export class MlhContext {
  mRoot: number | null = null;
  gated = 0;

  reset(): void {
    this.mRoot = null;
    this.gated = 0;
  }
}

// What the wrapped evaluator needs beyond state.ts's WdlEval: the moves-left
// head's raw output. state.ts's WdlEvaluator (U7b1, not edited here) has no
// such field -- nothing in U7b1 reads it -- so this is a leaf-evaluator shape
// local to the MLH seam, analogous to the real ORT evaluator's `movesLeft`
// (design section 2.7, U8, not yet built).
export type MlhLeafEval = { policyLogits: Float32Array; wdl: Float64Array; movesLeft: number };
export type MlhLeafEvaluator = (state: MatchState) => MlhLeafEval;

// KOM-1078: the async-leaf-evaluator shape, for a real per-leaf-async
// evaluator (onnxruntime-web). Mirrors MlhLeafEvaluator exactly, apart from
// the return type.
export type MlhLeafEvaluatorAsync = (state: MatchState) => MlhLeafEval | Promise<MlhLeafEval>;

// The arithmetic make_evaluator_mlh / _make_mlh_evaluator apply to one leaf
// (the Python MLH search wrapper and the fixture exporter), factored
// out so wrapMlhEvaluator and wrapMlhEvaluatorAsync run exactly the same
// statements in the exact same order on the leaf's (policyLogits, wdl,
// movesLeft) once it is in hand -- the two differ only in how that leaf was
// obtained (now, or awaited). lam <= 0 is stock behaviour -- the term is
// never computed and `ctx.mRoot` never read. The first evaluation after a
// reset sets `ctx.mRoot` and contributes no term (the match search
// evaluates the root first, tree.ts's ensureRoot); every later leaf is
// shifted by mlh_utility_term.
function applyMlh(
  policyLogits: Float32Array,
  wdl: Float64Array,
  movesLeft: number,
  config: MlhConfig,
  ctx: MlhContext,
): WdlEval {
  if (config.lam <= 0.0) return { policyLogits, wdl };
  if (ctx.mRoot === null) {
    ctx.mRoot = MaxTurns * Math.min(Math.max(movesLeft, 0.0), 1.0);
    return { policyLogits, wdl };
  }
  const term = mlhUtilityTerm(wdl, movesLeft, ctx.mRoot, config);
  if (term !== 0.0) {
    ctx.gated += 1;
    return { policyLogits, wdl: shiftWdl(wdl, term) };
  }
  return { policyLogits, wdl };
}

// make_evaluator_mlh / _make_mlh_evaluator, wrapping `inner` into a plain
// WdlEvaluator (state.ts's seam, what MatchTree is built with).
export function wrapMlhEvaluator(
  inner: MlhLeafEvaluator,
  config: MlhConfig,
  ctx: MlhContext,
): WdlEvaluator {
  return (state: MatchState): WdlEval => {
    const { policyLogits, wdl, movesLeft } = inner(state);
    return applyMlh(policyLogits, wdl, movesLeft, config, ctx);
  };
}

// KOM-1078: the same wrapper over an async leaf evaluator, into a
// MaybeAsyncWdlEvaluator (tree.ts's resumable seam). `inner` may answer now
// or later; applyMlh's statements run identically either way, only ever
// after the leaf is actually in hand.
export function wrapMlhEvaluatorAsync(
  inner: MlhLeafEvaluatorAsync,
  config: MlhConfig,
  ctx: MlhContext,
): MaybeAsyncWdlEvaluator {
  return (state: MatchState): WdlEval | Promise<WdlEval> => {
    const maybe = inner(state);
    if (maybe instanceof Promise) {
      return maybe.then(({ policyLogits, wdl, movesLeft }) => applyMlh(policyLogits, wdl, movesLeft, config, ctx));
    }
    const { policyLogits, wdl, movesLeft } = maybe;
    return applyMlh(policyLogits, wdl, movesLeft, config, ctx);
  };
}
