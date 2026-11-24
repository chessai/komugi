// Fast-mode unit U1 (docs/design/clientside-fast-mode.md section 1.2, S-6):
// the integer virtual-loss overlay and the effective-statistics PUCT scorer
// a batch collector (U2) reads while leaves are in flight. Per re-review
// F12, select.ts's `collapse`/`clip` are private, so they are reimplemented
// here rather than exporting a frozen file.

import { pairwiseSum, sumRowsSequential } from './pairwise.ts';
import { type MatchNode, MOMENT_SIZE, WDL, quadraticForm, utility } from './node.ts';
import {
  type BiasReader,
  type MatchConfig,
  childBiasCorrection,
  firstPlayUrgency,
  puctScores,
  selectChild,
} from './select.ts';

// design 1.2: `v` is the virtual-loss multiplicity, config `virtual_loss`,
// default 1 (both visit-count and value virtual loss, the lc0/KataGo
// convention). select.ts's MatchConfig has no such field, so it is a local
// extension rather than an edit to a frozen file.
export interface VirtualConfig extends MatchConfig {
  virtual_loss?: number;
}

export function virtualLossOf(config: VirtualConfig): number {
  return config.virtual_loss ?? 1;
}

// design 1.2: "The overlay lives in a side map `Map<MatchNode, Int32Array>`
// owned by the batch collector ..., not on `MatchNode`". U1 only defines the
// shape; U2 owns beginPlayout/completePlayout's increments/decrements.
export type Overlay = Map<MatchNode, Int32Array>;

// Collector convenience: callers never need to special-case "no entry yet".
// Lazily allocates a zeroed per-edge array on first touch.
export function pendingOf(overlay: Overlay, node: MatchNode): Int32Array {
  let pending = overlay.get(node);
  if (pending === undefined) {
    pending = new Int32Array(node.width);
    overlay.set(node, pending);
  }
  return pending;
}

function isAllZero(pending: Int32Array | undefined): boolean {
  if (pending === undefined) return true;
  for (let i = 0; i < pending.length; i++) if (pending[i] !== 0) return false;
  return true;
}

function requireChildren(node: MatchNode, what: string): void {
  if (node.width === 0) {
    throw new Error(`${what} on a node with no legal actions`);
  }
}

// select.py's private `collapse` (select.ts:60-67): (L, WDL) triples -> (L,)
// utilities. Re-implemented per F12, not imported.
function collapse(triples: Float64Array, contempt: number): Float64Array {
  const n = triples.length / WDL;
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    out[i] = triples[i * WDL + 0]! - triples[i * WDL + 2]! + contempt * triples[i * WDL + 1]!;
  }
  return out;
}

function clip(x: number, lo: number, hi: number): number {
  return Math.min(Math.max(x, lo), hi);
}

// design 1.2's effective arrays, built in a per-node scratch copy -- never
// written back to `node` -- only where a caller actually needs them (an
// all-zero overlay never reaches this). `weight`/`wdlSum`/`moment` are W'(a),
// S'(a) and moment'(a); `total` is W_tot'(n).
export interface EffectiveStats {
  readonly weight: Float64Array;
  readonly wdlSum: Float64Array;
  readonly moment: Float64Array;
  readonly total: number;
}

// design 1.2: W'(a) = W(a) + v*pending[a]; S'(a) = S(a) + v*pending[a]*(0,0,1)
// (a full loss in the parent mover's frame); moment'(a) = moment(a) +
// v*pending[a] on the `ll` slot only (packOuter((0,0,1)) is zero but on
// `ll`); W_tot'(n) = W_tot(n) + v*sum(pending). When every `pending[a]` is
// zero, `loss` is exactly 0 and every sum below is `x + 0.0`, which IEEE754
// returns bit-identical to `x` for every finite non-negative `x` here -- so
// this function need not special-case the zero overlay to stay bit-exact.
export function effectiveStats(node: MatchNode, config: VirtualConfig, overlay: Overlay): EffectiveStats {
  const width = node.width;
  const pending = overlay.get(node) ?? new Int32Array(width);
  const v = virtualLossOf(config);
  const weight = new Float64Array(width);
  const wdlSum = new Float64Array(width * WDL);
  const moment = Float64Array.from(node.moment);
  let total = node.totalWeight();
  for (let i = 0; i < width; i++) {
    const loss = v * pending[i]!;
    weight[i] = node.weight[i]! + loss;
    wdlSum[i * WDL + 0] = node.wdlSum[i * WDL + 0]!;
    wdlSum[i * WDL + 1] = node.wdlSum[i * WDL + 1]!;
    wdlSum[i * WDL + 2] = node.wdlSum[i * WDL + 2]! + loss;
    moment[i * MOMENT_SIZE + 5]! += loss;
    total += loss;
  }
  return { weight, wdlSum, moment, total };
}

// node_utility (select.ts:70-74), read over the effective rollup: S-6's "ec
// (via nodeSigma/moment')" row depends on this through nodeSumOfSquares.
export function nodeUtilityVirtual(node: MatchNode, config: MatchConfig, eff: EffectiveStats): number {
  if (eff.total <= 0.0) return utility(node.wdlNet, config.contempt);
  const subtreeSum = sumRowsSequential(eff.wdlSum, node.width, WDL);
  return utility(subtreeSum, config.contempt) / eff.total;
}

// node_sum_of_squares (select.ts:78-84), effective.
export function nodeSumOfSquaresVirtual(node: MatchNode, config: MatchConfig, eff: EffectiveStats): number {
  if (eff.total <= 0.0) return 0.0;
  const mean = nodeUtilityVirtual(node, config, eff);
  const squares = pairwiseSum(quadraticForm(eff.moment, config.contempt));
  return Math.max(squares - eff.total * mean * mean, 0.0);
}

// node_sigma (select.ts:87-93), effective.
export function nodeSigmaVirtual(node: MatchNode, config: MatchConfig, eff: EffectiveStats): number {
  const weighted =
    config.sigma_prior_weight * config.sigma_prior * config.sigma_prior + nodeSumOfSquaresVirtual(node, config, eff);
  const sigma = Math.sqrt(weighted / (config.sigma_prior_weight + eff.total));
  return Math.max(sigma, config.sigma_floor);
}

// variance_scale (select.ts:96-100), effective.
export function varianceScaleVirtual(node: MatchNode, config: MatchConfig, eff: EffectiveStats): number {
  if (!config.dynamic_cpuct) return 1.0;
  const scale = config.sigma_scale;
  return (1.0 - scale) + (scale * nodeSigmaVirtual(node, config, eff)) / config.sigma_prior;
}

// exploration_constant (select.ts:104-109), effective -- S-6 row 2: "dynamic
// c_puct sees the same loss the mean does".
export function explorationConstantVirtual(node: MatchNode, config: MatchConfig, eff: EffectiveStats): number {
  if (!config.dynamic_cpuct) return config.c_puct;
  const growth = config.c_log * Math.log((eff.total + config.c_base) / config.c_base);
  return (config.c_explore + growth) * varianceScaleVirtual(node, config, eff);
}

// child_mean_wdl (select.ts:138-149), effective -- S-6 row 3's mean half:
// S'(a)/W'(a) per edge, zero where W'(a) is not positive (an edge nobody has
// claimed, real or virtual).
export function childMeanWdlVirtual(eff: EffectiveStats): Float64Array {
  const width = eff.weight.length;
  const out = new Float64Array(width * WDL);
  for (let i = 0; i < width; i++) {
    const w = eff.weight[i]!;
    if (w > 0.0) {
      out[i * WDL + 0] = eff.wdlSum[i * WDL + 0]! / w;
      out[i * WDL + 1] = eff.wdlSum[i * WDL + 1]! / w;
      out[i * WDL + 2] = eff.wdlSum[i * WDL + 2]! / w;
    }
  }
  return out;
}

// child_utilities (select.ts:188-195), effective mean/denom plus REAL bias
// correction and REAL first-play urgency (design 1.2 "what the overlay does
// not touch"; S-6 rows 4 and 5): `childBiasCorrection` and `firstPlayUrgency`
// are called on `node` itself, never on `eff`, so they read only completed
// backups. An edge is "claimed" -- visited for real or claimed by a pending
// descent -- exactly when `eff.weight[i] > 0.0`, which collapses to the real
// `node.weight[i] > 0.0` test when every `pending[a]` is zero.
export function childUtilitiesVirtual(
  node: MatchNode,
  config: MatchConfig,
  eff: EffectiveStats,
  bias?: BiasReader,
): Float64Array {
  const correction = childBiasCorrection(node, config, bias);
  const mean = childMeanWdlVirtual(eff);
  const width = node.width;
  const out = new Float64Array(width);
  let fpu: number | undefined;
  for (let i = 0; i < width; i++) {
    if (eff.weight[i]! <= 0.0) {
      if (fpu === undefined) fpu = firstPlayUrgency(node, config);
      out[i] = fpu;
      continue;
    }
    const c0 = mean[i * WDL + 0]! + correction[i * WDL + 0]!;
    const c1 = mean[i * WDL + 1]! + correction[i * WDL + 1]!;
    const c2 = mean[i * WDL + 2]! + correction[i * WDL + 2]!;
    const raw = collapse(Float64Array.from([c0, c1, c2]), config.contempt)[0]!;
    out[i] = clip(raw, -1.0, 1.0);
  }
  return out;
}

// puct_scores (select.ts:200-212), effective. design 1.2: "`puctScoresVirtual`
// mirrors `puctScores`'s composition term for term (same association order)".
// At an all-zero overlay this calls `puctScores` directly (design 1.2: "the
// collector calls the existing `selectChild` directly, so an uncontended
// descent runs exactly today's statements"), which is bit-identical to this
// function's own effective computation on a zero overlay by construction
// (see `effectiveStats`), but avoids building scratch arrays for nothing.
export function puctScoresVirtual(
  node: MatchNode,
  config: VirtualConfig,
  overlay: Overlay,
  bias?: BiasReader,
): Float64Array {
  requireChildren(node, 'puctScoresVirtual');
  const pending = overlay.get(node);
  if (isAllZero(pending)) return puctScores(node, config, bias);
  const eff = effectiveStats(node, config, overlay);
  const ec = explorationConstantVirtual(node, config, eff);
  const sq = Math.sqrt(1.0 + eff.total);
  const utilities = childUtilitiesVirtual(node, config, eff, bias);
  const out = new Float64Array(node.width);
  for (let i = 0; i < node.width; i++) {
    const exploration = (ec * node.prior[i]! * sq) / (1.0 + eff.weight[i]!);
    out[i] = utilities[i]! + exploration;
  }
  return out;
}

// select_child (select.ts:216-221), effective, with the same argmax /
// lowest-index tie-break. The all-zero fallback (as above) mirrors design
// 1.2's collector behaviour exactly.
export function selectChildVirtual(
  node: MatchNode,
  config: VirtualConfig,
  overlay: Overlay,
  bias?: BiasReader,
): number {
  const pending = overlay.get(node);
  if (isAllZero(pending)) return selectChild(node, config, bias);
  const scores = puctScoresVirtual(node, config, overlay, bias);
  let best = 0;
  for (let i = 1; i < scores.length; i++) if (scores[i]! > scores[best]!) best = i;
  return best;
}
