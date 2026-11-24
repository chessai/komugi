// Ports src/komugi/search/match/select.py: the PUCT descent (design §A1) and
// the LCB move choice (§A3) (docs/design/clientside-engine-port.md section
// 2.6). Every scalar here is derived from contempt-free storage at call
// time; nothing is cached on a node.

import { pairwiseSum, pairwiseSumF32 } from './pairwise.ts';
import {
  type MatchNode,
  WDL,
  biasKeyOf,
  edgeActionOf,
  quadraticForm,
  utility,
} from './node.ts';

// Every MatchConfig field select.ts reads (config.py). Matches the recorded
// search fixture's JSON field names (its `_config_json`) so a parsed
// search.json config object satisfies this type directly.
export interface MatchConfig {
  num_actions: number;
  contempt: number;
  dynamic_cpuct: boolean;
  c_puct: number;
  c_explore: number;
  c_log: number;
  c_base: number;
  sigma_prior: number;
  sigma_prior_weight: number;
  sigma_scale: number;
  sigma_floor: number;
  fpu_reduction: number;
  fpu_reduction_root: number;
  subtree_bias_factor: number;
  k_lcb: number;
  min_visit_prop_for_lcb: number;
  var_floor: number;
  variance_mode?: 'moment' | 'plugin';
}

// The table is an argument, never a global (design P-D3): `bias` undefined
// means "no table to consult" and makes every correction identically zero.
export interface BiasReader {
  read(key: number): Float64Array;
}

function requireChildren(node: MatchNode, what: string): void {
  if (node.width === 0) {
    throw new Error(
      `${what} on a node with no legal actions: a terminal position has no children to compare, ` +
        `and returning an index into an empty array would be a silent out-of-bounds`,
    );
  }
}

function clip(x: number, lo: number, hi: number): number {
  return Math.min(Math.max(x, lo), hi);
}

// _collapse (select.py:268-274): (L, WDL) triples -> (L,) utilities.
function collapse(triples: Float64Array, contempt: number): Float64Array {
  const n = triples.length / WDL;
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    out[i] = triples[i * WDL + 0]! - triples[i * WDL + 2]! + contempt * triples[i * WDL + 1]!;
  }
  return out;
}

// node_utility (select.py:64-82).
export function nodeUtility(node: MatchNode, config: MatchConfig): number {
  const total = node.totalWeight();
  if (total <= 0.0) return utility(node.wdlNet, config.contempt);
  return utility(node.subtreeWdlSum(), config.contempt) / total;
}

// node_sum_of_squares (select.py:85-111), clamped at zero against
// cancellation on a converged node.
export function nodeSumOfSquares(node: MatchNode, config: MatchConfig): number {
  const total = node.totalWeight();
  if (total <= 0.0) return 0.0;
  const mean = nodeUtility(node, config);
  const squares = pairwiseSum(quadraticForm(node.moment, config.contempt));
  return Math.max(squares - total * mean * mean, 0.0);
}

// node_sigma (select.py:114-129).
export function nodeSigma(node: MatchNode, config: MatchConfig): number {
  const total = node.totalWeight();
  const weighted =
    config.sigma_prior_weight * config.sigma_prior * config.sigma_prior + nodeSumOfSquares(node, config);
  const sigma = Math.sqrt(weighted / (config.sigma_prior_weight + total));
  return Math.max(sigma, config.sigma_floor);
}

// variance_scale (select.py:132-142).
export function varianceScale(node: MatchNode, config: MatchConfig): number {
  if (!config.dynamic_cpuct) return 1.0;
  const scale = config.sigma_scale;
  return (1.0 - scale) + scale * nodeSigma(node, config) / config.sigma_prior;
}

// exploration_constant (select.py:145-157): the one `math.log` call site --
// libm/V8 ulp risk, so this is a tolerance-compared intermediate (4.3 M3).
export function explorationConstant(node: MatchNode, config: MatchConfig): number {
  if (!config.dynamic_cpuct) return config.c_puct;
  const total = node.totalWeight();
  const growth = config.c_log * Math.log((total + config.c_base) / config.c_base);
  return (config.c_explore + growth) * varianceScale(node, config);
}

// child_bias_correction (select.py:160-228): lambda * b(key(a)) *
// direct-neural-fraction, per edge, flattened (L, WDL) row-major.
export function childBiasCorrection(
  node: MatchNode,
  config: MatchConfig,
  bias?: BiasReader,
): Float64Array {
  const out = new Float64Array(node.width * WDL);
  if (bias !== undefined) {
    const previousAction = edgeActionOf(node.biasKey, config.num_actions);
    for (let i = 0; i < node.width; i++) {
      const weight = node.weight[i]!;
      const child = node.children.get(i);
      if (weight <= 0.0 || child === undefined || child.isTerminal) continue;
      const fraction = clip((weight - child.totalWeight()) / weight, 0.0, 1.0);
      const key = biasKeyOf(node.toPlay, node.actions[i]!, previousAction, config.num_actions);
      const reading = bias.read(key);
      out[i * WDL + 0] = fraction * reading[0]!;
      out[i * WDL + 1] = fraction * reading[1]!;
      out[i * WDL + 2] = fraction * reading[2]!;
    }
  }
  for (let j = 0; j < out.length; j++) out[j] = config.subtree_bias_factor * out[j]!;
  return out;
}

// child_mean_wdl (select.py:231-246): S(a)/W(a) per edge, zero where unvisited.
export function childMeanWdl(node: MatchNode): Float64Array {
  const out = new Float64Array(node.width * WDL);
  for (let i = 0; i < node.width; i++) {
    const w = node.weight[i]!;
    if (w > 0.0) {
      out[i * WDL + 0] = node.wdlSum[i * WDL + 0]! / w;
      out[i * WDL + 1] = node.wdlSum[i * WDL + 1]! / w;
      out[i * WDL + 2] = node.wdlSum[i * WDL + 2]! / w;
    }
  }
  return out;
}

// corrected_child_wdl (select.py:249-265): raw mean plus the pooled bias
// correction, read time only.
export function correctedChildWdl(
  node: MatchNode,
  config: MatchConfig,
  bias?: BiasReader,
): Float64Array {
  const mean = childMeanWdl(node);
  const correction = childBiasCorrection(node, config, bias);
  const out = new Float64Array(mean.length);
  for (let i = 0; i < out.length; i++) out[i] = mean[i]! + correction[i]!;
  return out;
}

// child_raw_utilities (select.py:277-287): the UNCORRECTED utilities the
// variance is formed against.
export function childRawUtilities(node: MatchNode, config: MatchConfig): Float64Array {
  return collapse(childMeanWdl(node), config.contempt);
}

// first_play_urgency (select.py:313-328): design §A1's score for an edge
// with no weight. `explored` is a float32 pairwise sum over a gathered
// contiguous copy of prior[weight > 0] (4.3's reduction table).
export function firstPlayUrgency(node: MatchNode, config: MatchConfig): number {
  const reduction = node.isRoot ? config.fpu_reduction_root : config.fpu_reduction;
  let count = 0;
  for (let i = 0; i < node.width; i++) if (node.weight[i]! > 0.0) count++;
  const gathered = new Float32Array(count);
  let k = 0;
  for (let i = 0; i < node.width; i++) if (node.weight[i]! > 0.0) gathered[k++] = node.prior[i]!;
  const explored = pairwiseSumF32(gathered);
  return nodeUtility(node, config) - reduction * Math.sqrt(Math.max(explored, 0.0));
}

// child_utilities (select.py:290-310): clip(corrected utility, -1, 1) on a
// visited edge, first_play_urgency(node) -- a scalar, not per-edge -- on an
// unvisited one.
export function childUtilities(node: MatchNode, config: MatchConfig, bias?: BiasReader): Float64Array {
  const utilities = collapse(correctedChildWdl(node, config, bias), config.contempt);
  for (let i = 0; i < utilities.length; i++) utilities[i] = clip(utilities[i]!, -1.0, 1.0);
  const fpu = firstPlayUrgency(node, config);
  const out = new Float64Array(node.width);
  for (let i = 0; i < node.width; i++) out[i] = node.weight[i]! > 0.0 ? utilities[i]! : fpu;
  return out;
}

// puct_scores (select.py:331-355): U(a) + c_puct(n)*P(a)*sqrt(1+W_tot(n)) /
// (1+W(a)). The numerator is sqrt(1 + W_tot), not sqrt(W_tot) -- see
// select.py's own note on why that matters at a freshly expanded node.
export function puctScores(node: MatchNode, config: MatchConfig, bias?: BiasReader): Float64Array {
  requireChildren(node, 'puct_scores');
  const total = node.totalWeight();
  const ec = explorationConstant(node, config);
  const sq = Math.sqrt(1.0 + total);
  const utilities = childUtilities(node, config, bias);
  const out = new Float64Array(node.width);
  for (let i = 0; i < node.width; i++) {
    const exploration = ((ec * node.prior[i]!) * sq) / (1.0 + node.weight[i]!);
    out[i] = utilities[i]! + exploration;
  }
  return out;
}

// select_child (select.py:358-368): argmax of puct_scores, ties to the
// lowest action id (np.argmax semantics over an ascending `actions`).
export function selectChild(node: MatchNode, config: MatchConfig, bias?: BiasReader): number {
  const scores = puctScores(node, config, bias);
  let best = 0;
  for (let i = 1; i < scores.length; i++) if (scores[i]! > scores[best]!) best = i;
  return best;
}

// child_variances (select.py:371-408): var(a), floored, formed about the
// UNCORRECTED mean. No log/pow anywhere in this path, so it is exact.
export function childVariances(node: MatchNode, config: MatchConfig): Float64Array {
  const floor = config.var_floor * config.var_floor;
  const out = new Float64Array(node.width);
  if (config.variance_mode === 'plugin') {
    const means = childMeanWdl(node);
    const squaredContempt = config.contempt * config.contempt;
    const centre = collapse(means, config.contempt);
    for (let i = 0; i < node.width; i++) {
      const squares = means[i * WDL + 0]! + means[i * WDL + 2]! + squaredContempt * means[i * WDL + 1]!;
      const raw = squares - centre[i]! * centre[i]!;
      out[i] = node.weight[i]! > 0.0 ? Math.max(raw, floor) : floor;
    }
    return out;
  }
  const qf = quadraticForm(node.moment, config.contempt);
  const centre = childRawUtilities(node, config);
  for (let i = 0; i < node.width; i++) {
    const raw0 = node.weight[i]! > 0.0 ? qf[i]! / node.weight[i]! : 0.0;
    const raw = raw0 - centre[i]! * centre[i]!;
    out[i] = node.weight[i]! > 0.0 ? Math.max(raw, floor) : floor;
  }
  return out;
}

// lcb_eligible (select.py:411-422).
export function lcbEligible(node: MatchNode, config: MatchConfig): boolean[] {
  let best = 0.0;
  for (let i = 0; i < node.width; i++) if (node.weight[i]! > best) best = node.weight[i]!;
  const threshold = Math.max(2.0, config.min_visit_prop_for_lcb * best);
  const out: boolean[] = new Array(node.width);
  for (let i = 0; i < node.width; i++) out[i] = node.weight[i]! >= threshold;
  return out;
}

// lcb_scores (select.py:425-445): -Infinity on an unvisited edge, which can
// never win an eligibility-blind comparison.
export function lcbScores(node: MatchNode, config: MatchConfig, bias?: BiasReader): Float64Array {
  requireChildren(node, 'lcb_scores');
  const utilities = collapse(correctedChildWdl(node, config, bias), config.contempt);
  const variances = childVariances(node, config);
  const out = new Float64Array(node.width);
  for (let i = 0; i < node.width; i++) {
    if (node.weight[i]! > 0.0) {
      const spread = variances[i]! / node.weight[i]!;
      out[i] = utilities[i]! - config.k_lcb * Math.sqrt(spread);
    } else {
      out[i] = -Infinity;
    }
  }
  return out;
}

// select_action (select.py:448-543): the eligible edge with the greatest
// lcb(a); ties to greater W(a), then lower action id; plain max-weight when
// no edge is eligible. `allowed` narrows, never widens (KOM-233).
export function selectAction(
  node: MatchNode,
  config: MatchConfig,
  opts: { bias?: BiasReader; allowed?: boolean[] | Uint8Array } = {},
): number {
  requireChildren(node, 'select_action');
  const { bias, allowed } = opts;
  const weights = node.weight;
  let eligible = lcbEligible(node, config);

  if (allowed !== undefined) {
    if (allowed.length !== node.width) {
      throw new Error(
        `select_action: allowed must have length ${node.width}, got ${allowed.length}`,
      );
    }
    let anyPermitted = false;
    for (let i = 0; i < allowed.length; i++) {
      if (allowed[i]) {
        anyPermitted = true;
        break;
      }
    }
    if (!anyPermitted) {
      throw new Error('select_action: allowed permits no edge at all, which is not a move choice this function can make');
    }
    const combined: boolean[] = new Array(node.width);
    let anyEligible = false;
    for (let i = 0; i < node.width; i++) {
      combined[i] = eligible[i]! && Boolean(allowed[i]);
      if (combined[i]) anyEligible = true;
    }
    eligible = combined;
    if (!anyEligible) {
      const scores = lcbScores(node, config, bias);
      const priors = node.prior;
      let best = -1;
      for (let i = 0; i < node.width; i++) {
        if (allowed[i]) {
          best = i;
          break;
        }
      }
      for (let i = 0; i < node.width; i++) {
        if (!allowed[i]) continue;
        if (scores[i]! > scores[best]! || (scores[i]! === scores[best]! && priors[i]! > priors[best]!)) {
          best = i;
        }
      }
      return best;
    }
  }

  let anyEligible = false;
  for (let i = 0; i < node.width; i++) {
    if (eligible[i]) {
      anyEligible = true;
      break;
    }
  }
  if (anyEligible) {
    const scores = lcbScores(node, config, bias);
    let best = 0;
    for (let i = 0; i < node.width; i++) {
      if (!eligible[i]) continue;
      if (!eligible[best]) {
        best = i;
        continue;
      }
      if (scores[i]! > scores[best]! || (scores[i]! === scores[best]! && weights[i]! > weights[best]!)) {
        best = i;
      }
    }
    return best;
  }

  let best = 0;
  for (let i = 1; i < node.width; i++) if (weights[i]! > weights[best]!) best = i;
  return best;
}

// rank_root_moves: selectAction's own tiebreak, over every edge rather than
// just the winner. Eligible edges sort by (lcbScore desc, weight desc),
// ineligible by (weight desc), eligible always ahead of ineligible; ties
// keep their original (ascending action-index) order, matching the
// first-eligible-wins / first-max-weight-wins tiebreak selectAction uses.
export function rankRootMoves(node: MatchNode, config: MatchConfig, bias?: BiasReader): number[] {
  const eligible = lcbEligible(node, config);
  const scores = lcbScores(node, config, bias);
  const weights = node.weight;
  const order: number[] = new Array(node.width);
  for (let i = 0; i < node.width; i++) order[i] = i;
  order.sort((a, b) => {
    if (eligible[a] !== eligible[b]) return eligible[a] ? -1 : 1;
    if (eligible[a]!) {
      if (scores[a]! !== scores[b]!) return scores[a]! > scores[b]! ? -1 : 1;
    }
    if (weights[a]! !== weights[b]!) return weights[a]! > weights[b]! ? -1 : 1;
    return 0;
  });
  return order;
}
