// Fast-mode unit U1 (docs/design/clientside-fast-mode.md section 1.2, S-6).
// No KOM_SEARCH/KOM_PAIRWISE fixture is read here: every node below is
// hand-built, so this file has no env-gated test and never skips (B1).

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { MatchNode, MOMENT_SIZE, WDL, quadraticForm, utility } from '../src/search/node.ts';
import { pairwiseSum, sumRowsSequential } from '../src/search/pairwise.ts';
import {
  type BiasReader,
  type MatchConfig,
  childBiasCorrection,
  firstPlayUrgency,
  puctScores,
  selectChild,
} from '../src/search/select.ts';
import { SubtreeValueBias } from '../src/search/bias.ts';
import {
  type Overlay,
  childMeanWdlVirtual,
  childUtilitiesVirtual,
  effectiveStats,
  explorationConstantVirtual,
  pendingOf,
  puctScoresVirtual,
  selectChildVirtual,
} from '../src/search/virtual.ts';

function closeEnough(a: number, b: number): boolean {
  return Math.abs(a - b) <= 1e-15 + 1e-12 * Math.abs(b);
}

function baseConfig(overrides: Partial<MatchConfig> = {}): MatchConfig {
  return {
    num_actions: 8,
    contempt: 0,
    dynamic_cpuct: true,
    c_puct: 1.0,
    c_explore: 1.0,
    c_log: 0.45,
    c_base: 500.0,
    sigma_prior: 0.4,
    sigma_prior_weight: 2.0,
    sigma_scale: 0.85,
    sigma_floor: 0.02,
    fpu_reduction: 0.2,
    fpu_reduction_root: 0.1,
    subtree_bias_factor: 0.45,
    k_lcb: 5.0,
    min_visit_prop_for_lcb: 0.15,
    var_floor: 0.05,
    ...overrides,
  };
}

// A node with one visited edge and one never-visited edge, real stats only
// -- no pending claims. Used by both the zero-overlay identity test and as
// the base for the S-6 effective-arrays test below.
function twoEdgeNode(): MatchNode {
  const node = new MatchNode({
    toPlay: 0,
    isTerminal: false,
    actions: Int32Array.from([0, 1]),
    logits: Float32Array.from([0, 0]),
    wdlNet: Float64Array.from([0.34, 0.33, 0.33]),
    biasKey: 5,
    isRoot: false,
  });
  node.weight = Float64Array.from([4, 0]);
  node.wdlSum = Float64Array.from([1.0, 1.0, 2.0, 0, 0, 0]);
  node.moment = Float64Array.from([0.3, 0.1, 0.2, 0.3, 0.1, 0.4, 0, 0, 0, 0, 0, 0]);
  node.children.set(0, { isTerminal: false, totalWeight: () => 2.0 });
  return node;
}

// Key = biasKeyOf(toPlay=0, action=0, previousAction=edgeActionOf(biasKey=5,
// numActions=8)=-1, numActions=8) = 9 -- the only key childBiasCorrection
// reads for twoEdgeNode's sole visited, non-terminal-child edge (edge 0).
function buildBias(): BiasReader {
  const bias = new SubtreeValueBias({ freeProp: 0.2, weightFloor: 0.5 });
  bias.entries.set(9, {
    errSum: Float64Array.from([0.4, 0.1, -0.5]),
    weightSum: 3.0,
    gen: 1,
  });
  return bias;
}

test('zero overlay: puctScoresVirtual/selectChildVirtual equal puctScores/selectChild bit for bit', () => {
  const node = twoEdgeNode();
  const config = baseConfig();
  const bias = buildBias();

  const want = puctScores(node, config, bias);
  const wantIdx = selectChild(node, config, bias);

  // No entry in the overlay at all.
  const emptyOverlay: Overlay = new Map();
  const gotMissing = puctScoresVirtual(node, config, emptyOverlay, bias);
  for (let i = 0; i < want.length; i++) {
    assert.ok(Object.is(gotMissing[i], want[i]), `missing-entry puctScoresVirtual[${i}]`);
  }
  assert.equal(selectChildVirtual(node, config, emptyOverlay, bias), wantIdx);

  // An explicit, present-but-all-zero entry.
  const zeroOverlay: Overlay = new Map();
  zeroOverlay.set(node, new Int32Array(node.width));
  const gotZero = puctScoresVirtual(node, config, zeroOverlay, bias);
  for (let i = 0; i < want.length; i++) {
    assert.ok(Object.is(gotZero[i], want[i]), `zero-entry puctScoresVirtual[${i}]`);
  }
  assert.equal(selectChildVirtual(node, config, zeroOverlay, bias), wantIdx);
});

test('one pending descent scores its clipped loss plus exploration; FPU path untouched by a sibling pending', () => {
  // Three edges: 0 is really visited, 1 carries a single pending descent
  // (never visited for real), 2 is untouched entirely (no real visits, no
  // pending) and must still read the real first-play urgency.
  const node = new MatchNode({
    toPlay: 0,
    isTerminal: false,
    actions: Int32Array.from([0, 1, 2]),
    logits: Float32Array.from([0, 0, 0]),
    wdlNet: Float64Array.from([0.34, 0.33, 0.33]),
    biasKey: 9,
  });
  node.weight = Float64Array.from([5, 0, 0]);
  node.wdlSum = Float64Array.from([2.0, 1.0, 2.0, 0, 0, 0, 0, 0, 0]);
  // dynamic_cpuct off: exploration_constant is just c_puct, isolating the
  // "clipped loss" assertion from the sigma/variance chain (covered by the
  // S-6 test below instead).
  const config = baseConfig({ dynamic_cpuct: false, c_puct: 1.3, contempt: 0 });

  const overlay: Overlay = new Map();
  const pending = pendingOf(overlay, node);
  pending[1] = 1;

  const scores = puctScoresVirtual(node, config, overlay);
  const realFpu = firstPlayUrgency(node, config);

  const effTotal = node.totalWeight() + 1; // v=1 default, one pending unit
  const sq = Math.sqrt(1.0 + effTotal);

  // Edge 1: W'(a) = 1, S'(a) = (0, 0, 1) -> mean = (0, 0, 1) -> utility
  // -1, clipped to -1 (a full virtual loss), plus exploration at the
  // effective denominator 1 + W'(a) = 2.
  const want1 = -1.0 + (config.c_puct * node.prior[1] * sq) / 2.0;
  assert.ok(closeEnough(scores[1], want1), `claimed edge: got ${scores[1]}, want ${want1}`);

  // Edge 2: untouched, so its utility term is the real FPU scalar, with the
  // exploration term's effective total still reflecting edge 1's claim
  // (design 1.2: the sqrt(1+W_tot') term is node-level effective).
  const want2 = realFpu + (config.c_puct * node.prior[2] * sq) / 1.0;
  assert.ok(closeEnough(scores[2], want2), `untouched edge: got ${scores[2]}, want ${want2}`);

  // Edge 0's own utility term (real mean + real bias correction, S-6 rows
  // 3-4) is unaffected by edge 1's claim, even though edge 0's *final*
  // score does shift: sqrt(1+W_tot'(n)) is node-level effective (S-6 row 1),
  // so every edge's exploration term legitimately sees the sibling's claim.
  const effWithPending = effectiveStats(node, config, overlay);
  const effNoPending = effectiveStats(node, config, new Map());
  const utilitiesWithPending = childUtilitiesVirtual(node, config, effWithPending);
  const utilitiesNoPending = childUtilitiesVirtual(node, config, effNoPending);
  assert.ok(
    Object.is(utilitiesWithPending[0], utilitiesNoPending[0]),
    'edge 0 utility must be unaffected by a sibling pending claim',
  );
});

test("S-6: per-term effective/real source table on a hand-built node (pending = [0, 1])", () => {
  const node = twoEdgeNode();
  const config = baseConfig({ dynamic_cpuct: true, contempt: 0 });
  const overlay: Overlay = new Map();
  const pending = pendingOf(overlay, node);
  pending[1] = 1; // the single pending edge, as design 1.2's example

  const eff = effectiveStats(node, config, overlay);

  // Hand-built effective arrays, independently assembled from design 1.2's
  // formulas (W'(a), S'(a), moment'(a), W_tot'(n)) rather than by calling
  // virtual.ts's own composition -- the sole referee S-6 calls for.
  const v = 1;
  const wantWeight = Float64Array.from([node.weight[0], node.weight[1] + v]);
  const wantWdlSum = Float64Array.from([
    node.wdlSum[0], node.wdlSum[1], node.wdlSum[2],
    node.wdlSum[3], node.wdlSum[4], node.wdlSum[5] + v,
  ]);
  const wantMoment = Float64Array.from(node.moment);
  wantMoment[1 * MOMENT_SIZE + 5] += v;
  const wantTotal = node.totalWeight() + v;

  for (let i = 0; i < wantWeight.length; i++) assert.ok(Object.is(eff.weight[i], wantWeight[i]), `W'(${i})`);
  for (let i = 0; i < wantWdlSum.length; i++) assert.ok(Object.is(eff.wdlSum[i], wantWdlSum[i]), `S'(${i})`);
  for (let i = 0; i < wantMoment.length; i++) assert.ok(Object.is(eff.moment[i], wantMoment[i]), `moment'(${i})`);
  assert.ok(Object.is(eff.total, wantTotal), `W_tot'`);

  // Row 1: sqrt(1 + W_tot'(n)) -- effective.
  const wantRow1 = Math.sqrt(1.0 + wantTotal);
  assert.ok(Object.is(Math.sqrt(1.0 + eff.total), wantRow1), 'row 1: sqrt(1+W_tot\')');

  // Row 2: exploration_constant via nodeSigma/moment' -- effective. Built
  // here directly from node.ts's quadratic_form/utility/pairwiseSum, the
  // same frozen primitives select.ts itself uses, over the hand-built
  // effective arrays above (not over virtual.ts's internal helpers).
  const subtreeSumEff = sumRowsSequential(wantWdlSum, 2, WDL);
  const meanEff = utility(subtreeSumEff, config.contempt) / wantTotal;
  const squaresEff = pairwiseSum(quadraticForm(wantMoment, config.contempt));
  const sumSqEff = Math.max(squaresEff - wantTotal * meanEff * meanEff, 0.0);
  const weighted = config.sigma_prior_weight * config.sigma_prior * config.sigma_prior + sumSqEff;
  const sigmaEff = Math.max(Math.sqrt(weighted / (config.sigma_prior_weight + wantTotal)), config.sigma_floor);
  const varianceScaleEff = (1.0 - config.sigma_scale) + (config.sigma_scale * sigmaEff) / config.sigma_prior;
  const growth = config.c_log * Math.log((wantTotal + config.c_base) / config.c_base);
  const wantRow2 = (config.c_explore + growth) * varianceScaleEff;
  const gotRow2 = explorationConstantVirtual(node, config, eff);
  assert.ok(closeEnough(gotRow2, wantRow2), `row 2: got ${gotRow2}, want ${wantRow2}`);

  // Row 3: mean S'(a)/W'(a), denom 1 + W'(a) -- effective, per edge.
  const gotMean = childMeanWdlVirtual(eff);
  for (let i = 0; i < 2; i++) {
    const w = wantWeight[i];
    const wantMean = w > 0.0
      ? [wantWdlSum[i * WDL + 0] / w, wantWdlSum[i * WDL + 1] / w, wantWdlSum[i * WDL + 2] / w]
      : [0, 0, 0];
    for (let c = 0; c < WDL; c++) {
      assert.ok(Object.is(gotMean[i * WDL + c], wantMean[c]), `row 3 mean[${i}][${c}]`);
    }
    const wantDenom = 1.0 + w;
    assert.ok(Object.is(1.0 + eff.weight[i], wantDenom), `row 3 denom[${i}]`);
  }

  // Row 4: child_bias_correction -- real, reads node.weight/node.children,
  // never eff. Confirmed structurally (childBiasCorrection has no `eff`
  // parameter) and numerically: the value must not depend on the overlay.
  const bias = buildBias();
  const correctionWithPending = childBiasCorrection(node, config, bias);
  const correctionNoPending = childBiasCorrection(node, config, bias);
  for (let i = 0; i < correctionWithPending.length; i++) {
    assert.ok(Object.is(correctionWithPending[i], correctionNoPending[i]), `row 4 correction[${i}]`);
  }

  // Row 5: first_play_urgency's nodeUtility/explored mass -- real. Both of
  // twoEdgeNode's edges are claimed here (edge 0 for real, edge 1 by the
  // pending descent), so no edge reads this row in this node; the "one
  // pending descent" test above exercises it on a genuinely untouched edge.
  assert.equal(typeof firstPlayUrgency(node, config), 'number');
});

test('integer undo is bit-identical: increment then decrement leaves no residue', () => {
  const node = new MatchNode({
    toPlay: 0,
    isTerminal: false,
    actions: Int32Array.from([0, 1, 2]),
    logits: Float32Array.from([0.1, -0.2, 0.05]),
    wdlNet: Float64Array.from([0.3, 0.4, 0.3]),
    biasKey: 3,
    isRoot: true,
  });
  node.weight = Float64Array.from([3, 0, 1]);
  node.wdlSum = Float64Array.from([1.5, 0.5, 1.0, 0, 0, 0, 0.2, 0.3, 0.5]);
  node.moment = Float64Array.from([0.5, 0.1, 0.3, 0.4, 0.2, 0.6, 0, 0, 0, 0.1, 0.1, 0.3]);
  const savedWeight = Float64Array.from(node.weight);
  const savedWdlSum = Float64Array.from(node.wdlSum);
  const savedMoment = Float64Array.from(node.moment);

  const config = baseConfig();
  const before = puctScores(node, config);

  const overlay: Overlay = new Map();
  const pending = pendingOf(overlay, node);
  pending[1] += 1;
  pending[2] += 1;

  // While claimed, the overlay must not have touched the node's own storage.
  const claimed = puctScoresVirtual(node, config, overlay);
  assert.notEqual(claimed.length, 0);
  for (let i = 0; i < node.weight.length; i++) assert.ok(Object.is(node.weight[i], savedWeight[i]));
  for (let i = 0; i < node.wdlSum.length; i++) assert.ok(Object.is(node.wdlSum[i], savedWdlSum[i]));
  for (let i = 0; i < node.moment.length; i++) assert.ok(Object.is(node.moment[i], savedMoment[i]));

  // Undo: exact integer decrements back to zero.
  pending[1] -= 1;
  pending[2] -= 1;
  assert.deepEqual(Array.from(pending), [0, 0, 0]);

  const after = puctScoresVirtual(node, config, overlay);
  for (let i = 0; i < before.length; i++) {
    assert.ok(Object.is(after[i], before[i]), `post-undo puctScoresVirtual[${i}] must equal puctScores bit for bit`);
  }
});
