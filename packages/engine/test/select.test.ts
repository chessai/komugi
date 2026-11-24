// U7a: node/select/bias/pairwise against the KOM-1060 search fixture
// (docs/design/clientside-engine-port.md section 4.3, section 6 U7a).
// KOM_SEARCH: the recorded search fixture's search.json. KOM_PAIRWISE: its
// --pairwise-vectors pw.json. Both are plain JSON (not JSONL) and are never
// committed, so they are read directly here rather than through
// test/fixture.ts's loadJsonl (which is for the KOM-1055 fixture format).

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { loadPairwiseFixture, loadSearchFixture } from './fixture.ts';
import {
  type BiasReader,
  type MatchConfig,
  childBiasCorrection,
  childMeanWdl,
  childVariances,
  explorationConstant,
  firstPlayUrgency,
  lcbEligible,
  nodeUtility,
  puctScores,
  selectAction,
} from '../src/search/select.ts';
import { SubtreeValueBias } from '../src/search/bias.ts';
import { MatchNode, WDL, legalHashOf, legalHashToBigInt } from '../src/search/node.ts';
import { pairwiseSum, pairwiseSumF32, sumRowsSequential } from '../src/search/pairwise.ts';

// Reconstructs the exact IEEE754 double a Python float.hex() string spells,
// via the bit pattern rather than decimal reassembly (which would reintroduce
// the rounding this test exists to rule out).
function parseFloatHex(spelling: string): number {
  if (/^-?0x0\.0p\+0$/.test(spelling)) return spelling.startsWith('-') ? -0 : 0;
  const m = /^(-?)0x1\.([0-9a-f]*)p([+-]\d+)$/.exec(spelling);
  if (!m) throw new Error(`unrecognized float hex: ${spelling}`);
  const negative = m[1] === '-';
  const fracHex = m[2].padEnd(13, '0').slice(0, 13);
  const highBits = parseInt(fracHex.slice(0, 5), 16);
  const lowBits = parseInt(fracHex.slice(5, 13), 16);
  const biasedExp = Number(m[3]) + 1023;
  const highWord = ((negative ? 0x80000000 : 0) | (biasedExp << 20) | highBits) >>> 0;
  const buf = new ArrayBuffer(8);
  const dv = new DataView(buf);
  dv.setUint32(0, highWord, false);
  dv.setUint32(4, lowBits >>> 0, false);
  return dv.getFloat64(0, false);
}

type F32Vector = { m: number | null; k: number; values: number[]; sum_hex: string };
type Wdl3Vector = { length: number; values: number[][]; sum_hex: string[] };
type F64Vector = { length: number; values: number[]; sum_hex: string };
type PairwiseFixture = {
  float32_pairwise: F32Vector[];
  float64_wdl_axis0: Wdl3Vector[];
  float64_pairwise: F64Vector[];
};

test('parseFloatHex round-trips known spellings', () => {
  assert.equal(parseFloatHex('0x0.0p+0'), 0);
  assert.equal(parseFloatHex('0x1.8000000000000p+0'), 1.5);
  assert.equal(parseFloatHex('0x1.0000000000000p+0'), 1);
  assert.equal(parseFloatHex('-0x1.0000000000000p+0'), -1);
});

test('pairwiseSum / pairwiseSumF32 / sumRowsSequential are bit-exact on pw.json (4.3 M1/M2)', () => {
  const pw = loadPairwiseFixture<PairwiseFixture>();

  let f32Checked = 0;
  for (const vec of pw.float32_pairwise) {
    const got = pairwiseSumF32(Float32Array.from(vec.values));
    const want = parseFloatHex(vec.sum_hex);
    assert.ok(
      Object.is(got, want),
      `float32_pairwise m=${vec.m} k=${vec.k}: got ${got}, want ${want}`,
    );
    f32Checked++;
  }
  assert.ok(f32Checked > 0, 'no float32_pairwise vectors');

  let wdlChecked = 0;
  for (const vec of pw.float64_wdl_axis0) {
    const flat = new Float64Array(vec.length * 3);
    for (let i = 0; i < vec.length; i++) {
      flat[i * 3 + 0] = vec.values[i][0];
      flat[i * 3 + 1] = vec.values[i][1];
      flat[i * 3 + 2] = vec.values[i][2];
    }
    const got = sumRowsSequential(flat, vec.length, 3);
    for (let col = 0; col < 3; col++) {
      const want = parseFloatHex(vec.sum_hex[col]);
      assert.ok(
        Object.is(got[col], want),
        `float64_wdl_axis0 length=${vec.length} col=${col}: got ${got[col]}, want ${want}`,
      );
    }
    wdlChecked++;
  }
  assert.ok(wdlChecked > 0, 'no float64_wdl_axis0 vectors');

  let f64Checked = 0;
  for (const vec of pw.float64_pairwise) {
    const got = pairwiseSum(Float64Array.from(vec.values));
    const want = parseFloatHex(vec.sum_hex);
    assert.ok(Object.is(got, want), `float64_pairwise length=${vec.length}: got ${got}, want ${want}`);
    f64Checked++;
  }
  assert.ok(f64Checked > 0, 'no float64_pairwise vectors');
});

test('legalHashOf matches the public FNV-1a 64 empty-input vector and a straightforward reference port', () => {
  // Public FNV-1a 64 test vector: hashing zero octets leaves the digest at
  // the offset basis 0xCBF29CE484222325, which as a signed int64 is
  // -3750763034362895579 -- an oracle independent of both ports below.
  assert.equal(legalHashToBigInt(legalHashOf(new Int32Array(0))), -3750763034362895579n);

  function referenceFnv1a64(actions: Int32Array): bigint {
    const bytes = new Uint8Array(actions.buffer, actions.byteOffset, actions.byteLength);
    let digest = 0xcbf29ce484222325n;
    const prime = 0x100000001b3n;
    const mask = (1n << 64n) - 1n;
    for (const byte of bytes) {
      digest = ((digest ^ BigInt(byte)) * prime) & mask;
    }
    return digest >= 1n << 63n ? digest - (1n << 64n) : digest;
  }

  const samples: Int32Array[] = [
    new Int32Array(0),
    Int32Array.from([0]),
    Int32Array.from([1, 2, 3]),
    Int32Array.from([0, 18000, 37340]),
    Int32Array.from([-1, 2147483647, -2147483648]),
  ];
  for (const sample of samples) {
    const got = legalHashToBigInt(legalHashOf(sample));
    const want = referenceFnv1a64(sample);
    assert.equal(got, want, `legalHashOf mismatch on ${Array.from(sample)}`);
  }
});

type SearchRecord = {
  config: MatchConfig & Record<string, unknown>;
  position: { to_play: number; is_root: boolean; bias_key: number };
  root: {
    actions: number[];
    prior: number[];
    logits: number[];
    visits: number[];
    weight: number[];
    wdl_sum: number[][];
    moment: number[][];
    wdl_net: number[];
    lcb_scores: (number | string)[];
    puct_scores: (number | string)[];
  };
  children: { index: number; action: number; total_weight: number; is_terminal: boolean }[];
  bias_entries: { key: number; err_sum: number[]; weight_sum: number; generation: number }[];
  intermediates: {
    exploration_constant: number;
    child_bias_correction: number[][];
    first_play_urgency: number;
    node_utility: number;
  };
};
type SearchFixture = { evaluator_vectors: unknown[]; searches: SearchRecord[] };

function toFinite(v: number | string): number {
  if (v === 'Infinity') return Infinity;
  if (v === '-Infinity') return -Infinity;
  if (v === 'NaN') return NaN;
  return v;
}

function closeEnough(a: number, b: number): boolean {
  return Math.abs(a - b) <= 1e-15 + 1e-12 * Math.abs(b);
}

function flatten2D(rows: number[][], cols: number): Float64Array {
  const out = new Float64Array(rows.length * cols);
  for (let i = 0; i < rows.length; i++) {
    for (let j = 0; j < cols; j++) out[i * cols + j] = rows[i][j];
  }
  return out;
}

function buildNode(rec: SearchRecord): MatchNode {
  const node = new MatchNode({
    toPlay: rec.position.to_play,
    isTerminal: false,
    actions: Int32Array.from(rec.root.actions),
    logits: Float32Array.from(rec.root.logits),
    wdlNet: Float64Array.from(rec.root.wdl_net),
    biasKey: rec.position.bias_key,
    isRoot: rec.position.is_root,
  });
  node.visits = Int32Array.from(rec.root.visits);
  node.weight = Float64Array.from(rec.root.weight);
  node.wdlSum = flatten2D(rec.root.wdl_sum, WDL);
  node.moment = flatten2D(rec.root.moment, 6);
  for (const child of rec.children) {
    node.children.set(child.index, {
      isTerminal: child.is_terminal,
      totalWeight: () => child.total_weight,
    });
  }
  return node;
}

function buildBias(rec: SearchRecord): BiasReader {
  const bias = new SubtreeValueBias({
    freeProp: Number(rec.config.bias_free_prop),
    weightFloor: Number(rec.config.bias_weight_floor),
  });
  for (const entry of rec.bias_entries) {
    bias.entries.set(entry.key, {
      errSum: Float64Array.from(entry.err_sum),
      weightSum: entry.weight_sum,
      gen: entry.generation,
    });
  }
  return bias;
}

// select_action (select.py:448-543), re-derived here directly from the
// fixture's own recorded lcb_scores/weight/prior rather than from this port's
// lcbScores -- an independent check of the eligibility/tie-break control flow
// alone, since the statistic-assembly correctness is checked separately.
function referenceSelectAction(weights: Float64Array, lcb: Float64Array, prior: Float32Array, config: MatchConfig): number {
  let best = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < weights.length; i++) if (weights[i] > best) best = weights[i];
  const threshold = Math.max(2.0, config.min_visit_prop_for_lcb * best);
  let anyEligible = false;
  for (let i = 0; i < weights.length; i++) if (weights[i] >= threshold) anyEligible = true;
  if (anyEligible) {
    let bestIdx = 0;
    let haveBest = false;
    for (let i = 0; i < weights.length; i++) {
      if (weights[i] < threshold) continue;
      if (!haveBest) {
        bestIdx = i;
        haveBest = true;
        continue;
      }
      if (lcb[i] > lcb[bestIdx] || (lcb[i] === lcb[bestIdx] && weights[i] > weights[bestIdx])) bestIdx = i;
    }
    return bestIdx;
  }
  let bestIdx = 0;
  for (let i = 1; i < weights.length; i++) if (weights[i] > weights[bestIdx]) bestIdx = i;
  return bestIdx;
}

test('node_utility, exploration_constant, child_bias_correction, first_play_urgency against the search fixture (4.3 M3)', () => {
  const fixture = loadSearchFixture<SearchFixture>();
  assert.ok(fixture.searches.length > 0, 'no search records');

  let priorChecked = 0;
  let utilityExactChecked = 0;
  let ecToleranceChecked = 0;
  let cbcToleranceChecked = 0;
  let fpuToleranceChecked = 0;

  for (const rec of fixture.searches) {
    const node = buildNode(rec);
    const bias = buildBias(rec);
    const config = rec.config;

    // masked_prior recomputed fresh from logits must equal the fixture's
    // root.prior exactly: a direct bit-exact test of pairwiseSum (float64
    // exp.sum()) feeding masked_prior.
    const wantPrior = Float32Array.from(rec.root.prior);
    assert.equal(node.prior.length, wantPrior.length);
    for (let i = 0; i < node.prior.length; i++) {
      assert.ok(Object.is(node.prior[i], wantPrior[i]), `${rec.position.bias_key}: prior[${i}]`);
    }
    priorChecked++;

    const gotUtility = nodeUtility(node, config);
    assert.ok(
      Object.is(gotUtility, rec.intermediates.node_utility),
      `node_utility: got ${gotUtility}, want ${rec.intermediates.node_utility}`,
    );
    utilityExactChecked++;

    const gotEc = explorationConstant(node, config);
    assert.ok(
      closeEnough(gotEc, rec.intermediates.exploration_constant),
      `exploration_constant: got ${gotEc}, want ${rec.intermediates.exploration_constant}`,
    );
    ecToleranceChecked++;

    const gotCbc = childBiasCorrection(node, config, bias);
    const wantCbc = flatten2D(rec.intermediates.child_bias_correction, WDL);
    for (let i = 0; i < gotCbc.length; i++) {
      assert.ok(closeEnough(gotCbc[i], wantCbc[i]), `child_bias_correction[${i}]`);
    }
    cbcToleranceChecked++;

    const gotFpu = firstPlayUrgency(node, config);
    assert.ok(
      closeEnough(gotFpu, rec.intermediates.first_play_urgency),
      `first_play_urgency: got ${gotFpu}, want ${rec.intermediates.first_play_urgency}`,
    );
    fpuToleranceChecked++;
  }

  console.log(
    `select intermediates: searches=${fixture.searches.length} prior=${priorChecked} ` +
      `utilityExact=${utilityExactChecked} ecTol=${ecToleranceChecked} cbcTol=${cbcToleranceChecked} fpuTol=${fpuToleranceChecked}`,
  );
});

test('puct_scores / lcb_scores recomputed from the recorded intermediates are bit-exact (4.3 M3, S11)', () => {
  const fixture = loadSearchFixture<SearchFixture>();

  let puctChecked = 0;
  let lcbChecked = 0;
  let selectActionChecked = 0;

  for (const rec of fixture.searches) {
    const node = buildNode(rec);
    const config = rec.config;
    const total = node.totalWeight();

    const recordedEc = rec.intermediates.exploration_constant;
    const recordedCbc = flatten2D(rec.intermediates.child_bias_correction, WDL);
    const recordedFpu = rec.intermediates.first_play_urgency;
    const meanWdl = childMeanWdl(node);
    const sq = Math.sqrt(1.0 + total);

    // puct_scores (select.py:331-355), substituting the recorded
    // exploration_constant/child_bias_correction/first_play_urgency for our
    // own log-derived versions so only exact (sqrt, +, -, *, /) arithmetic
    // remains between here and root.puct_scores.
    for (let i = 0; i < node.width; i++) {
      const corrected0 = meanWdl[i * WDL + 0] + recordedCbc[i * WDL + 0];
      const corrected1 = meanWdl[i * WDL + 1] + recordedCbc[i * WDL + 1];
      const corrected2 = meanWdl[i * WDL + 2] + recordedCbc[i * WDL + 2];
      const rawUtility = corrected0 - corrected2 + config.contempt * corrected1;
      const clipped = Math.min(Math.max(rawUtility, -1.0), 1.0);
      const utilityPart = node.weight[i] > 0.0 ? clipped : recordedFpu;
      const exploration = ((recordedEc * node.prior[i]) * sq) / (1.0 + node.weight[i]);
      const assembled = utilityPart + exploration;
      const want = toFinite(rec.root.puct_scores[i]);
      assert.ok(Object.is(assembled, want), `puct_scores[${i}]: got ${assembled}, want ${want}`);
    }
    puctChecked++;

    // lcb_scores (select.py:425-445): child_variances has no log/pow in it
    // (only quadratic_form, division and sqrt), so it is computed directly;
    // only the bias correction is substituted.
    const variances = childVariances(node, config);
    for (let i = 0; i < node.width; i++) {
      const want = toFinite(rec.root.lcb_scores[i]);
      if (node.weight[i] <= 0.0) {
        assert.ok(Object.is(want, -Infinity), `lcb_scores[${i}]: expected -Infinity on an unvisited edge`);
        continue;
      }
      const corrected0 = meanWdl[i * WDL + 0] + recordedCbc[i * WDL + 0];
      const corrected1 = meanWdl[i * WDL + 1] + recordedCbc[i * WDL + 1];
      const corrected2 = meanWdl[i * WDL + 2] + recordedCbc[i * WDL + 2];
      const rawUtility = corrected0 - corrected2 + config.contempt * corrected1;
      const spread = variances[i] / node.weight[i];
      const assembled = rawUtility - config.k_lcb * Math.sqrt(spread);
      assert.ok(Object.is(assembled, want), `lcb_scores[${i}]: got ${assembled}, want ${want}`);
    }
    lcbChecked++;

    // select_action's control flow, checked against the fixture's own
    // recorded lcb_scores/weight/prior (section above this comment already
    // established those are bit-exact against our own assembly).
    const lcbArr = Float64Array.from(rec.root.lcb_scores.map(toFinite));
    const wantIdx = referenceSelectAction(node.weight, lcbArr, node.prior, config);
    const bias = buildBias(rec);
    const gotIdx = selectAction(node, config, { bias });
    assert.equal(gotIdx, wantIdx, `select_action index mismatch at bias_key=${rec.position.bias_key}`);
    selectActionChecked++;
  }

  console.log(
    `select scores: searches=${fixture.searches.length} puct=${puctChecked} lcb=${lcbChecked} ` +
      `selectAction=${selectActionChecked}`,
  );
  assert.ok(puctChecked > 0 && lcbChecked > 0 && selectActionChecked > 0, 'no search records checked');
});

test('selectAction: a terminal (zero-width) node raises; an exhausted allowed mask falls back to (lcb, prior)', () => {
  const emptyNode = new MatchNode({
    toPlay: 0,
    isTerminal: true,
    actions: new Int32Array(0),
    logits: new Float32Array(0),
    wdlNet: Float64Array.from([1, 0, 0]),
    biasKey: 0,
  });
  assert.throws(() => selectAction(emptyNode, baseConfig()));

  // Three actions, none visited enough to be LCB-eligible (weight < 2): the
  // allowed intersection with eligibility is empty, so the fallback picks
  // the permitted edge with the greatest (lcb, prior) -- here action 1,
  // whose win-heavy backed-up WDL gives it a higher lcb than action 2's
  // loss-heavy one, even though both have the same weight.
  const node = new MatchNode({
    toPlay: 0,
    isTerminal: false,
    actions: Int32Array.from([0, 1, 2]),
    logits: Float32Array.from([0, 0, 0]),
    wdlNet: Float64Array.from([0, 1, 0]),
    biasKey: 0,
    isRoot: true,
  });
  node.weight = Float64Array.from([0, 1, 1]);
  node.wdlSum = Float64Array.from([0, 0, 0, 0.1, 0.9, 0, -0.9, 0.1, 0]);
  const eligible = lcbEligible(node, baseConfig());
  assert.deepEqual(eligible, [false, false, false]);

  const idx = selectAction(node, baseConfig(), { allowed: [false, true, true] });
  assert.equal(idx, 1);

  assert.throws(() => selectAction(node, baseConfig(), { allowed: [false, false, false] }));
  assert.throws(() => selectAction(node, baseConfig(), { allowed: [true] }));
});

function baseConfig(): MatchConfig {
  return {
    num_actions: 37341,
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
  };
}
