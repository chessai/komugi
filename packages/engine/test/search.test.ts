// U7b2: search loop / root win / refutation survey / three-way rule / MLH
// against the KOM-1060 search fixture (docs/design/clientside-engine-port.md
// section 4.3, section 6 U7b2). KOM_SEARCH: the recorded search fixture's
// search.json (plain JSON, read directly, as tree.test.ts/select.test.ts do).

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { fakeLogits, fakeMovesLeft, fakeWdl, fnv1a32 } from '../src/fake_evaluator.ts';
import { boardFromFen } from '../src/fen.ts';
import { Game } from '../src/game.ts';
import { RepetitionRule } from '../src/repetition.ts';
import { mlhUtilityTerm, shiftWdl, wrapMlhEvaluator, MlhContext, type MlhConfig, type MlhLeafEval } from '../src/search/mlh.ts';
import { search, type SearchConfig } from '../src/search/search.ts';
import { MatchGameState } from '../src/search/state.ts';
import { MatchTree } from '../src/search/tree.ts';
import { loadSearchFixture } from './fixture.ts';

function toFinite(v: number | string): number {
  if (v === 'Infinity') return Infinity;
  if (v === '-Infinity') return -Infinity;
  if (v === 'NaN') return NaN;
  return v;
}

function flatten2D(rows: number[][], cols: number): Float64Array {
  const out = new Float64Array(rows.length * cols);
  for (let i = 0; i < rows.length; i++) {
    for (let j = 0; j < cols; j++) out[i * cols + j] = rows[i][j];
  }
  return out;
}

type TacticalRecord = {
  root_witness: number | null;
  refuted: number[];
  overridden: number | null;
  provider_calls: number;
  candidate_transitions: number;
  skipped_reason: string | null;
  truncated: boolean;
} | null;

type SearchRecord = {
  config: SearchConfig & {
    discovery: boolean;
    draftmin: number;
    mlh: { lam: number; tau0: number; tau1: number; slope: number };
  };
  position: { fen: string; history: string[]; stamp: string; to_play: number; is_root: boolean; bias_key: number };
  result: {
    action: number;
    playouts_used: number;
    evaluations: number;
    nodes: number;
    mlh_gated: number;
    tactical: TacticalRecord;
  };
  root: {
    actions: number[];
    visits: number[];
    weight: number[];
    wdl_sum: number[][];
    moment: number[][];
    wdl_net: number[];
    lcb_scores: (number | string)[];
  };
  intermediates: { node_utility: number };
};

type SearchFixture = { evaluator_vectors: unknown[]; searches: SearchRecord[] };

function loadFixture(): SearchFixture {
  return loadSearchFixture<SearchFixture>();
}

// Builds the root MatchGameState straight from the record's FEN, exactly as
// tree.test.ts's rootStateFor does (root reconstruction by SAN replay is
// root.ts's job, U8, which this unit does not depend on -- design section
// 2.5).
function rootStateFor(rec: SearchRecord): MatchGameState {
  const board = boardFromFen(rec.position.fen);
  const rule = rec.position.stamp === 'present' ? RepetitionRule.PLACEMENT : RepetitionRule.FULL;
  const game = new Game(board, rule);
  return new MatchGameState(game, { minPlacements: rec.config.draftmin, isRoot: true });
}

// The fake evaluator (fake_evaluator.ts, reused by import) extended with the
// moves-left head mlh.ts's wrapper needs: same `h`, same three functions,
// just read together instead of through makeFakeEvaluator's narrower
// WdlEval-only return.
function makeFakeMlhLeaf(numActions: number): (state: MatchGameState) => MlhLeafEval {
  return (state: MatchGameState): MlhLeafEval => {
    const h = fnv1a32(state.placementKeyBytes());
    return { policyLogits: fakeLogits(h, numActions), wdl: fakeWdl(h), movesLeft: fakeMovesLeft(h) };
  };
}

test('search() reproduces exact fixture results, production tuple first (4.3 comparison rule)', () => {
  const fixture = loadFixture();
  assert.ok(fixture.searches.length > 0, 'no search records in the fixture');
  const isProduction = (rec: SearchRecord): boolean =>
    rec.config.discovery === true && rec.config.draftmin > 0 && rec.config.mlh.lam > 0;
  assert.ok(fixture.searches.some(isProduction), 'no production-tuple records in the fixture');

  // the fixture exporter's generate_searches() builds ONE server --
  // hence one MatchTree and one subtree-value-bias table -- per (spec, sims)
  // group, and runs every position of that group through it sequentially
  // (_build_server once, then a plain `for position in positions` loop).
  // MatchSearch.search() clears the TREE itself on every call (tree_reuse is
  // off), but the bias table is deliberately never cleared (bias.ts/tree.ts:
  // "keyed on (mover, move, previous move), not on any node") -- so a bias
  // entry a position's own root never gets enough weight to populate can
  // still read nonzero if an EARLIER position in the same group populated
  // the same (mover, action, previous action) key. The fixture is grouped
  // contiguously by (playouts, discovery, draftmin, mlh): a tree/evaluator
  // is created on the first record of a group and reused -- not recreated
  // per record -- for exactly as long as the key stays the same.
  function groupKeyOf(rec: SearchRecord): string {
    return [
      rec.config.playouts,
      rec.config.discovery,
      rec.config.draftmin,
      rec.config.mlh.lam,
      rec.config.mlh.tau0,
      rec.config.mlh.tau1,
      rec.config.mlh.slope,
    ].join('|');
  }

  let checked = 0;
  let mlhGatedSeen = 0;
  let draftSkipSeen = 0;
  let groupKey: string | null = null;
  let tree: MatchTree | null = null;
  let ctx: MlhContext | null = null;

  for (const rec of fixture.searches) {
    const key = groupKeyOf(rec);
    if (key !== groupKey) {
      groupKey = key;
      const leaf = makeFakeMlhLeaf(rec.config.num_actions);
      ctx = new MlhContext();
      const mlhConfig: MlhConfig = {
        lam: rec.config.mlh.lam,
        tau0: rec.config.mlh.tau0,
        tau1: rec.config.mlh.tau1,
        slope: rec.config.mlh.slope,
      };
      const evaluator = wrapMlhEvaluator(leaf, mlhConfig, ctx);
      tree = new MatchTree(rec.config, evaluator);
    } else {
      // Reset before every search (the Python MLH search wrapper's own
      // lifecycle, the fixture exporter's `_run_search`): the first evaluation
      // after this sets ctx.mRoot fresh for THIS position's root.
      ctx!.reset();
    }

    const state = rootStateFor(rec);
    const result = search(tree!, state, rec.config, { discovery: rec.config.discovery });

    const where = `${rec.position.fen} (discovery=${rec.config.discovery}, draftmin=${rec.config.draftmin}, lam=${rec.config.mlh.lam}, sims=${rec.config.playouts})`;

    assert.equal(result.action, rec.result.action, `${where}: action`);
    assert.equal(result.playoutsUsed, rec.result.playouts_used, `${where}: playouts_used`);
    assert.equal(result.evaluationsUsed, rec.result.evaluations, `${where}: evaluations`);
    assert.equal(tree!.nodeCount, rec.result.nodes, `${where}: nodes`);
    assert.equal(ctx!.gated, rec.result.mlh_gated, `${where}: mlh_gated`);

    const node = tree!.root;
    assert.ok(node !== null, `${where}: tree has no root after search`);
    assert.deepEqual(Array.from(node!.actions), rec.root.actions, `${where}: root.actions`);
    for (let i = 0; i < node!.width; i++) {
      assert.ok(Object.is(node!.visits[i], rec.root.visits[i]), `${where}: visits[${i}]`);
      assert.ok(Object.is(node!.weight[i], rec.root.weight[i]), `${where}: weight[${i}]`);
    }
    const wantWdlSum = flatten2D(rec.root.wdl_sum, 3);
    for (let i = 0; i < wantWdlSum.length; i++) {
      assert.ok(Object.is(node!.wdlSum[i], wantWdlSum[i]), `${where}: wdl_sum[${i}]`);
    }
    const wantMoment = flatten2D(rec.root.moment, 6);
    for (let i = 0; i < wantMoment.length; i++) {
      assert.ok(Object.is(node!.moment[i], wantMoment[i]), `${where}: moment[${i}]`);
    }
    for (let i = 0; i < node!.width; i++) {
      const want = toFinite(rec.root.lcb_scores[i]);
      assert.ok(Object.is(result.lcbScores[i], want), `${where}: lcb_scores[${i}]`);
    }
    assert.ok(
      Object.is(result.rootUtility, rec.intermediates.node_utility),
      `${where}: node_utility`,
    );

    const want = rec.result.tactical;
    if (want === null) {
      assert.equal(result.tactical, null, `${where}: tactical must be null when discovery is off`);
    } else {
      assert.ok(result.tactical !== null, `${where}: tactical must be set when discovery is on`);
      const tactical = result.tactical!;
      assert.equal(tactical.rootWitness, want.root_witness, `${where}: tactical.root_witness`);
      assert.deepEqual(tactical.refuted, want.refuted, `${where}: tactical.refuted`);
      assert.equal(tactical.overridden, want.overridden, `${where}: tactical.overridden`);
      assert.equal(tactical.providerCalls, want.provider_calls, `${where}: tactical.provider_calls`);
      assert.equal(
        tactical.candidateTransitions,
        want.candidate_transitions,
        `${where}: tactical.candidate_transitions`,
      );
      assert.equal(tactical.skippedReason, want.skipped_reason, `${where}: tactical.skipped_reason`);
      assert.equal(tactical.truncated, want.truncated, `${where}: tactical.truncated`);
      if (tactical.skippedReason === 'draft') draftSkipSeen++;
    }

    if (ctx!.gated > 0) mlhGatedSeen++;
    checked++;
  }

  console.log(
    `search: records checked=${checked} mlh_gated_seen=${mlhGatedSeen} draft_skip_seen=${draftSkipSeen}`,
  );
  assert.ok(checked > 0, 'no records checked');
  assert.ok(mlhGatedSeen > 0, 'no record exercised the MLH gate: the production tuple should (design 4.3 M4)');
  assert.ok(draftSkipSeen > 0, 'no record hit the draft skip: a masked draft root should (design 4.3 M4)');
});

// The three properties adapter.py's own unit test pins for mlh_utility_term
// (tests/selfplay/unit/test_adapter_mlh.py), checked here with no fixture:
// shut below the gate, odd under w<->l, and bounded by lambda. The
// pre-registered parameters (design §3.4) are restated rather than read off
// a fixture record, so this runs even with KOM_SEARCH unset.
test('mlhUtilityTerm: shut below the gate, odd, and bounded (adapter.py properties)', () => {
  const config: MlhConfig = { lam: 0.05, tau0: 0.2, tau1: 0.4, slope: 32 };
  const mRoot = 100.0;

  // shut below the gate: |v| < tau0 is exactly 0 for any moves_left.
  const below = Float64Array.from([0.55, 0.3, 0.45]); // v = 0.1 < tau0 = 0.2
  assert.equal(mlhUtilityTerm(below, 0.05, mRoot, config), 0.0);
  assert.equal(mlhUtilityTerm(below, 0.95, mRoot, config), 0.0);

  // odd under w<->l: swapping w and l negates the term exactly.
  const wdl = Float64Array.from([0.7, 0.1, 0.2]); // v = 0.5 > tau0
  const swapped = Float64Array.from([wdl[2], wdl[1], wdl[0]]);
  const term = mlhUtilityTerm(wdl, 0.1, mRoot, config);
  const swappedTerm = mlhUtilityTerm(swapped, 0.1, mRoot, config);
  assert.ok(Object.is(term, -swappedTerm), `term=${term} must be the exact negation of swapped=${swappedTerm}`);

  // bounded: |term| <= lambda * |v| <= lambda for every moves_left, at both
  // slope extremes.
  for (const movesLeft of [0.0, 0.1, 0.3, 0.5, 0.9, 1.0]) {
    const t = mlhUtilityTerm(wdl, movesLeft, mRoot, config);
    assert.ok(Math.abs(t) <= config.lam + 1e-15, `|term|=${Math.abs(t)} must be <= lambda=${config.lam}`);
  }

  // skipped, not zeroed, at lambda == 0: lam<=0 is the stock evaluator's own
  // early return in wrapMlhEvaluator, not mlhUtilityTerm's job, so that
  // property is checked on the wrapper below instead.
});

test('shiftWdl keeps a distribution and moves w - l by exactly the clamped term', () => {
  const wdl = Float64Array.from([0.7, 0.1, 0.2]);
  const shifted = shiftWdl(wdl, 0.05);
  let total = 0.0;
  for (const x of shifted) total += x;
  assert.ok(Object.is(total, 1.0), `shiftWdl must still sum to exactly 1, got ${total}`);
  for (const x of shifted) assert.ok(x >= 0.0, 'shiftWdl must not produce a negative mass');
  const v0 = wdl[0] - wdl[2];
  const v1 = shifted[0] - shifted[2];
  assert.ok(Math.abs(v1 - v0 - 0.05) < 1e-12, `w - l must move by the term: got delta ${v1 - v0}`);

  // Saturation: a term that would push w - l past 1 clamps instead of
  // overshooting or breaking the simplex.
  const saturating = shiftWdl(Float64Array.from([0.9, 0.05, 0.05]), 10.0);
  let saturatingTotal = 0.0;
  for (const x of saturating) saturatingTotal += x;
  assert.ok(Object.is(saturatingTotal, 1.0), 'a saturating shift must still sum to exactly 1');
  assert.ok(saturating[0] - saturating[2] <= 1.0 + 1e-12, 'w - l must not exceed 1 after a saturating shift');
});

test('wrapMlhEvaluator at lam=0 is the stock evaluator, unread context included', () => {
  const inner = () => ({
    policyLogits: new Float32Array([1, 2, 3]),
    wdl: Float64Array.from([0.5, 0.3, 0.2]),
    movesLeft: 0.9,
  });
  const ctx = new MlhContext();
  const wrapped = wrapMlhEvaluator(inner, { lam: 0.0, tau0: 0.2, tau1: 0.4, slope: 32 }, ctx);
  const first = wrapped({} as never);
  const second = wrapped({} as never);
  assert.deepEqual(Array.from(first.wdl), [0.5, 0.3, 0.2]);
  assert.deepEqual(Array.from(second.wdl), [0.5, 0.3, 0.2]);
  assert.equal(ctx.mRoot, null, 'lam=0 must never set the moves-left reference');
  assert.equal(ctx.gated, 0, 'lam=0 must never gate');
});
