// KOM-1195 unit U3: SearchOptions.onProgress / yieldToHost at the chunked
// loop's chunk boundary (docs/design/kom1195-analysis-api.md sections 3-4,
// unit U3 in section 9). The continuous branch, the `shouldStop` type and
// both `batch === 1` dispatch lines are untouched by this unit -- these
// tests exercise only the chunked loop (`opts.shouldStop !== undefined`).

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { makeFakeEvaluator } from '../src/fake_evaluator.ts';
import { boardFromFen } from '../src/fen.ts';
import { Game } from '../src/game.ts';
import { RepetitionRule } from '../src/repetition.ts';
import {
  searchBatchedGen,
  searchFast,
  searchFastAsync,
  type SearchConfig,
  type SearchOptions,
} from '../src/search/search.ts';
import { MatchGameState } from '../src/search/state.ts';
import { MatchTree, runBatchGenAsync } from '../src/search/tree.ts';
import { loadSearchFixture } from './fixture.ts';

type SearchRecord = {
  config: SearchConfig & {
    num_actions: number;
    discovery: boolean;
    draftmin: number;
    mlh: { lam: number; tau0: number; tau1: number; slope: number };
  };
  position: { fen: string; stamp: string };
  result: { playouts_used: number };
};

type SearchFixture = { searches: SearchRecord[] };

function loadRecords(): SearchRecord[] {
  const fixture = loadSearchFixture<SearchFixture>();
  assert.ok(fixture.searches.length > 0, 'no search records in the fixture');
  return fixture.searches;
}

function rootStateFor(rec: SearchRecord): MatchGameState {
  const board = boardFromFen(rec.position.fen);
  const rule = rec.position.stamp === 'present' ? RepetitionRule.PLACEMENT : RepetitionRule.FULL;
  const game = new Game(board, rule);
  return new MatchGameState(game, { minPlacements: rec.config.draftmin, isRoot: true });
}

// A record whose real (continuous) search ran a wide, multi-chunk playout
// loop -- the same selection search_batch.test.ts's own wideRecord() uses
// for its deadline tests, so chunking at batch 4 gives several boundaries.
function wideRecord(): SearchRecord {
  for (const rec of loadRecords()) {
    const tree = new MatchTree(rec.config, makeFakeEvaluator(rec.config.num_actions));
    tree.ensureRoot(rootStateFor(rec));
    if (tree.root!.width > 1 && rec.result.playouts_used > 8 && rec.config.discovery) return rec;
  }
  throw new Error('no fixture record with width > 1, discovery on and playouts_used > 8');
}

test('onProgress values are strictly increasing, the last equals stats.playouts, and a never-firing shouldStop leaves deadlineHit false', () => {
  const rec = wideRecord();
  const state = rootStateFor(rec);
  const tree = new MatchTree(rec.config, makeFakeEvaluator(rec.config.num_actions));
  const batch = 4;
  const progress: number[] = [];
  const opts: SearchOptions = {
    discovery: rec.config.discovery,
    shouldStop: () => false,
    onProgress: (p) => progress.push(p),
  };
  const result = searchFast(tree, state, rec.config, opts, batch);

  assert.ok(progress.length > 0, 'onProgress must have fired at least once');
  for (let i = 1; i < progress.length; i++) {
    assert.ok(progress[i]! > progress[i - 1]!, `progress must strictly increase at index ${i}`);
  }
  assert.equal(progress[progress.length - 1], result.playoutsUsed, 'the last progress value must equal playoutsUsed');
  assert.equal(result.playoutsUsed, rec.config.playouts, 'a never-firing shouldStop must reach the full budget');
  assert.equal(result.deadlineHit, false, 'reaching the cap must never set deadlineHit');
});

test('a shouldStop that turns true mid-search gives deadlineHit: true and stops onProgress at that boundary', () => {
  const rec = wideRecord();
  const state = rootStateFor(rec);
  const tree = new MatchTree(rec.config, makeFakeEvaluator(rec.config.num_actions));
  const batch = 4;
  const K = 2;
  assert.ok(K * batch < rec.config.playouts, 'K*batch must be smaller than the full budget to actually truncate');
  let calls = 0;
  const progress: number[] = [];
  const opts: SearchOptions = {
    discovery: rec.config.discovery,
    onProgress: (p) => progress.push(p),
    shouldStop: () => {
      calls += 1;
      return calls >= K;
    },
  };
  const result = searchFast(tree, state, rec.config, opts, batch);

  assert.equal(result.deadlineHit, true, 'deadlineHit must be true once shouldStop fires');
  assert.equal(result.playoutsUsed, K * batch, 'playoutsUsed must stop at K whole chunks of batch');
  assert.equal(progress[progress.length - 1], result.playoutsUsed, 'the last progress value must equal the truncated playoutsUsed');
  for (let i = 1; i < progress.length; i++) {
    assert.ok(progress[i]! > progress[i - 1]!, `progress must strictly increase at index ${i}`);
  }
});

test('the result is identical with onProgress, yieldToHost and a never-firing shouldStop set, and with a never-firing shouldStop alone', async () => {
  const rec = wideRecord();
  const state = rootStateFor(rec);
  const batch = 4;

  const plainTree = new MatchTree(rec.config, makeFakeEvaluator(rec.config.num_actions));
  const plainOpts: SearchOptions = { discovery: rec.config.discovery, shouldStop: () => false };
  const plainResult = await searchFastAsync(plainTree, state, rec.config, plainOpts, batch);

  const hookedTree = new MatchTree(rec.config, makeFakeEvaluator(rec.config.num_actions));
  let progressCalls = 0;
  let yieldCalls = 0;
  const hookedOpts: SearchOptions = {
    discovery: rec.config.discovery,
    shouldStop: () => false,
    onProgress: () => {
      progressCalls += 1;
    },
    yieldToHost: () => {
      yieldCalls += 1;
      return Promise.resolve([]);
    },
  };
  const hookedResult = await searchFastAsync(hookedTree, state, rec.config, hookedOpts, batch);

  assert.ok(progressCalls > 0, 'onProgress must have fired');
  assert.ok(yieldCalls > 0, 'yieldToHost must have fired');
  assert.equal(hookedResult.action, plainResult.action, 'action must be unaffected by the hooks');
  assert.equal(hookedResult.playoutsUsed, plainResult.playoutsUsed, 'playoutsUsed must be unaffected by the hooks');
  assert.equal(hookedResult.deadlineHit, plainResult.deadlineHit, 'deadlineHit must be unaffected by the hooks');
  assert.deepEqual(Array.from(hookedResult.visits), Array.from(plainResult.visits), 'visits must be unaffected by the hooks');
  assert.deepEqual(Array.from(hookedResult.weights), Array.from(plainResult.weights), 'weights must be unaffected by the hooks');
  assert.deepEqual(Array.from(hookedResult.lcbScores), Array.from(plainResult.lcbScores), 'lcbScores must be unaffected by the hooks');
});

test("yieldToHost is called once per boundary with playouts remaining, each time before that boundary's shouldStop poll (section 3 order)", async () => {
  const rec = wideRecord();
  const state = rootStateFor(rec);
  const tree = new MatchTree(rec.config, makeFakeEvaluator(rec.config.num_actions));
  const batch = 4;
  const order: string[] = [];
  const opts: SearchOptions = {
    discovery: rec.config.discovery,
    onProgress: () => order.push('progress'),
    yieldToHost: () => {
      order.push('yield');
      return Promise.resolve([]);
    },
    shouldStop: () => {
      order.push('stop');
      return false;
    },
  };
  const result = await runBatchGenAsync(searchBatchedGen(tree, state, rec.config, opts, null, batch));

  const totalChunks = Math.ceil(rec.config.playouts / batch);
  assert.equal(result.playoutsUsed, rec.config.playouts, 'a never-firing shouldStop must reach the full budget');

  // Every boundary but the last has playouts remaining, so it must run
  // progress, then yield, then the shouldStop poll, in that order (section
  // 3). The final boundary reaches the cap, so it emits progress only --
  // shouldStop is never consulted there (the cap-exclusion rule).
  const expected: string[] = [];
  for (let i = 0; i < totalChunks - 1; i++) expected.push('progress', 'yield', 'stop');
  expected.push('progress');
  assert.deepEqual(order, expected, 'progress/yield/stop order at every boundary');
});
