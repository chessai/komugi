// Fast-mode unit U3: searchBatchedGen / searchFast / searchFastAsync
// (docs/design/clientside-fast-mode.md sections 1.7-1.8, 6.1-6.2 items 5-7,
// 1.7's C3 deadline fix). KOM_SEARCH/KOM_PAIRWISE are required: every test
// FAILS (never skips) without them (B1).

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { fakeLogits, fakeMovesLeft, fakeWdl, fnv1a32, makeFakeEvaluator } from '../src/fake_evaluator.ts';
import { boardFromFen } from '../src/fen.ts';
import { Game } from '../src/game.ts';
import { RepetitionRule } from '../src/repetition.ts';
import { MlhContext, wrapMlhEvaluator, type MlhConfig, type MlhLeafEval } from '../src/search/mlh.ts';
import {
  searchBatchedGen,
  searchFast,
  type MatchResult,
  type MatchTactical,
  type SearchConfig,
  type SearchOptions,
} from '../src/search/search.ts';
import { MatchGameState, type MatchState } from '../src/search/state.ts';
import { MatchTree, runBatchGenSync, type BatchWdlEvaluator } from '../src/search/tree.ts';
import { loadSearchFixture } from './fixture.ts';

function toFinite(v: number | string): number {
  if (v === 'Infinity') return Infinity;
  if (v === '-Infinity') return -Infinity;
  if (v === 'NaN') return NaN;
  return v;
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
    num_actions: number;
    discovery: boolean;
    draftmin: number;
    mlh: { lam: number; tau0: number; tau1: number; slope: number };
  };
  position: { fen: string; stamp: string };
  result: {
    action: number;
    playouts_used: number;
    evaluations: number;
    nodes: number;
    tactical: TacticalRecord;
  };
  root: { actions: number[]; visits: number[]; weight: number[]; lcb_scores: (number | string)[] };
};

type SearchFixture = { searches: SearchRecord[] };

function loadFixture(): SearchFixture {
  return loadSearchFixture<SearchFixture>();
}

function rootStateFor(rec: SearchRecord): MatchGameState {
  const board = boardFromFen(rec.position.fen);
  const rule = rec.position.stamp === 'present' ? RepetitionRule.PLACEMENT : RepetitionRule.FULL;
  const game = new Game(board, rule);
  return new MatchGameState(game, { minPlacements: rec.config.draftmin, isRoot: true });
}

function makeFakeMlhLeaf(numActions: number): (state: MatchState) => MlhLeafEval {
  return (state: MatchState): MlhLeafEval => {
    const h = fnv1a32((state as MatchGameState).placementKeyBytes());
    return { policyLogits: fakeLogits(h, numActions), wdl: fakeWdl(h), movesLeft: fakeMovesLeft(h) };
  };
}

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

// One (tree, MlhContext) per contiguous fixture group, exactly as
// the fixture exporter built the server that produced it (search.test.ts's
// own grouping, reused here so a bias-table carryover within a group lands
// on the same key sequence the fixture recorded).
class FixtureRunner {
  private groupKey: string | null = null;
  private tree: MatchTree | null = null;
  private ctx: MlhContext | null = null;

  next(rec: SearchRecord): MatchTree {
    const key = groupKeyOf(rec);
    if (key !== this.groupKey) {
      this.groupKey = key;
      const leaf = makeFakeMlhLeaf(rec.config.num_actions);
      this.ctx = new MlhContext();
      const mlhConfig: MlhConfig = {
        lam: rec.config.mlh.lam,
        tau0: rec.config.mlh.tau0,
        tau1: rec.config.mlh.tau1,
        slope: rec.config.mlh.slope,
      };
      const evaluator = wrapMlhEvaluator(leaf, mlhConfig, this.ctx);
      this.tree = new MatchTree(rec.config, evaluator);
    } else {
      this.ctx!.reset();
    }
    return this.tree!;
  }
}

// Compares a MatchResult against the fixture exporter's recorded ground
// truth for one record -- the same fields search.test.ts pins for search()
// itself, checked here against a different search function (searchFast at
// batch=1, or searchBatchedGen called directly).
function assertMatchesFixture(where: string, result: MatchResult, rec: SearchRecord): void {
  assert.equal(result.action, rec.result.action, `${where}: action`);
  assert.equal(result.playoutsUsed, rec.result.playouts_used, `${where}: playouts_used`);
  assert.equal(result.evaluationsUsed, rec.result.evaluations, `${where}: evaluations`);
  assert.equal(result.deadlineHit, false, `${where}: deadlineHit must be false absent a deadline`);
  assert.deepEqual(Array.from(result.actions), rec.root.actions, `${where}: root.actions`);
  for (let i = 0; i < result.visits.length; i++) {
    assert.ok(Object.is(result.visits[i], rec.root.visits[i]), `${where}: visits[${i}]`);
    assert.ok(Object.is(result.weights[i], rec.root.weight[i]), `${where}: weight[${i}]`);
  }
  for (let i = 0; i < result.lcbScores.length; i++) {
    const want = toFinite(rec.root.lcb_scores[i]);
    assert.ok(Object.is(result.lcbScores[i], want), `${where}: lcb_scores[${i}]`);
  }
  const want = rec.result.tactical;
  if (want === null) {
    assert.equal(result.tactical, null, `${where}: tactical must be null when discovery is off`);
  } else {
    assert.ok(result.tactical !== null, `${where}: tactical must be set when discovery is on`);
    const tactical: MatchTactical = result.tactical!;
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
  }
}

function loadRecords(): SearchRecord[] {
  const fixture = loadFixture();
  assert.ok(fixture.searches.length > 0, 'no search records in the fixture');
  return fixture.searches;
}

test('searchFast(batch=1) equals the fixture on every record (6.1 dispatch identity)', () => {
  const runner = new FixtureRunner();
  let checked = 0;
  for (const rec of loadRecords()) {
    const tree = runner.next(rec);
    const state = rootStateFor(rec);
    const opts: SearchOptions = { discovery: rec.config.discovery };
    const result = searchFast(tree, state, rec.config, opts, 1);
    assertMatchesFixture(`${rec.position.fen} (searchFast b=1)`, result, rec);
    checked++;
  }
  assert.ok(checked > 0, 'no records checked');
});

test('searchBatchedGen forced directly at batch=1 equals the fixture (1.7 degeneracy, not through dispatch)', () => {
  const runner = new FixtureRunner();
  let checked = 0;
  for (const rec of loadRecords()) {
    const tree = runner.next(rec);
    const state = rootStateFor(rec);
    const opts: SearchOptions = { discovery: rec.config.discovery };
    const result = runBatchGenSync(searchBatchedGen(tree, state, rec.config, opts, null, 1));
    assertMatchesFixture(`${rec.position.fen} (searchBatchedGen b=1 direct)`, result, rec);
    checked++;
  }
  assert.ok(checked > 0, 'no records checked');
});

function sumInt32(a: Int32Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i];
  return s;
}

test('design 6.2 items 5-7 at batch 4/8: filters unchanged where they fire, invariants hold where playouts run', () => {
  const batches = [4, 8];
  let zeroPlayoutChecked = 0;
  let widePlayoutChecked = 0;
  for (const batch of batches) {
    const runner = new FixtureRunner();
    for (const rec of loadRecords()) {
      const tree = runner.next(rec);
      const state = rootStateFor(rec);
      const opts: SearchOptions = { discovery: rec.config.discovery };
      const result = searchFast(tree, state, rec.config, opts, batch);

      if (rec.result.playouts_used === 0) {
        // The root-win scan fired, or the root has width <= 1: neither
        // depends on batch size, so the batched result must equal the
        // fixture exactly, same as batch 1 (item 7).
        assertMatchesFixture(`${rec.position.fen} B${batch} (zero-playout filter)`, result, rec);
        zeroPlayoutChecked++;
      } else {
        // Real playouts ran: batching changes the search (virtual loss), so
        // the action need not match, but the bookkeeping invariants (items
        // 2, 4) must, and no deadline was set.
        assert.equal(result.deadlineHit, false, `${rec.position.fen} B${batch}: deadlineHit`);
        assert.equal(result.playoutsUsed, rec.config.playouts, `${rec.position.fen} B${batch}: playoutsUsed`);
        assert.equal(sumInt32(result.visits), result.playoutsUsed, `${rec.position.fen} B${batch}: visit conservation`);
        assert.ok(
          result.actions.includes(result.action),
          `${rec.position.fen} B${batch}: action must be one of the root's own actions`,
        );
        widePlayoutChecked++;
      }
    }
  }
  console.log(`search_batch: zero-playout filter checks=${zeroPlayoutChecked}, wide-playout checks=${widePlayoutChecked}`);
  assert.ok(widePlayoutChecked > 0, 'no record exercised the real batched playout loop');
});

// The real fixture happens to carry no root-win/width<=1 record (the check
// above found zero), so item 7's "where the root-win scan fires, the
// batched result equals the exact one (playoutsUsed == 0, same tactical
// record)" is pinned directly instead, with a toy MatchState whose first
// root action is an immediate win for the mover -- independent of batch
// size, since the scan runs before any playout (1.8).
class ToyWinState implements MatchState {
  path: number[];
  constructor(path: number[] = []) {
    this.path = path;
  }
  legalMask(): Uint8Array {
    const mask = new Uint8Array(2);
    if (this.path.length === 0) {
      mask[0] = 1;
      mask[1] = 1;
    }
    return mask;
  }
  play(action: number): void {
    this.path.push(action);
  }
  undo(): void {
    this.path.pop();
  }
  clone(): ToyWinState {
    return new ToyWinState(this.path.slice());
  }
  isTerminal(): boolean {
    return this.path.length >= 1;
  }
  toPlay(): number {
    return this.path.length % 2;
  }
  terminalWdl(): Float64Array {
    // Action 0 ends the game at once: toPlay() here is the mover AFTER the
    // root's move (1), and its WDL is a loss ([0,0,1]) -- winnerOf reads
    // that as `1 - toPlay()` = the ROOT's mover having won. Action 1 is a
    // draw, so it is never the scan's witness.
    if (this.path[0] === 0) return Float64Array.from([0, 0, 1]);
    return Float64Array.from([0, 1, 0]);
  }
  placementKeyBytes(): Uint8Array {
    return Uint8Array.from(this.path);
  }
}

test('root-win scan fires identically at batch 1/4/8 (6.2 item 7, toy state)', () => {
  const base = loadRecords()[0].config;
  const config: SearchConfig = { ...base, num_actions: 2, playouts: 10 };
  const evaluator = () => ({ policyLogits: new Float32Array(2), wdl: Float64Array.from([0.0, 1.0, 0.0]) });
  for (const batch of [1, 4, 8]) {
    const tree = new MatchTree(config, evaluator);
    const result = searchFast(tree, new ToyWinState(), config, { discovery: true }, batch);
    assert.equal(result.playoutsUsed, 0, `B${batch}: root-win must spend no playout`);
    assert.equal(result.action, 0, `B${batch}: the witnessed action must be chosen`);
    assert.equal(result.deadlineHit, false, `B${batch}: deadlineHit`);
    assert.ok(result.tactical !== null, `B${batch}: tactical must be set`);
    assert.equal(result.tactical!.rootWitness, 0, `B${batch}: tactical.rootWitness`);
  }
});

function wideRecord(): SearchRecord {
  for (const rec of loadRecords()) {
    const tree = new MatchTree(rec.config, makeFakeEvaluator(rec.config.num_actions));
    tree.ensureRoot(rootStateFor(rec));
    if ((tree.root!.width > 1) && rec.result.playouts_used > 8 && rec.config.discovery) return rec;
  }
  throw new Error('no fixture record with width > 1, discovery on and playouts_used > 8');
}

test('a fake-clock deadline stops between batches with deadlineHit and the real playouts (1.7 C3)', () => {
  const rec = wideRecord();
  const state = rootStateFor(rec);
  const tree = new MatchTree(rec.config, makeFakeEvaluator(rec.config.num_actions));

  // KOM-1220: under a deadline the search now runs in chunks of `batch`
  // (here 4) and polls shouldStop BETWEEN chunks, not after every playout.
  // wideRecord() guarantees config.playouts > 8, so two whole chunks (8
  // playouts) always fit inside the budget with room to spare.
  const batch = 4;
  const K = 2;
  assert.ok(K * batch < rec.config.playouts, 'K*batch must be smaller than the full budget to actually truncate');
  let calls = 0;
  const opts: SearchOptions = {
    discovery: rec.config.discovery,
    shouldStop: () => {
      calls += 1;
      return calls >= K;
    },
  };
  const result = searchFast(tree, state, rec.config, opts, batch);

  assert.equal(result.deadlineHit, true, 'deadlineHit must be true once the clock expires');
  assert.equal(result.playoutsUsed, K * batch, 'playoutsUsed must equal K whole chunks of batch, not K playouts');
  assert.ok(result.playoutsUsed < rec.config.playouts, 'a deadline run must not silently reach the full budget');
  assert.ok(
    result.evaluationsUsed <= result.playoutsUsed + 1,
    'evaluations must not exceed one per playout plus the root',
  );
  // The root-win scan and survey read only completed statistics and always
  // run (1.8): discovery is on, so tactical must be set regardless of the
  // truncation.
  assert.ok(result.tactical !== null, 'discovery=true must still produce a tactical record after truncation');
  assert.ok(result.actions.includes(result.action), 'the truncated result must still choose a legal root action');
});

test('shouldStop that never fires reaches the full budget with deadlineHit=false', () => {
  const rec = wideRecord();
  const state = rootStateFor(rec);
  const tree = new MatchTree(rec.config, makeFakeEvaluator(rec.config.num_actions));
  const opts: SearchOptions = { discovery: rec.config.discovery, shouldStop: () => false };
  const result = searchFast(tree, state, rec.config, opts, 4);
  assert.equal(result.deadlineHit, false);
  assert.equal(result.playoutsUsed, rec.config.playouts);
});

test('N-4: a clock on the batched path charges one evaluatorCalls per batch, plus the root', () => {
  const rec = wideRecord();
  const state = rootStateFor(rec);
  const single = makeFakeEvaluator(rec.config.num_actions);
  const tree = new MatchTree(rec.config, single);
  // A standalone batch evaluator that does not itself call tree.evaluator
  // (the double-count hazard startBatchTimer's own comment documents).
  const batchEvaluator: BatchWdlEvaluator = (states) => states.map((s) => single(s));
  tree.batchEvaluator = batchEvaluator;

  let now = 0;
  const opts: SearchOptions = { discovery: rec.config.discovery, clock: () => now++ };
  const result = searchFast(tree, state, rec.config, opts, 4);

  assert.ok(result.phases !== null, 'a clock must produce phases');
  const expectedBatches = Math.ceil(rec.config.playouts / 4);
  // Root counts once through the single-leaf seam; the playout loop adds at
  // most one evaluator call per batch (fewer if a batch closes on terminals
  // alone, 1.5) -- so this is an upper bound, matching N-4's own intent that
  // a batched search's evaluator calls are not inflated to one per row.
  assert.ok(
    result.phases!.evaluatorCalls <= expectedBatches + 1,
    `evaluatorCalls=${result.phases!.evaluatorCalls} must not exceed batches(${expectedBatches})+1`,
  );
  assert.ok(result.phases!.evaluatorCalls >= 1, 'the root call alone must be counted');
});
