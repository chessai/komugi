// Fast-mode U7 (docs/design/clientside-fast-mode.md section 5; ticket
// KOM-1100): pondering. Fake evaluator throughout, same convention as
// worker_fast.test.ts -- real-net parity is a later unit's job. No fixture
// needed (ponder.ts's own pure pieces, plus WorkerEngine wiring, are all
// testable against the fake evaluator the worker already defaults to).

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { decodeAction } from '../src/actions.ts';
import { wrapMlhBatch } from '../src/batch_evaluator.ts';
import type { MlhBatchEvaluator } from '../src/batch_evaluator.ts';
import { ACTION_SIZE, ADVANCED_POSITION } from '../src/constants.ts';
import { makeFakeMlhEvaluator } from '../src/fake_evaluator.ts';
import { boardToFen } from '../src/fen.ts';
import {
  clampPonderK,
  PonderAborted,
  PonderCache,
  ponderKey,
  predictReplies,
  runPonderSearch,
  sha256Hex,
  shouldPonder,
} from '../src/ponder.ts';
import type { PonderCacheEntry, PonderSignal } from '../src/ponder.ts';
import { PRODUCTION_IDENTITY } from '../src/protocol.ts';
import type { WorkerReply, WorkerRequest } from '../src/protocol.ts';
import { gameFromHistory } from '../src/root.ts';
import { buildSan, sanRecords } from '../src/san.ts';
import { MlhContext } from '../src/search/mlh.ts';
import type { MatchNode } from '../src/search/node.ts';
import { DEFAULT_TRANSITION_CAP } from '../src/search/search.ts';
import type { SearchConfig, SearchOptions } from '../src/search/search.ts';
import { MatchGameState } from '../src/search/state.ts';
import { MatchTree } from '../src/search/tree.ts';
import type { BatchWdlEvaluator, MaybeAsyncWdlEvaluator } from '../src/search/tree.ts';
import { WorkerEngine } from '../src/engine.ts';

function isOk(reply: WorkerReply): reply is WorkerReply & { action: number; san: string; stats: Record<string, unknown> } {
  return !('err' in reply);
}

function statsOf(reply: WorkerReply): Record<string, unknown> {
  const stats = { ...(reply as { stats: Record<string, unknown> }).stats };
  delete stats.ms;
  delete stats.phases;
  delete stats.pondered;
  return stats;
}

type BiasPlainEntry = [number, number[], number, number];

function biasPlain(table: Map<number, { errSum: Float64Array; weightSum: number; gen: number }>): BiasPlainEntry[] {
  return Array.from(table.entries())
    .sort(([a], [b]) => a - b)
    .map(([key, entry]) => [key, Array.from(entry.errSum), entry.weightSum, entry.gen]);
}

function fakeBatchEvaluator(): MlhBatchEvaluator {
  const leaf = makeFakeMlhEvaluator(ACTION_SIZE);
  return (states) => states.map((state) => leaf(state));
}

// A short real, legal history from the advanced start -- always the current
// position's first enumerated legal move (root.test.ts's own convention).
function buildHistory(n: number): { history: string[]; fen: string } {
  const history: string[] = [];
  let board = gameFromHistory(history).board;
  for (let i = 0; i < n; i++) {
    const records = sanRecords(board);
    assert.ok(records.length > 0, `no legal move at ply ${i}`);
    history.push(records[0].san);
    board = gameFromHistory(history).board;
  }
  return { history, fen: boardToFen(board) };
}

const BASE = buildHistory(2);

function baseRequest(id: number | string, overrides: Partial<WorkerRequest> = {}): WorkerRequest {
  return { id, fen: BASE.fen, history: BASE.history, sims: 16, batch: 4, ...overrides };
}

// Replays `history` + the engine's own SAN + a predicted reply SAN -- what
// an incoming request would carry if the human actually played that reply.
function nextRequest(
  id: number | string,
  history: readonly string[],
  engineSan: string,
  replySan: string,
  overrides: Partial<WorkerRequest> = {},
): WorkerRequest {
  const nextHistory = [...history, engineSan, replySan];
  const fen = boardToFen(gameFromHistory(nextHistory).board);
  return { id, fen, history: nextHistory, sims: 16, batch: 4, ...overrides };
}

// Mirrors ponder.ts's own `predictReplies`, used only to compute the SAN of
// the top prediction so a test can build the exact request that should hit.
function topPredictionSan(root: MatchNode, afterEngineMoveBoard: ReturnType<typeof gameFromHistory>['board'], chosenAction: number): string {
  const [predicted] = predictReplies(root, chosenAction, 1);
  assert.ok(predicted !== undefined, 'expected at least one predicted reply under the chosen child');
  return buildSan(afterEngineMoveBoard, decodeAction(afterEngineMoveBoard, predicted).struct);
}

test('sha256Hex matches the FIPS 180-4 test vectors', () => {
  assert.equal(sha256Hex(''), 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855');
  assert.equal(sha256Hex('abc'), 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad');
});

test('C2: two histories reaching the same position by different paths (different repetition counts) key differently', () => {
  const common = {
    fen: 'shared-fen',
    sims: 16,
    batch: 4,
    configIdentity: 'id',
    backend: null,
    modelSha256: null,
    maxMs: null,
    mainTableEpoch: 0,
  };
  const direct = ponderKey({ ...common, history: ['m1', 'm2'] });
  const viaRepetition = ponderKey({ ...common, history: ['m1', 'm3', 'm4', 'm2'] });
  assert.notEqual(direct, viaRepetition);
});

test('PonderCache.get consumes the entry whether it matches or not', () => {
  const cache = new PonderCache<null>();
  const entry: PonderCacheEntry<null> = { result: {} as PonderCacheEntry<null>['result'], tree: {} as PonderCacheEntry<null>['tree'], context: null };
  cache.set('k', entry);
  assert.equal(cache.size, 1);
  assert.ok(cache.get('k') !== null);
  assert.equal(cache.size, 0);
  assert.equal(cache.get('k'), null);
});

test('shouldPonder/clampPonderK', () => {
  assert.equal(shouldPonder({ enabled: true, k: 1 }, { hidden: false, discharging: false }), true);
  assert.equal(shouldPonder({ enabled: true, k: 1 }, { hidden: true, discharging: false }), false);
  assert.equal(shouldPonder({ enabled: true, k: 1 }, { hidden: false, discharging: true }), false);
  assert.equal(shouldPonder({ enabled: false, k: 1 }, { hidden: false, discharging: false }), false);
  assert.equal(clampPonderK(0), 1);
  assert.equal(clampPonderK(5), 3);
  assert.equal(clampPonderK(2), 2);
});

test('F9/S-5: the prediction is the argmax of visits under the chosen child, reading no evaluator', async () => {
  const engine = new WorkerEngine();
  const reply1 = await engine.handleFast(baseRequest(1), fakeBatchEvaluator());
  assert.ok(isOk(reply1), JSON.stringify(reply1));
  if (!isOk(reply1)) return;

  const root = engine.tree!.root!;
  const chosenIndex = root.childIndex(reply1.action);
  assert.ok(chosenIndex !== null);
  const child = root.children.get(chosenIndex!) as MatchNode;
  let expected = -1;
  let bestVisits = -1;
  for (let i = 0; i < child.width; i++) {
    if (child.visits[i] > bestVisits) {
      bestVisits = child.visits[i];
      expected = child.actions[i];
    }
  }

  const evaluationsBefore = engine.tree!.evaluations;
  const predicted = predictReplies(root, reply1.action, 1);
  assert.deepEqual(predicted, [expected]);
  assert.equal(engine.tree!.evaluations, evaluationsBefore, 'predictReplies must call no evaluator');
});

test('a ponder hit equals the unpondered reply, the main bias table matches the no-pondering baseline, and F8: the next prediction reads the pondered tree', async () => {
  const engine = new WorkerEngine();
  const req1 = baseRequest(1);
  const reply1 = await engine.handleFast(req1, fakeBatchEvaluator());
  assert.ok(isOk(reply1), JSON.stringify(reply1));
  if (!isOk(reply1)) return;

  await engine.ponderingPromise();
  assert.ok(engine.ponderCacheSize > 0, 'expected at least one cached predicted reply');
  const treeAfterReply1 = engine.tree;

  const afterEngineMove = gameFromHistory([...BASE.history, reply1.san]).board;
  const root = engine.tree!.root!;
  const replySan = topPredictionSan(root, afterEngineMove, reply1.action);
  const req2 = nextRequest(2, BASE.history, reply1.san, replySan);

  const reply2 = await engine.handleFast(req2, fakeBatchEvaluator());
  assert.ok(isOk(reply2), JSON.stringify(reply2));
  if (!isOk(reply2)) return;
  assert.equal(reply2.stats.pondered, true);

  // F8: the hit must have adopted the pondered tree -- the main tree's own
  // identity changes.
  assert.notEqual(engine.tree, treeAfterReply1);

  // F8, continued: the hit also starts the NEXT ponder session, predicting
  // off the newly adopted tree.
  await engine.ponderingPromise();
  assert.ok(engine.ponderCacheSize > 0, 'expected the post-hit continuation to have pondered too');

  // N-6: the no-pondering baseline -- a fresh engine, pondering disabled,
  // processing the SAME two requests for real.
  const plain = new WorkerEngine();
  plain.handlePonderSettings({ id: 's', type: 'ponderSettings', enabled: false });
  await plain.handleFast(req1, fakeBatchEvaluator());
  const plainReply2 = await plain.handleFast(req2, fakeBatchEvaluator());
  assert.ok(isOk(plainReply2), JSON.stringify(plainReply2));
  if (!isOk(plainReply2)) return;

  assert.equal(reply2.action, plainReply2.action);
  assert.equal(reply2.san, plainReply2.san);
  assert.equal((reply2 as unknown as { stamp: string }).stamp, (plainReply2 as unknown as { stamp: string }).stamp);
  assert.deepEqual(statsOf(reply2), statsOf(plainReply2));
  assert.deepEqual(biasPlain(engine.tree!.bias.entries), biasPlain(plain.tree!.bias.entries));
});

test('a ponder miss leaves the main bias table byte-identical to the no-pondering baseline', async () => {
  const engine = new WorkerEngine();
  const req1 = baseRequest(1);
  await engine.handleFast(req1, fakeBatchEvaluator());
  await engine.ponderingPromise();
  assert.ok(engine.ponderCacheSize > 0);

  // Same position, different sims: guaranteed not to match anything cached.
  const missReq = baseRequest(2, { sims: 17 });
  const missReply = await engine.handleFast(missReq, fakeBatchEvaluator());
  assert.ok(isOk(missReply));
  if (isOk(missReply)) assert.notEqual(missReply.stats.pondered, true);

  const plain = new WorkerEngine();
  plain.handlePonderSettings({ id: 's', type: 'ponderSettings', enabled: false });
  await plain.handleFast(req1, fakeBatchEvaluator());
  await plain.handleFast(missReq, fakeBatchEvaluator());

  assert.deepEqual(biasPlain(engine.tree!.bias.entries), biasPlain(plain.tree!.bias.entries));
});

test('C4: aborting mid-batch throws PonderAborted and makes no evaluator call after the aborted batch resolved', async () => {
  // `runPonderSearch` is exactly what `worker.ts`'s own ponder session calls
  // per predicted reply (design 5.2/5.3) -- exercised directly here, on
  // BASE's own position/budget (already shown by the tests above to need
  // several batches, not just the root), independent of which continuation
  // WorkerEngine happens to predict for any given fixture position.
  const configSource = new WorkerEngine();
  await configSource.handleFast(baseRequest('cfg'), fakeBatchEvaluator());
  const baseConfig = configSource.tree!.config;

  const game = gameFromHistory(BASE.history);
  const state = new MatchGameState(game, { minPlacements: 10, isRoot: true });
  const searchConfig: SearchConfig = { ...baseConfig, playouts: 16 };
  const searchOpts: SearchOptions = { discovery: true, ply: game.turns, tacticalTransitionCap: DEFAULT_TRANSITION_CAP };

  let calls = 0;
  const signal: PonderSignal = { aborted: false };
  const leaf = makeFakeMlhEvaluator(ACTION_SIZE);
  const inner: MlhBatchEvaluator = (states) => states.map((state) => leaf(state));
  const mlhCtx = new MlhContext();
  const wrapped = wrapMlhBatch(inner, PRODUCTION_IDENTITY.mlh, mlhCtx);
  const countingBatch: BatchWdlEvaluator = (states) => {
    calls += 1;
    if (calls === 2) signal.aborted = true;
    return wrapped(states);
  };
  const rootEvaluator: MaybeAsyncWdlEvaluator = (s) => Promise.resolve(countingBatch([s])).then((rows) => rows[0]);
  const scratchTree = new MatchTree(searchConfig, rootEvaluator, countingBatch);

  await assert.rejects(
    () => runPonderSearch(scratchTree, state, searchConfig, searchOpts, 4, signal),
    PonderAborted,
  );
  assert.equal(calls, 2, 'no evaluator call after the abort was observed');
});

test('C2: a main-tree search between ponder start and lookup invalidates the prediction (epoch mismatch)', async () => {
  const engine = new WorkerEngine();
  const req1 = baseRequest(1);
  const reply1 = await engine.handleFast(req1, fakeBatchEvaluator());
  assert.ok(isOk(reply1), JSON.stringify(reply1));
  if (!isOk(reply1)) return;

  await engine.ponderingPromise();
  assert.ok(engine.ponderCacheSize > 0);

  const afterEngineMove = gameFromHistory([...BASE.history, reply1.san]).board;
  const root = engine.tree!.root!;
  const replySan = topPredictionSan(root, afterEngineMove, reply1.action);
  const req2 = nextRequest(2, BASE.history, reply1.san, replySan);

  // A main-tree search slips in between the ponder's own store and this
  // lookup -- `handle()` bypasses `handleFast`'s own abort-first discipline
  // entirely, so this is the defense-in-depth path, not the usual one.
  engine.handle({ id: 'interloper', fen: ADVANCED_POSITION, sims: 4 });

  const missReply = await engine.handleFast(req2, fakeBatchEvaluator());
  assert.ok(isOk(missReply), JSON.stringify(missReply));
  if (isOk(missReply)) assert.notEqual(missReply.stats.pondered, true);
});

test('5.4: pondering off when the gating flags say hidden or discharging', async () => {
  for (const gating of [{ hidden: true }, { discharging: true }] as const) {
    const engine = new WorkerEngine();
    engine.handlePonderSettings({ id: 's', type: 'ponderSettings', ...gating });
    const reply = await engine.handleFast(baseRequest(1), fakeBatchEvaluator());
    assert.ok(isOk(reply), JSON.stringify(reply));
    await engine.ponderingPromise();
    assert.equal(engine.ponderCacheSize, 0);
  }
});
