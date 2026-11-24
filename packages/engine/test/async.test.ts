// KOM-1078: resumable search for a per-leaf-async evaluator (onnxruntime-web
// has no synchronous InferenceSession.run). search/tree.ts's MatchTree now
// drives its one evaluator call per expansion through a generator
// (expandGen/evaluateGen) so a real async evaluator can suspend the search
// exactly there and resume with its result -- no blocking, no replay from
// scratch (the e2e_driver.mjs cache-miss workaround this supersedes).
//
// These tests use the deterministic fake evaluator throughout (worker.ts's
// own default), promisified (evaluator.ts's makeFakeMlhEvaluatorAsync) --
// real-net parity is ort.test.ts's job, not this one's. What this file pins:
// the async path computes exactly the same thing the sync path does (no
// fixture needed -- the referee is the sync path itself), and the
// suspension is real (the tree does not advance until the promise settles).

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { ADVANCED_POSITION } from '../src/constants.ts';
import { makeFakeMlhEvaluator, makeFakeMlhEvaluatorAsync } from '../src/fake_evaluator.ts';
import { boardFromFen } from '../src/fen.ts';
import { Game } from '../src/game.ts';
import { RepetitionRule } from '../src/repetition.ts';
import { sanRecords } from '../src/san.ts';
import { MlhContext, wrapMlhEvaluator, wrapMlhEvaluatorAsync } from '../src/search/mlh.ts';
import { search, searchAsync, type SearchConfig } from '../src/search/search.ts';
import { MatchGameState } from '../src/search/state.ts';
import { MatchTree, type MatchConfig } from '../src/search/tree.ts';
import { WorkerEngine } from '../src/engine.ts';
import type { WorkerReply, WorkerRequest } from '../src/protocol.ts';

const ACTION_SIZE = 37341;

const BASE_CONFIG: Omit<MatchConfig, 'playouts'> = {
  num_actions: ACTION_SIZE,
  contempt: 0.0,
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
  bias_weight_exponent: 0.85,
  bias_free_prop: 0.8,
  bias_weight_floor: 1e-6,
  k_lcb: 5.0,
  min_visit_prop_for_lcb: 0.15,
  var_floor: 0.05,
  variance_mode: 'moment',
};

function rootState(): MatchGameState {
  const game = new Game(boardFromFen(ADVANCED_POSITION), RepetitionRule.PLACEMENT);
  return new MatchGameState(game, { minPlacements: 10, isRoot: true });
}

function snapshotNode(node: { actions: Int32Array; visits: Int32Array; weight: Float64Array; wdlSum: Float64Array; moment: Float64Array }) {
  return {
    actions: Array.from(node.actions),
    visits: Array.from(node.visits),
    weight: Array.from(node.weight),
    wdlSum: Array.from(node.wdlSum),
    moment: Array.from(node.moment),
  };
}

test('MatchTree: ensureRootAsync/playoutAsync reproduce ensureRoot/playout bit for bit (fake evaluator, production MLH)', async () => {
  const playouts = 24;
  const config: MatchConfig = { ...BASE_CONFIG, playouts };
  const mlhConfig = { lam: 0.05, tau0: 0.8, tau1: 0.9, slope: 32 };

  const syncState = rootState();
  const syncTree = new MatchTree(config, wrapMlhEvaluator(makeFakeMlhEvaluator(ACTION_SIZE), mlhConfig, new MlhContext()));
  syncTree.ensureRoot(syncState);
  for (let i = 0; i < playouts; i++) syncTree.playout(syncState);

  const asyncState = rootState();
  const asyncTree = new MatchTree(
    config,
    wrapMlhEvaluatorAsync(makeFakeMlhEvaluatorAsync(ACTION_SIZE), mlhConfig, new MlhContext()),
  );
  await asyncTree.ensureRootAsync(asyncState);
  for (let i = 0; i < playouts; i++) await asyncTree.playoutAsync(asyncState);

  assert.ok(syncTree.root !== null && asyncTree.root !== null);
  assert.deepEqual(snapshotNode(asyncTree.root!), snapshotNode(syncTree.root!));
  assert.equal(asyncTree.evaluations, syncTree.evaluations);
  assert.equal(asyncTree.nodeCount, syncTree.nodeCount);
});

test('search()/searchAsync() reach the exact same MatchResult (discovery on, draftmin, MLH gated)', async () => {
  const config: SearchConfig = { ...BASE_CONFIG, playouts: 32 };
  const mlhConfig = { lam: 0.05, tau0: 0.8, tau1: 0.9, slope: 32 };
  const opts = { discovery: true, ply: 0 };

  const syncTree = new MatchTree(config, wrapMlhEvaluator(makeFakeMlhEvaluator(ACTION_SIZE), mlhConfig, new MlhContext()));
  const syncResult = search(syncTree, rootState(), config, opts);

  const asyncTree = new MatchTree(
    config,
    wrapMlhEvaluatorAsync(makeFakeMlhEvaluatorAsync(ACTION_SIZE), mlhConfig, new MlhContext()),
  );
  const asyncResult = await searchAsync(asyncTree, rootState(), config, opts);

  assert.equal(asyncResult.action, syncResult.action);
  assert.equal(asyncResult.playoutsUsed, syncResult.playoutsUsed);
  assert.equal(asyncResult.evaluationsUsed, syncResult.evaluationsUsed);
  assert.deepEqual(Array.from(asyncResult.visits), Array.from(syncResult.visits));
  assert.deepEqual(Array.from(asyncResult.weights), Array.from(syncResult.weights));
  assert.deepEqual(Array.from(asyncResult.rootWdl), Array.from(syncResult.rootWdl));
  assert.deepEqual(asyncResult.tactical, syncResult.tactical);
});

function isOk(reply: WorkerReply): reply is WorkerReply & { stats: Record<string, unknown> } {
  return !('err' in reply);
}

function statsOf(reply: WorkerReply): Record<string, unknown> {
  return { ...(reply as { stats: Record<string, unknown> }).stats, ms: 0 };
}

test('WorkerEngine.handleAsync reproduces handle exactly, request by request, bias table included', async () => {
  const requests: WorkerRequest[] = [
    { id: 1, fen: ADVANCED_POSITION, sims: 16 },
    { id: 2, fen: ADVANCED_POSITION, sims: 16 },
  ];

  const syncEngine = new WorkerEngine();
  const syncReplies = requests.map((r) => syncEngine.handle(r));

  const asyncEngine = new WorkerEngine();
  const asyncReplies: WorkerReply[] = [];
  for (const r of requests) asyncReplies.push(await asyncEngine.handleAsync(r));

  for (let i = 0; i < requests.length; i++) {
    const sync = syncReplies[i];
    const async_ = asyncReplies[i];
    assert.ok(isOk(sync) && isOk(async_), `request ${i}: ${JSON.stringify({ sync, async_ })}`);
    if (!isOk(sync) || !isOk(async_)) continue;
    assert.equal(async_.action, sync.action, `request ${i}: action`);
    assert.equal(async_.san, sync.san, `request ${i}: san`);
    assert.deepEqual(statsOf(async_), statsOf(sync), `request ${i}: stats`);
  }

  const syncTree = syncEngine.tree;
  const asyncTree = asyncEngine.tree;
  assert.ok(syncTree !== null && asyncTree !== null);
  const syncBias = syncTree!.bias.keys().map((k) => [k, Array.from(syncTree!.bias.entry(k)!.errSum), syncTree!.bias.entry(k)!.weightSum]);
  const asyncBias = asyncTree!.bias.keys().map((k) => [k, Array.from(asyncTree!.bias.entry(k)!.errSum), asyncTree!.bias.entry(k)!.weightSum]);
  assert.deepEqual(asyncBias, syncBias, 'the bias table carried across requests matches the sync path exactly');
});

test('a mixed sync/async request sequence on one engine carries the same retained state as all-sync', async () => {
  const requests: WorkerRequest[] = [
    { id: 1, fen: ADVANCED_POSITION, sims: 16 },
    { id: 2, fen: ADVANCED_POSITION, sims: 16 },
  ];

  const allSync = new WorkerEngine();
  const r1 = allSync.handle(requests[0]);
  const r2 = allSync.handle(requests[1]);

  const mixed = new WorkerEngine();
  const m1 = await mixed.handleAsync(requests[0]);
  const m2 = mixed.handle(requests[1]);

  assert.ok(isOk(r1) && isOk(r2) && isOk(m1) && isOk(m2));
  if (!isOk(r1) || !isOk(r2) || !isOk(m1) || !isOk(m2)) return;
  assert.equal(m1.action, r1.action);
  assert.equal(m2.action, r2.action);
  assert.deepEqual(statsOf(m2), statsOf(r2));
});

// Direct proof of resumability (not replay): a playout that needs an
// evaluation must stop exactly there and must NOT have written the new node
// into the tree until the evaluator's promise actually settles. A
// replay-style bridge (throw on miss, re-run the whole search from scratch
// once the answer is known) could still pass the bit-exactness tests above
// by accident; this test only passes if the generator genuinely suspended.
test('playoutAsync genuinely suspends at the evaluation and resumes with no extra work', async () => {
  const config: MatchConfig = { ...BASE_CONFIG, playouts: 1 };
  type FakeEval = ReturnType<ReturnType<typeof makeFakeMlhEvaluator>>;
  let resolveEval: ((value: FakeEval) => void) | null = null;
  let calls = 0;
  const inner = makeFakeMlhEvaluator(ACTION_SIZE);
  const deferredEvaluator = (state: MatchGameState): Promise<FakeEval> => {
    calls += 1;
    return new Promise<FakeEval>((resolve) => {
      resolveEval = resolve;
    });
  };
  const tree = new MatchTree(config, wrapMlhEvaluatorAsync(deferredEvaluator, { lam: 0, tau0: 0.2, tau1: 0.4, slope: 32 }, new MlhContext()));
  const state = rootState();

  const ensurePromise = tree.ensureRootAsync(state);
  // Not resolved yet: the tree must not have a root until the promise settles.
  assert.equal(tree.root, null, 'the root must not exist before the evaluator resolves');
  assert.equal(calls, 1, 'exactly one evaluation was requested, not a retried batch');

  assert.ok(resolveEval !== null);
  resolveEval!(inner(state));
  await ensurePromise;
  assert.ok(tree.root !== null, 'the root exists once the evaluator resolves');
  assert.equal(calls, 1, 'resuming must not have asked the evaluator again');

  const playoutPromise = tree.playoutAsync(state);
  const nodesBefore = tree.nodeCount;
  // Give the microtask queue a turn without resolving the pending evaluation.
  await Promise.resolve();
  assert.equal(tree.nodeCount, nodesBefore, 'no new node before the second evaluation resolves');

  assert.ok(resolveEval !== null);
  resolveEval!(inner(state));
  await playoutPromise;
  assert.equal(calls, 2, 'the playout asked for exactly one more evaluation, not a replay of the first');
});
