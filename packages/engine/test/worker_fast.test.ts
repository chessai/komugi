// Fast-mode U6a (docs/design/clientside-fast-mode.md section 8 U6; ticket
// KOM-1092): WorkerEngine.handleFast -- batch/maxMs wiring, the shared MLH
// context (S-1), the promise-chain request queue (M-4) and the bias
// snapshot/restore helper (N1). Fake evaluator throughout, same convention
// as worker.test.ts: real-net parity is a later unit's job.

import assert from 'node:assert/strict';
import { test } from 'node:test';

import type { MlhBatchEvaluator } from '../src/batch_evaluator.ts';
import { ACTION_SIZE, ADVANCED_POSITION } from '../src/constants.ts';
import { makeFakeMlhEvaluator } from '../src/fake_evaluator.ts';
import type { WorkerReply, WorkerRequest } from '../src/protocol.ts';
import { restoreBias, snapshotBias, WorkerEngine } from '../src/engine.ts';

function isOk(reply: WorkerReply): reply is WorkerReply & { stats: Record<string, unknown> } {
  return !('err' in reply);
}

function statsOf(reply: WorkerReply): Record<string, unknown> {
  return { ...(reply as { stats: Record<string, unknown> }).stats, ms: 0 };
}

// Maps the same fake leaf evaluator handle()'s own default installs
// (evaluator.ts's makeFakeMlhEvaluator) over a batch's rows, so a
// batch-1-vs-handle() comparison is an apples-to-apples identity, not a
// comparison against a different fake.
function fakeBatchEvaluator(): MlhBatchEvaluator {
  const leaf = makeFakeMlhEvaluator(ACTION_SIZE);
  return (states) => states.map((state) => leaf(state));
}

type BiasPlainEntry = [number, number[], number, number];

function biasPlain(table: Map<number, { errSum: Float64Array; weightSum: number; gen: number }>): BiasPlainEntry[] {
  return Array.from(table.entries())
    .sort(([a], [b]) => a - b)
    .map(([key, entry]) => [key, Array.from(entry.errSum), entry.weightSum, entry.gen]);
}

test('batch 1, no deadline: handleFast equals handle() bit for bit (excluding ms) -- two fresh engines (N-6)', async () => {
  const fast = new WorkerEngine();
  const plain = new WorkerEngine();
  const request: WorkerRequest = { id: 1, fen: ADVANCED_POSITION, sims: 16, batch: 1 };

  const fastReply = await fast.handleFast(request, fakeBatchEvaluator());
  const plainReply = plain.handle(request);

  assert.ok(isOk(fastReply), `expected an ok reply, got ${JSON.stringify(fastReply)}`);
  assert.ok(isOk(plainReply));
  if (!isOk(fastReply) || !isOk(plainReply)) return;
  assert.equal(fastReply.action, plainReply.action);
  assert.equal(fastReply.san, plainReply.san);
  assert.equal(fastReply.stamp, plainReply.stamp);
  assert.deepEqual(statsOf(fastReply), statsOf(plainReply));
});

test('S-1: production MLH (lam > 0) gating shared between the root call and batched leaves equals handle()', async () => {
  const fast = new WorkerEngine();
  const plain = new WorkerEngine();
  // tau0 near 0 and a short ramp make the gate fire on almost any nonzero
  // |w - l|, so this exercises applyMlh's shift branch, not just the lam<=0
  // (or below-tau0) no-op path.
  const request: WorkerRequest = {
    id: 2,
    fen: ADVANCED_POSITION,
    sims: 16,
    batch: 1,
    config: { mlh: { lam: 0.5, tau0: 0.0, tau1: 0.0001, slope: 1 } },
  };

  const fastReply = await fast.handleFast(request, fakeBatchEvaluator());
  const plainReply = plain.handle(request);

  assert.ok(isOk(fastReply) && isOk(plainReply));
  if (!isOk(fastReply) || !isOk(plainReply)) return;
  assert.equal(fastReply.action, plainReply.action);
  assert.deepEqual(statsOf(fastReply), statsOf(plainReply));
});

test('batch 8: reply schema carries deadlineHit alongside the usual stats', async () => {
  const engine = new WorkerEngine();
  const request: WorkerRequest = { id: 3, fen: ADVANCED_POSITION, sims: 32, batch: 8 };
  const reply = await engine.handleFast(request, fakeBatchEvaluator());
  assert.ok(isOk(reply), `expected an ok reply, got ${JSON.stringify(reply)}`);
  if (!isOk(reply)) return;
  const stats = reply.stats;
  assert.equal(stats.deadlineHit, false);
  assert.equal(stats.playouts, 32);
  assert.equal(typeof stats.evaluations, 'number');
  assert.equal(typeof stats.nodes, 'number');
  assert.ok(Array.isArray(stats.rootWdl) && stats.rootWdl.length === 3);
  assert.equal(typeof stats.q, 'number');
  assert.equal(typeof stats.ms, 'number');
  assert.equal(stats.discovery, 'on');
  assert.equal(stats.draftmin, 10);
  assert.equal(stats.mlh, '0.05/0.8/0.9/32');
});

test('C3: a maxMs deadline (fake clock) truncates the search and stamps deadlineHit, with the real partial playouts', async () => {
  let now = 0;
  const engine = new WorkerEngine({ clock: () => now });
  const request: WorkerRequest = { id: 4, fen: ADVANCED_POSITION, sims: 1000, batch: 4, maxMs: 5 };
  const leaf = makeFakeMlhEvaluator(ACTION_SIZE);
  let calls = 0;
  // Call 1 is the root's single-row expansion; every call after it is one
  // chunk of up to `batch` (4) playouts in the deadline loop (KOM-1220).
  // Once the clock crosses the deadline the loop must stop after exactly
  // one more completed chunk.
  const batchEvaluator: MlhBatchEvaluator = (states) => {
    calls += 1;
    if (calls > 1) now = 100;
    return states.map((state) => leaf(state));
  };

  const reply = await engine.handleFast(request, batchEvaluator);
  assert.ok(isOk(reply), `expected an ok reply, got ${JSON.stringify(reply)}`);
  if (!isOk(reply)) return;
  assert.equal(reply.stats.deadlineHit, true);
  assert.ok(
    (reply.stats.playouts as number) > 0 && (reply.stats.playouts as number) < 1000,
    `expected a partial playout count, got ${reply.stats.playouts}`,
  );
});

test('M-4: two overlapping handleFast calls are serialised, matching two sequentially awaited handle() calls', async () => {
  const fast = new WorkerEngine();
  const plain = new WorkerEngine();
  const first: WorkerRequest = { id: 10, fen: ADVANCED_POSITION, sims: 16, batch: 1 };
  const second: WorkerRequest = { id: 11, fen: ADVANCED_POSITION, sims: 16, batch: 1 };

  // The first request's evaluator genuinely suspends (a real microtask
  // delay), so an un-queued `handleFast` would let the second request's
  // tree.clear() run underneath the first's still-in-flight root expansion.
  const leaf = makeFakeMlhEvaluator(ACTION_SIZE);
  const delayedBatchEvaluator: MlhBatchEvaluator = async (states) => {
    await new Promise((resolve) => setTimeout(resolve, 0));
    return states.map((state) => leaf(state));
  };

  const firstPromise = fast.handleFast(first, delayedBatchEvaluator);
  const secondPromise = fast.handleFast(second, fakeBatchEvaluator());
  const [firstReply, secondReply] = await Promise.all([firstPromise, secondPromise]);

  const expectedFirst = plain.handle(first);
  const expectedSecond = plain.handle(second);

  assert.ok(isOk(firstReply) && isOk(secondReply) && isOk(expectedFirst) && isOk(expectedSecond));
  if (!isOk(firstReply) || !isOk(secondReply) || !isOk(expectedFirst) || !isOk(expectedSecond)) return;
  assert.equal(firstReply.action, expectedFirst.action);
  assert.deepEqual(statsOf(firstReply), statsOf(expectedFirst));
  assert.equal(secondReply.action, expectedSecond.action);
  assert.deepEqual(statsOf(secondReply), statsOf(expectedSecond));
});

test('N1: snapshotBias/restoreBias round-trip the bias table byte for byte, including after in-place updates', () => {
  const engine = new WorkerEngine();
  assert.ok(isOk(engine.handle({ id: 20, fen: ADVANCED_POSITION, sims: 16 })));
  const tree = engine.tree;
  assert.ok(tree !== null);
  if (tree === null) return;

  const snapshot = snapshotBias(tree);
  const snapshotPlain = biasPlain(snapshot);
  assert.ok(snapshotPlain.length > 0, 'the first search wrote the bias table');

  // A second search mutates entries in place (bias.ts update()'s own
  // contract): errSum/weightSum/gen change on the SAME Float64Array objects
  // a shallow copy would have shared.
  assert.ok(isOk(engine.handle({ id: 21, fen: ADVANCED_POSITION, sims: 16 })));
  assert.notDeepEqual(biasPlain(tree.bias.entries), snapshotPlain, 'the second search actually changed the table');

  restoreBias(tree, snapshot);
  assert.deepEqual(biasPlain(tree.bias.entries), snapshotPlain, 'restore reproduces the snapshot exactly');
  assert.deepEqual(biasPlain(snapshot), snapshotPlain, 'restoring did not retroactively mutate the kept snapshot');

  // Mutating the now-live, restored table must not move the kept snapshot --
  // restoreBias deep-copies on the way in, not just snapshotBias on the way
  // out.
  const liveEntry = tree.bias.entries.values().next().value;
  assert.ok(liveEntry !== undefined);
  if (liveEntry !== undefined) liveEntry.errSum[0] += 1000;
  assert.deepEqual(biasPlain(snapshot), snapshotPlain, 'the snapshot was not aliased by the live table');

  // A whole-map replace, not a merge: restoring an empty snapshot onto a
  // non-empty table must leave the table empty, not keep post-snapshot keys.
  restoreBias(tree, new Map());
  assert.equal(tree.bias.entries.size, 0, 'restore replaces the whole map');
});
