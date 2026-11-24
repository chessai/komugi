// KOM-1220: under a deadline (`maxMs`), searchBatchedGen must chunk playouts
// by the request's `batch` (checking shouldStop between chunks), not force
// batch 1. Fake evaluator throughout, no fixture needed -- same convention
// as worker_fast.test.ts and ponder.test.ts.

import assert from 'node:assert/strict';
import { test } from 'node:test';

import type { MlhBatchEvaluator } from '../src/batch_evaluator.ts';
import { ACTION_SIZE, ADVANCED_POSITION } from '../src/constants.ts';
import { makeFakeMlhEvaluator } from '../src/fake_evaluator.ts';
import type { WorkerReply, WorkerRequest } from '../src/protocol.ts';
import { WorkerEngine } from '../src/engine.ts';

function isOk(reply: WorkerReply): reply is WorkerReply & { action: number; stats: Record<string, unknown> } {
  return !('err' in reply);
}

test('AC2: a far deadline at batch 8 evaluates in chunks of 8, spends the full budget, deadlineHit false', async () => {
  const engine = new WorkerEngine();
  const leaf = makeFakeMlhEvaluator(ACTION_SIZE);
  const sizes: number[] = [];
  const batchEvaluator: MlhBatchEvaluator = (states) => {
    sizes.push(states.length);
    return states.map((state) => leaf(state));
  };
  const request: WorkerRequest = { id: 1, fen: ADVANCED_POSITION, sims: 32, batch: 8, maxMs: 1_000_000_000 };

  const reply = await engine.handleFast(request, batchEvaluator);
  assert.ok(isOk(reply), `expected an ok reply, got ${JSON.stringify(reply)}`);
  if (!isOk(reply)) return;

  assert.equal(reply.stats.deadlineHit, false);
  assert.equal(reply.stats.playouts, 32, 'exactly config.playouts (sims) playouts must run');

  // sizes[0] is the root's own single-row expansion (always size 1); every
  // call after it is one chunk of the deadline loop.
  const chunkSizes = sizes.slice(1);
  assert.ok(
    chunkSizes.some((n) => n === 8),
    `expected at least one evaluator call of batch size 8, got ${JSON.stringify(sizes)}`,
  );
  assert.ok(
    chunkSizes.every((n) => n <= 8),
    `no chunk may exceed the requested batch of 8, got ${JSON.stringify(sizes)}`,
  );
});

test('AC3: a fake clock advanced per evaluator call stops within one batch-8 chunk of the deadline', async () => {
  const STEP = 5;
  const MAX_MS = 12;
  let now = 0;
  const engine = new WorkerEngine({ clock: () => now });
  const leaf = makeFakeMlhEvaluator(ACTION_SIZE);
  const sizes: number[] = [];
  const nowAfter: number[] = [];
  const batchEvaluator: MlhBatchEvaluator = (states) => {
    now += STEP;
    sizes.push(states.length);
    nowAfter.push(now);
    return states.map((state) => leaf(state));
  };
  const request: WorkerRequest = { id: 2, fen: ADVANCED_POSITION, sims: 1000, batch: 8, maxMs: MAX_MS };

  const reply = await engine.handleFast(request, batchEvaluator);
  assert.ok(isOk(reply), `expected an ok reply, got ${JSON.stringify(reply)}`);
  if (!isOk(reply)) return;

  assert.equal(reply.stats.deadlineHit, true, 'the deadline must fire well before sims=1000 is reached');
  assert.ok(
    (reply.stats.playouts as number) > 0 && (reply.stats.playouts as number) < 1000,
    `expected a partial playout count, got ${reply.stats.playouts}`,
  );

  const chunkSizes = sizes.slice(1);
  const chunkNowAfter = nowAfter.slice(1);
  assert.ok(
    chunkSizes.some((n) => n === 8),
    `expected at least one evaluator call of batch size 8, got ${JSON.stringify(sizes)}`,
  );

  // Replay the trace to find the playout count at the FIRST chunk whose
  // completion pushed the clock past the deadline (deadlineStart was 0: no
  // evaluator call had fired yet when prepareFast read the clock).
  let cumulative = 0;
  let crossingPlayouts: number | null = null;
  for (let i = 0; i < chunkSizes.length; i++) {
    cumulative += chunkSizes[i]!;
    if (chunkNowAfter[i]! - 0 >= MAX_MS) {
      crossingPlayouts = cumulative;
      break;
    }
  }
  assert.ok(crossingPlayouts !== null, 'the clock never crossed the deadline in this trace');
  assert.ok(
    (reply.stats.playouts as number) <= crossingPlayouts! + 8,
    `playouts (${reply.stats.playouts}) must stop within one chunk of 8 of the deadline crossing (${crossingPlayouts})`,
  );
});

test('AC: batch 1 with a deadline behaves exactly as before (mid-budget stop, same playouts and action)', async () => {
  const K = 6; // clock value (call count) at which shouldStop first fires
  let calls = 0;
  const engine = new WorkerEngine({ clock: () => calls });
  const leaf = makeFakeMlhEvaluator(ACTION_SIZE);
  const batchEvaluator: MlhBatchEvaluator = (states) => {
    calls += 1;
    return states.map((state) => leaf(state));
  };
  // sims is far larger than where the deadline fires, so this never comes
  // close to the F14 edge (shouldStop firing exactly on the last playout) --
  // batch=1 chunking is then byte-identical to the pre-KOM-1220 one-playout
  // loop, which this pins against a fixed, deterministic fake-evaluator run.
  const request: WorkerRequest = { id: 3, fen: ADVANCED_POSITION, sims: 1000, batch: 1, maxMs: K };

  const reply = await engine.handleFast(request, batchEvaluator);
  assert.ok(isOk(reply), `expected an ok reply, got ${JSON.stringify(reply)}`);
  if (!isOk(reply)) return;

  assert.equal(reply.stats.deadlineHit, true);
  assert.equal(reply.stats.playouts, 5, 'root call (calls=1) + 5 one-playout chunks reaches calls=6=K');
  assert.equal(
    reply.action,
    32871,
    'the fixed fake evaluator + position deterministically picks this root action',
  );
});
