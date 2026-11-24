// KOM-1202 unit 2a (design docs/design/kom1202-levels.md section 3): the
// section 3 validation rules (1-5) through a real `handleFast`/`handle`.
// Fake init/evaluator throughout, same convention as worker_init.test.ts --
// no real model, no network.

import assert from 'node:assert/strict';
import { test } from 'node:test';

import type { SessionFactory } from '../src/backend.ts';
import { makeFakeMlhEvaluator } from '../src/fake_evaluator.ts';
import { ACTION_SIZE, ADVANCED_POSITION } from '../src/constants.ts';
import type { MlhBatchEvaluator } from '../src/batch_evaluator.ts';
import type { WorkerInitEnv } from '../src/engine.ts';
import { WorkerEngine } from '../src/engine.ts';
import type { WorkerLevelRequest, WorkerReply, WorkerRequest } from '../src/protocol.ts';

function isOk(reply: WorkerReply): reply is WorkerReply & { stats: Record<string, unknown> } {
  return !('err' in reply);
}

function fakeBatchEvaluator(): MlhBatchEvaluator {
  const leaf = makeFakeMlhEvaluator(ACTION_SIZE);
  return (states) => states.map((state) => leaf(state));
}

// Minimal fake session factory -- no `gpu` is ever passed to `init()` below,
// so the probe never attempts `createWebgpu`, and `isolated: false` forces
// 1 thread regardless of `hardwareConcurrency` (same convention as
// worker_init.test.ts's "non-isolated context" case). The settled backend
// is always wasm/1 thread; rule 4's tests only need "some" backend set.
function fakeSessionFactory(): SessionFactory {
  return {
    setNumThreads(): void {},
    async createWebgpu() {
      throw new Error('fake: webgpu must never be probed here');
    },
    async createWasm() {
      return {
        async warmup(): Promise<void> {},
        async run(): Promise<void> {},
      };
    },
  };
}

async function initWasm1t(engine: WorkerEngine): Promise<void> {
  const env: WorkerInitEnv = {
    sessionFactory: fakeSessionFactory(),
    isolated: false,
    hardwareConcurrency: 4,
    createBatchBackend: async () => fakeBatchEvaluator(),
  };
  const reply = await engine.init({ id: 'init', modelUrl: 'model.onnx', modelSha256: 'sha-a' }, env);
  assert.ok(!('err' in reply), `fake init failed: ${JSON.stringify(reply)}`);
}

test('rule 1: level plus sims, or neither, is a request error', async () => {
  const engine = new WorkerEngine();
  await initWasm1t(engine);
  const evaluator = fakeBatchEvaluator();

  const bothReply = await engine.handleFast(
    { id: 1, fen: ADVANCED_POSITION, sims: 16, level: 'casual' } as unknown as WorkerLevelRequest,
    evaluator,
  );
  assert.ok('err' in bothReply);
  assert.match(bothReply.err, /exactly one of sims, level, or analyze is required/);

  const neitherReply = await engine.handleFast(
    { id: 2, fen: ADVANCED_POSITION } as unknown as WorkerRequest,
    evaluator,
  );
  assert.ok('err' in neitherReply);
  assert.match(neitherReply.err, /exactly one of sims, level, or analyze is required/);
});

test('rule 2: level plus batch, or level plus maxMs, is a request error', async () => {
  const engine = new WorkerEngine();
  await initWasm1t(engine);
  const evaluator = fakeBatchEvaluator();

  const withBatch = await engine.handleFast(
    { id: 3, fen: ADVANCED_POSITION, level: 'casual', batch: 2 } as unknown as WorkerLevelRequest,
    evaluator,
  );
  assert.ok('err' in withBatch);
  assert.match(withBatch.err, /level cannot be combined with batch or maxMs/);

  const withMaxMs = await engine.handleFast(
    { id: 4, fen: ADVANCED_POSITION, level: 'casual', maxMs: 100 } as unknown as WorkerLevelRequest,
    evaluator,
  );
  assert.ok('err' in withMaxMs);
  assert.match(withMaxMs.err, /level cannot be combined with batch or maxMs/);
});

test('rule 3: an unknown level is a request error matching /unknown level/', async () => {
  const engine = new WorkerEngine();
  await initWasm1t(engine);
  const evaluator = fakeBatchEvaluator();

  const bad: unknown[] = ['Focused', 'champion', 7, '__proto__'];
  for (const [i, level] of bad.entries()) {
    const reply = await engine.handleFast(
      { id: `bad-${i}`, fen: ADVANCED_POSITION, level } as unknown as WorkerLevelRequest,
      evaluator,
    );
    assert.ok('err' in reply, `expected an error reply for level ${JSON.stringify(level)}`);
    assert.match(reply.err, /unknown level/, `level ${JSON.stringify(level)}`);
  }
});

test('rule 4: a level request before init() is a request error', async () => {
  const engine = new WorkerEngine();
  const reply = await engine.handleFast(
    { id: 5, fen: ADVANCED_POSITION, level: 'casual' } as unknown as WorkerLevelRequest,
    fakeBatchEvaluator(),
  );
  assert.ok('err' in reply);
  assert.match(reply.err, /level requires a selected backend; call init\(\) first/);
});

test('rule 5: handle()/handleAsync() reject a request that still carries level', () => {
  const engine = new WorkerEngine();
  const handleReply = engine.handle({
    id: 6,
    fen: ADVANCED_POSITION,
    sims: 16,
    level: 'casual',
  } as unknown as WorkerRequest);
  assert.ok('err' in handleReply);
  assert.match(handleReply.err, /level is only supported by handleFast/);
});

test('rule 5: handleAsync() rejects a request that still carries level', async () => {
  const engine = new WorkerEngine();
  const reply = await engine.handleAsync({
    id: 7,
    fen: ADVANCED_POSITION,
    sims: 16,
    level: 'casual',
  } as unknown as WorkerRequest);
  assert.ok('err' in reply);
  assert.match(reply.err, /level is only supported by handleFast/);
});

test('a bare {id, fen, level} request needs no sims -- stats.sims equals the level\'s playouts', async () => {
  const engine = new WorkerEngine();
  await initWasm1t(engine);
  const request: WorkerLevelRequest = { id: 8, fen: ADVANCED_POSITION, level: 'casual' };

  const reply = await engine.handleFast(request, fakeBatchEvaluator());
  assert.ok(isOk(reply), `expected an ok reply, got ${JSON.stringify(reply)}`);
  if (!isOk(reply)) return;
  assert.equal(reply.stats.sims, 100);
});
