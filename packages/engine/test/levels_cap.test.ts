// KOM-1202 unit 2c (design docs/design/kom1202-levels.md section 5, 7): AC3's
// cap test for all four levels, on top of KOM-1220's merged chunk loop
// (1beba68c) -- a capped search walks the playout budget in chunks of the
// request's `batch`, checking the clock between chunks. Also the
// deadline-carry fix (F8): a mid-search backend fallback must keep the
// ORIGINAL deadline, not start a fresh `maxMs` budget on the retry. Fake
// evaluators throughout, same convention as deadline_batch.test.ts and
// worker_init.test.ts -- no real model, no network.

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { BackendError } from '../src/batch_evaluator.ts';
import type { MlhBatchEvaluator } from '../src/batch_evaluator.ts';
import type { GpuLike, ProbeSession, SessionFactory } from '../src/backend.ts';
import { ACTION_SIZE, ADVANCED_POSITION } from '../src/constants.ts';
import { makeFakeMlhEvaluator } from '../src/fake_evaluator.ts';
import { LEVEL_NAMES, LEVELS } from '../src/levels.ts';
import type { WorkerReply, WorkerRequest } from '../src/protocol.ts';
import { WorkerEngine } from '../src/engine.ts';
import type { WorkerInitEnv } from '../src/engine.ts';

function isOk(reply: WorkerReply): reply is WorkerReply & { stats: Record<string, unknown> } {
  return !('err' in reply);
}

function realAdapterGpu(): GpuLike {
  return { requestAdapter: async () => ({ isFallbackAdapter: false }) };
}

// A probe-time-only fake: its `clock` is independent of the engine's own
// `this.clock` (the one `maxMs` deadlines read) -- this one only has to make
// webgpu look faster than wasm so init()'s ladder picks webgpu.
function fakeProbeFactory(): { factory: SessionFactory; now: () => number } {
  let clock = 0;
  const factory: SessionFactory = {
    setNumThreads(): void {},
    async createWebgpu(): Promise<ProbeSession> {
      return {
        async warmup(): Promise<void> {},
        async run(batch: number): Promise<void> {
          clock += 1 * batch;
        },
      };
    },
    async createWasm(): Promise<ProbeSession> {
      return {
        async warmup(): Promise<void> {},
        async run(batch: number): Promise<void> {
          clock += 14 * batch;
        },
      };
    },
  };
  return { factory, now: () => clock };
}

test("AC3: each level's capMs truncates the search within one batch chunk of the deadline", async () => {
  for (const name of LEVEL_NAMES) {
    const def = LEVELS[name];
    const capMs = def.capMs;
    const batch = def.batch.webgpu;
    // Review F3: d does not divide the cap, giving exactly 10 evaluator
    // calls (including the root's own single-row call) for every level.
    const d = capMs / 10 + 1;

    let now = 0;
    const engine = new WorkerEngine({ clock: () => now });
    const leaf = makeFakeMlhEvaluator(ACTION_SIZE);
    const sizes: number[] = [];
    const batchEvaluator: MlhBatchEvaluator = (states) => {
      sizes.push(states.length);
      now += d;
      return states.map((state) => leaf(state));
    };
    // No `history` (review V4): pondering is on by default and starts
    // whenever `request.history` is an array, which would move the fake
    // clock and break the exact call-count assertion below.
    const request: WorkerRequest = { id: `cap-${name}`, fen: ADVANCED_POSITION, sims: 100_000, batch, maxMs: capMs };

    const reply = await engine.handleFast(request, batchEvaluator);
    assert.ok(isOk(reply), `[${name}] expected an ok reply, got ${JSON.stringify(reply)}`);
    if (!isOk(reply)) continue;

    assert.equal(reply.stats.deadlineHit, true, `[${name}] deadline must fire well before sims is reached`);
    assert.ok(
      now >= capMs && now < capMs + d,
      `[${name}] elapsed ${now} must land in [${capMs}, ${capMs + d})`,
    );
    assert.equal(
      sizes.length,
      Math.ceil(capMs / d),
      `[${name}] expected exactly ceil(capMs/d) evaluator calls, got ${JSON.stringify(sizes)}`,
    );
    assert.ok(
      (reply.stats.playouts as number) < 100_000,
      `[${name}] expected a partial playout count, got ${reply.stats.playouts}`,
    );

    // sizes[0] is the root's own single-row expansion; every call after it
    // is one chunk of the deadline loop (KOM-1220).
    const chunkSizes = sizes.slice(1);
    assert.ok(
      chunkSizes.every((n) => n <= batch),
      `[${name}] no chunk may exceed the requested batch of ${batch}, got ${JSON.stringify(sizes)}`,
    );
    assert.ok(
      chunkSizes.some((n) => n === batch),
      `[${name}] expected at least one full-batch chunk, got ${JSON.stringify(sizes)}`,
    );
  }
});

test('deadline-carry (review F8): a mid-search fallback keeps the original deadline, not a fresh one', async () => {
  for (const name of LEVEL_NAMES) {
    const def = LEVELS[name];
    const capMs = def.capMs;
    const batch = def.batch.webgpu;
    const d = capMs / 10 + 1;
    const throwAfter = 5; // about capMs / (2 * d) calls, review F8's own test shape

    let now = 0;
    const engine = new WorkerEngine({ clock: () => now });
    const { factory, now: probeNow } = fakeProbeFactory();
    const webgpuLeaf = makeFakeMlhEvaluator(ACTION_SIZE);
    const wasmLeaf = makeFakeMlhEvaluator(ACTION_SIZE);

    let webgpuCalls = 0;
    const createBatchBackend: WorkerInitEnv['createBatchBackend'] = async (backend) => {
      if (backend === 'webgpu') {
        return (states) => {
          webgpuCalls += 1;
          if (webgpuCalls > throwAfter) throw new BackendError('fake: webgpu device lost mid-search');
          now += d;
          return states.map((state) => webgpuLeaf(state));
        };
      }
      return (states) => {
        now += d;
        return states.map((state) => wasmLeaf(state));
      };
    };

    const initReply = await engine.init(
      { id: 1, type: 'init', modelUrl: 'model.onnx', modelSha256: `sha-${name}` },
      {
        sessionFactory: factory,
        gpu: realAdapterGpu(),
        isolated: true,
        hardwareConcurrency: 8,
        now: probeNow,
        createBatchBackend,
      },
    );
    assert.ok(
      !('err' in initReply) && initReply.backend === 'webgpu',
      `[${name}] expected init to pick webgpu, got ${JSON.stringify(initReply)}`,
    );

    const request: WorkerRequest = { id: `carry-${name}`, fen: ADVANCED_POSITION, sims: 100_000, batch, maxMs: capMs };
    const reply = await engine.handleFast(request, engine.currentBatchEvaluator!);
    assert.ok(isOk(reply), `[${name}] expected an ok reply, got ${JSON.stringify(reply)}`);
    if (!isOk(reply)) continue;

    assert.equal(typeof reply.stats.fallback, 'string', `[${name}] expected a fallback reply`);
    assert.equal(reply.stats.deadlineHit, true, `[${name}] deadline must fire on the fallback attempt`);
    assert.ok(
      now < capMs + d,
      `[${name}] elapsed ${now} must stay within capMs(${capMs}) + d(${d}) -- the fallback must not restart the deadline`,
    );
  }
});
