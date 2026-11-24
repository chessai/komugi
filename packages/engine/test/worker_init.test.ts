// Fast-mode U6b (docs/design/clientside-fast-mode.md section 3, 3.3,
// section 8 U6; ticket KOM-1095): WorkerEngine.init() -- U5's probe wired
// through the worker (fake navigator.gpu/SessionFactory/localStorage, same
// convention as backend.test.ts), and the mid-search backend-failure
// fallback (design 3.3 M-5: fresh MlhContext via prepareFast(), bias table
// restored from the N1 snapshot taken before the aborted attempt). All
// fakes -- no real navigator.gpu, no real ORT, no real model -- so this
// file never skips.

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { backendCacheKey } from '../src/backend.ts';
import type { BackendKind, BackendRecord, GpuLike, ProbeSession, SessionFactory, StorageLike } from '../src/backend.ts';
import { BackendError } from '../src/batch_evaluator.ts';
import type { MlhBatchEvaluator } from '../src/batch_evaluator.ts';
import { ACTION_SIZE, ADVANCED_POSITION } from '../src/constants.ts';
import { makeFakeMlhEvaluator } from '../src/fake_evaluator.ts';
import type { WorkerInitReply, WorkerReply, WorkerRequest } from '../src/protocol.ts';
import { WorkerEngine } from '../src/engine.ts';
import type { WorkerInitEnv } from '../src/engine.ts';

function isOk(reply: WorkerReply): reply is WorkerReply & { stats: Record<string, unknown> } {
  return !('err' in reply);
}

function isInitOk(
  reply: WorkerInitReply,
): reply is WorkerInitReply & {
  backend: BackendKind;
  threads: number;
  batch: number;
  fallback?: { from: BackendKind; reason: string };
} {
  return !('err' in reply);
}

// Drops `ms` (wall-clock) and `backend`/`fallback`/`probeRecord` (expected
// to differ between a direct run and one that got there by falling back,
// design 6.1 N-6 plus the M-5 test's own "excluding ms/backend/fallback"
// rule -- KOM-1101's `probeRecord` is stamped in the same "only on a
// fallback reply" category as `fallback`).
function statsExcludingBackend(reply: WorkerReply): Record<string, unknown> {
  const stats = { ...(reply as { stats: Record<string, unknown> }).stats };
  delete stats.ms;
  delete stats.backend;
  delete stats.fallback;
  delete stats.probeRecord;
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

// Same fake SessionFactory convention as backend.test.ts: this file drives
// the probe through WorkerEngine.init() instead of calling probeBackend()
// directly, so the fakes themselves are unchanged.
interface FakeFactoryOpts {
  webgpuCreateThrows?: boolean;
  webgpuMsPerPosition?: number;
  wasmMsPerPosition?: number;
}

interface FakeFactoryCalls {
  webgpuCreate: number;
  wasmCreate: number;
  webgpuRun: number;
  wasmRun: number;
}

// `now` must be the clock `probeBackend` itself reads (design 3.2's median
// timing) -- the real `Date.now()` moves far too little across these
// synchronous fakes to tell a 1ms backend from a 14ms one, which is the
// whole point of the ladder (preamble measurements).
function makeFakeFactory(
  opts: FakeFactoryOpts,
): { factory: SessionFactory; calls: FakeFactoryCalls; now: () => number } {
  const calls: FakeFactoryCalls = { webgpuCreate: 0, wasmCreate: 0, webgpuRun: 0, wasmRun: 0 };
  const webgpuMs = opts.webgpuMsPerPosition ?? 1;
  const wasmMs = opts.wasmMsPerPosition ?? 14;
  let clock = 0;

  const factory: SessionFactory = {
    setNumThreads(): void {
      // not asserted here -- S-2's own ordering is backend.test.ts's job.
    },
    async createWebgpu(): Promise<ProbeSession> {
      calls.webgpuCreate++;
      if (opts.webgpuCreateThrows) throw new Error('fake: webgpu session creation failed');
      return {
        async warmup(): Promise<void> {},
        async run(batch: number): Promise<void> {
          calls.webgpuRun++;
          clock += webgpuMs * batch;
        },
      };
    },
    async createWasm(): Promise<ProbeSession> {
      calls.wasmCreate++;
      return {
        async warmup(): Promise<void> {},
        async run(batch: number): Promise<void> {
          calls.wasmRun++;
          clock += wasmMs * batch;
        },
      };
    },
  };
  return { factory, calls, now: () => clock };
}

function realAdapterGpu(): GpuLike {
  return { requestAdapter: async () => ({ isFallbackAdapter: false }) };
}

// KOM-1101 review fix: resolves exactly once (init's own probe, so it picks
// webgpu) and REJECTS every call after that -- the shape `resolveAdapterForCache`
// sees when a fallback runs right after the GPU that just failed, since
// `gpu.requestAdapter()` is the one call most likely to reject at that point.
function requestAdapterRejectsAfterFirstGpu(): GpuLike {
  let calls = 0;
  return {
    requestAdapter: async () => {
      calls += 1;
      if (calls === 1) return { isFallbackAdapter: false };
      throw new Error('fake: requestAdapter rejected');
    },
  };
}

function makeMemoryStorage(): StorageLike {
  const map = new Map<string, string>();
  return {
    getItem: (key: string): string | null => (map.has(key) ? (map.get(key) as string) : null),
    setItem: (key: string, value: string): void => {
      map.set(key, value);
    },
  };
}

// A `createBatchBackend` whose 'webgpu' evaluator throws a BackendError
// once it has already answered `throwAfter` batches -- "a backend fails
// mid-search", not at the very first leaf -- and whose 'wasm' evaluator
// never fails. Both backends compute the SAME deterministic fake leaf
// (makeFakeMlhEvaluator is a pure function of the leaf's own position, so
// a fresh instance here and in any other engine agree row for row).
function throwingWebgpuBackend(throwAfter: number): WorkerInitEnv['createBatchBackend'] {
  let calls = 0;
  const leaf = makeFakeMlhEvaluator(ACTION_SIZE);
  return async (backend: BackendKind) => {
    if (backend !== 'webgpu') return fakeBatchEvaluator();
    return (states) => {
      calls += 1;
      if (calls > throwAfter) throw new BackendError('fake: webgpu device lost mid-search');
      return states.map((state) => leaf(state));
    };
  };
}

function cleanBackend(): WorkerInitEnv['createBatchBackend'] {
  return async () => fakeBatchEvaluator();
}

// Records every `(backend, threads)` a `createBatchBackend` builder is
// actually asked to build, in order -- same convention as
// `throwingWasmThreadsBackend.builtAt`, but wraps an arbitrary builder (here
// `throwingWebgpuBackend`) instead of defining a new fake from scratch.
// `WorkerStats` has no `threads` field, so this is the only way a test can
// pin which thread count a mid-search fallback actually rebuilt at.
function trackBuilds(
  builder: WorkerInitEnv['createBatchBackend'],
): { builder: WorkerInitEnv['createBatchBackend']; builtAt: Array<{ backend: BackendKind; threads: number }> } {
  const builtAt: Array<{ backend: BackendKind; threads: number }> = [];
  const wrapped: WorkerInitEnv['createBatchBackend'] = async (backend, threads) => {
    builtAt.push({ backend, threads });
    return builder(backend, threads);
  };
  return { builder: wrapped, builtAt };
}

// KOM-1096 item 1: `createBatchBackend` throws for one specific backend --
// unlike `throwingWebgpuBackend`, this fails at SESSION BUILD time (the
// first call, before any search), the failure `init()` itself must walk the
// ladder for.
function buildThrowsFor(failFor: BackendKind): WorkerInitEnv['createBatchBackend'] {
  return async (backend: BackendKind) => {
    if (backend === failFor) throw new Error(`fake: ${backend} session creation failed`);
    return fakeBatchEvaluator();
  };
}

// KOM-1096 item 2: a `createBatchBackend` whose wasm evaluator fails mid-
// search ONLY when built at more than one thread -- the 1-thread rebuild
// (the ladder's next rung) always succeeds. Records every `(backend,
// threads)` the ladder actually asked it to build, in order, so a test can
// pin that the retry really landed on wasm/1 thread.
function throwingWasmThreadsBackend(
  throwAfter: number,
): { builder: WorkerInitEnv['createBatchBackend']; builtAt: Array<{ backend: BackendKind; threads: number }> } {
  const builtAt: Array<{ backend: BackendKind; threads: number }> = [];
  let calls = 0;
  const leaf = makeFakeMlhEvaluator(ACTION_SIZE);
  const builder: WorkerInitEnv['createBatchBackend'] = async (backend: BackendKind, threads: number) => {
    builtAt.push({ backend, threads });
    if (backend === 'wasm' && threads > 1) {
      return (states) => {
        calls += 1;
        if (calls > throwAfter) throw new BackendError('fake: wasm multi-thread session lost mid-search');
        return states.map((state) => leaf(state));
      };
    }
    return (states) => states.map((state) => leaf(state));
  };
  return { builder, builtAt };
}

test('init: a fake WebGPU adapter wins the ladder -> backend webgpu', async () => {
  const engine = new WorkerEngine();
  const { factory, now } = makeFakeFactory({ webgpuMsPerPosition: 1, wasmMsPerPosition: 14 });

  const reply = await engine.init(
    { id: 1, modelUrl: 'model.onnx', modelSha256: 'sha-a' },
    { sessionFactory: factory, gpu: realAdapterGpu(), isolated: true, hardwareConcurrency: 8, now, createBatchBackend: cleanBackend() },
  );

  assert.ok(isInitOk(reply), `expected an ok reply, got ${JSON.stringify(reply)}`);
  if (!isInitOk(reply)) return;
  assert.equal(reply.backend, 'webgpu');
  assert.equal(engine.currentBackend, 'webgpu');
  assert.ok(engine.currentBatchEvaluator !== null);
});

test('init: WebGPU session creation throws -> falls back to wasm', async () => {
  const engine = new WorkerEngine();
  const { factory, calls, now } = makeFakeFactory({ webgpuCreateThrows: true, wasmMsPerPosition: 14 });

  const reply = await engine.init(
    { id: 2, modelUrl: 'model.onnx', modelSha256: 'sha-a' },
    { sessionFactory: factory, gpu: realAdapterGpu(), isolated: true, hardwareConcurrency: 8, now, createBatchBackend: cleanBackend() },
  );

  assert.ok(isInitOk(reply), `expected an ok reply, got ${JSON.stringify(reply)}`);
  if (!isInitOk(reply)) return;
  assert.equal(reply.backend, 'wasm');
  assert.equal(calls.wasmCreate, 1, 'the wasm fallback session must still be created');
});

test('init: a non-isolated context forces 1 thread, even with a threads override', async () => {
  const engine = new WorkerEngine();
  const { factory, now } = makeFakeFactory({ wasmMsPerPosition: 14 });

  const reply = await engine.init(
    { id: 3, modelUrl: 'model.onnx', modelSha256: 'sha-a', threads: 6 },
    { sessionFactory: factory, isolated: false, hardwareConcurrency: 16, now, createBatchBackend: cleanBackend() },
  );

  assert.ok(isInitOk(reply), `expected an ok reply, got ${JSON.stringify(reply)}`);
  if (!isInitOk(reply)) return;
  assert.equal(reply.backend, 'wasm');
  assert.equal(reply.threads, 1, 'non-isolated must force 1 thread regardless of override');
});

test("init: a user's thread override is honoured, clamped to 1..8", async () => {
  const engineMid = new WorkerEngine();
  const { factory: midFactory, now: midNow } = makeFakeFactory({ wasmMsPerPosition: 14 });
  const midReply = await engineMid.init(
    { id: 4, modelUrl: 'model.onnx', modelSha256: 'sha-a', threads: 3 },
    { sessionFactory: midFactory, isolated: true, hardwareConcurrency: 16, now: midNow, createBatchBackend: cleanBackend() },
  );
  assert.ok(isInitOk(midReply), `expected an ok reply, got ${JSON.stringify(midReply)}`);
  if (!isInitOk(midReply)) return;
  assert.equal(midReply.threads, 3);

  const engineHigh = new WorkerEngine();
  const { factory: highFactory, now: highNow } = makeFakeFactory({ wasmMsPerPosition: 14 });
  const highReply = await engineHigh.init(
    { id: 5, modelUrl: 'model.onnx', modelSha256: 'sha-a', threads: 20 },
    { sessionFactory: highFactory, isolated: true, hardwareConcurrency: 16, now: highNow, createBatchBackend: cleanBackend() },
  );
  assert.ok(isInitOk(highReply), `expected an ok reply, got ${JSON.stringify(highReply)}`);
  if (!isInitOk(highReply)) return;
  assert.equal(highReply.threads, 8, 'an override above 8 is clamped to the ceiling');
});

test('init: a second init with the same storage hits the cache and runs no timing', async () => {
  const engine = new WorkerEngine();
  const storage = makeMemoryStorage();
  const { factory, calls, now } = makeFakeFactory({ webgpuMsPerPosition: 1, wasmMsPerPosition: 14 });
  const env: WorkerInitEnv = {
    sessionFactory: factory,
    gpu: realAdapterGpu(),
    isolated: true,
    hardwareConcurrency: 8,
    storage,
    now,
    createBatchBackend: cleanBackend(),
  };

  const first = await engine.init({ id: 6, modelUrl: 'model.onnx', modelSha256: 'sha-a' }, env);
  assert.ok(isInitOk(first), `expected an ok reply, got ${JSON.stringify(first)}`);
  const runsAfterFirst = calls.webgpuRun + calls.wasmRun;
  assert.ok(runsAfterFirst > 0, 'the first init must run timing');

  const second = await engine.init({ id: 7, modelUrl: 'model.onnx', modelSha256: 'sha-a' }, env);
  assert.ok(isInitOk(second), `expected an ok reply, got ${JSON.stringify(second)}`);
  assert.equal(second.backend, first.backend);
  assert.equal(calls.webgpuRun + calls.wasmRun, runsAfterFirst, 'a cache hit must run no timing at all');

  // A 31-day-old entry re-probes (backend.ts's own rule); this file only
  // needs to show init() actually threads `storage`/`now` through to the
  // probe, which the cache-hit assertion above already pins.
});

test('M-5: a mid-search backend failure restores the bias table and retries fresh on the fallback backend', async () => {
  const request: WorkerRequest = { id: 20, fen: ADVANCED_POSITION, sims: 32, batch: 4 };

  const engine1 = new WorkerEngine();
  const { factory: factory1, now: now1 } = makeFakeFactory({ webgpuMsPerPosition: 1, wasmMsPerPosition: 14 });
  const init1 = await engine1.init(
    { id: 19, modelUrl: 'model.onnx', modelSha256: 'sha-a' },
    {
      sessionFactory: factory1,
      gpu: realAdapterGpu(),
      isolated: true,
      hardwareConcurrency: 8,
      now: now1,
      createBatchBackend: throwingWebgpuBackend(1),
    },
  );
  assert.ok(isInitOk(init1) && init1.backend === 'webgpu', `expected init to pick webgpu, got ${JSON.stringify(init1)}`);
  if (!isInitOk(init1)) return;

  const reply1 = await engine1.handleFast(request, engine1.currentBatchEvaluator!);
  assert.ok(isOk(reply1), `expected an ok reply, got ${JSON.stringify(reply1)}`);
  if (!isOk(reply1)) return;
  assert.equal(reply1.stats.backend, 'wasm', 'the retry must land on the fallback backend');
  assert.equal(typeof reply1.stats.fallback, 'string', 'a fallback reply must stamp a reason');

  // A clean engine, probed straight to wasm (no WebGPU adapter offered at
  // all), running the identical request from an identically empty table.
  const engine2 = new WorkerEngine();
  const { factory: factory2 } = makeFakeFactory({ wasmMsPerPosition: 14 });
  const init2 = await engine2.init(
    { id: 21, modelUrl: 'model.onnx', modelSha256: 'sha-a' },
    { sessionFactory: factory2, isolated: true, hardwareConcurrency: 8, createBatchBackend: cleanBackend() },
  );
  assert.ok(isInitOk(init2) && init2.backend === 'wasm', `expected init to pick wasm, got ${JSON.stringify(init2)}`);
  if (!isInitOk(init2)) return;

  const reply2 = await engine2.handleFast(request, engine2.currentBatchEvaluator!);
  assert.ok(isOk(reply2), `expected an ok reply, got ${JSON.stringify(reply2)}`);
  if (!isOk(reply2)) return;

  assert.equal(reply1.action, reply2.action);
  assert.equal(reply1.san, reply2.san);
  assert.equal(reply1.stamp, reply2.stamp);
  assert.deepEqual(statsExcludingBackend(reply1), statsExcludingBackend(reply2));

  const tree1 = engine1.tree;
  const tree2 = engine2.tree;
  assert.ok(tree1 !== null && tree2 !== null);
  if (tree1 === null || tree2 === null) return;
  assert.deepEqual(biasPlain(tree1.bias.entries), biasPlain(tree2.bias.entries));
});

// KOM-1096 item 1.
test('init: WebGPU session build throws -> falls back to wasm with a fallback: { from, reason }', async () => {
  const engine = new WorkerEngine();
  const { factory, now } = makeFakeFactory({ webgpuMsPerPosition: 1, wasmMsPerPosition: 14 });

  const reply = await engine.init(
    { id: 30, modelUrl: 'model.onnx', modelSha256: 'sha-b' },
    {
      sessionFactory: factory,
      gpu: realAdapterGpu(),
      isolated: true,
      hardwareConcurrency: 8,
      now,
      createBatchBackend: buildThrowsFor('webgpu'),
    },
  );

  assert.ok(isInitOk(reply), `expected an ok reply, got ${JSON.stringify(reply)}`);
  if (!isInitOk(reply)) return;
  assert.equal(reply.backend, 'wasm');
  assert.equal(engine.currentBackend, 'wasm');
  assert.ok(reply.fallback, 'expected a fallback reason on the init reply');
  assert.equal(reply.fallback?.from, 'webgpu');
  assert.match(reply.fallback?.reason ?? '', /webgpu session creation failed/);
});

// KOM-1096 item 2.
test('fallback ladder: a wasm multi-thread failure retries at wasm 1 thread, and the cache records it', async () => {
  const engine = new WorkerEngine();
  const { factory } = makeFakeFactory({ wasmMsPerPosition: 14 });
  const storage = makeMemoryStorage();
  const { builder, builtAt } = throwingWasmThreadsBackend(1);

  const initReply = await engine.init(
    { id: 40, modelUrl: 'model.onnx', modelSha256: 'sha-c' },
    { sessionFactory: factory, isolated: true, hardwareConcurrency: 8, storage, createBatchBackend: builder },
  );
  assert.ok(isInitOk(initReply) && initReply.backend === 'wasm' && initReply.threads > 1, `expected a multi-thread wasm init, got ${JSON.stringify(initReply)}`);
  if (!isInitOk(initReply)) return;
  const initialThreads = initReply.threads;

  const request: WorkerRequest = { id: 41, fen: ADVANCED_POSITION, sims: 32, batch: 4 };
  const reply = await engine.handleFast(request, engine.currentBatchEvaluator!);
  assert.ok(isOk(reply), `expected an ok reply, got ${JSON.stringify(reply)}`);
  if (!isOk(reply)) return;
  assert.equal(reply.stats.backend, 'wasm');
  assert.equal(typeof reply.stats.fallback, 'string');

  const lastBuild = builtAt[builtAt.length - 1];
  assert.deepEqual(lastBuild, { backend: 'wasm', threads: 1 }, 'the retry must rebuild at wasm 1 thread');

  const key = backendCacheKey({ modelSha256: 'sha-c', adapter: null, hardwareConcurrency: 8, isolated: true });
  const cached = storage.getItem(key);
  assert.ok(cached !== null, 'the fallback must be recorded in the U5 cache');
  const record = JSON.parse(cached as string) as BackendRecord;
  assert.equal(record.backend, 'wasm');
  assert.equal(record.threads, 1);
  assert.notEqual(record.threads, initialThreads);
});

// KOM-1096 item 4.
// KOM-1101.
test('init: a valid relayed cachedRecord skips timing and is echoed as probeRecord', async () => {
  const engine = new WorkerEngine();
  const { factory, calls, now } = makeFakeFactory({ webgpuMsPerPosition: 1, wasmMsPerPosition: 14 });
  const key = backendCacheKey({ modelSha256: 'sha-relay', adapter: null, hardwareConcurrency: 8, isolated: true });
  const cachedRecord: BackendRecord = {
    backend: 'wasm',
    threads: 4,
    msPerPosition: { 4: 14 },
    chosenBatch: 4,
    probedAt: now(),
    key,
  };

  const reply = await engine.init(
    { id: 60, modelUrl: 'model.onnx', modelSha256: 'sha-relay', cachedRecord },
    { sessionFactory: factory, isolated: true, hardwareConcurrency: 8, now, createBatchBackend: cleanBackend() },
  );

  assert.ok(isInitOk(reply), `expected an ok reply, got ${JSON.stringify(reply)}`);
  if (!isInitOk(reply)) return;
  assert.equal(reply.backend, 'wasm');
  assert.equal(calls.webgpuRun + calls.wasmRun, 0, 'a relayed cache hit must run no timing');
  assert.deepEqual(reply.probeRecord, cachedRecord, 'the relayed record must be echoed back as probeRecord');
});

// KOM-1101.
test('init: an expired relayed cachedRecord re-probes and returns a fresh probeRecord', async () => {
  const engine = new WorkerEngine();
  const { factory, calls, now } = makeFakeFactory({ wasmMsPerPosition: 14 });
  const key = backendCacheKey({ modelSha256: 'sha-expired', adapter: null, hardwareConcurrency: 8, isolated: true });
  const staleRecord: BackendRecord = {
    backend: 'wasm',
    threads: 4,
    msPerPosition: { 4: 99 },
    chosenBatch: 4,
    probedAt: -40 * 24 * 60 * 60 * 1000, // 40 days before this fake clock's epoch -- past the 30-day maxAge
    key,
  };

  const reply = await engine.init(
    { id: 61, modelUrl: 'model.onnx', modelSha256: 'sha-expired', cachedRecord: staleRecord },
    { sessionFactory: factory, isolated: true, hardwareConcurrency: 8, now, createBatchBackend: cleanBackend() },
  );

  assert.ok(isInitOk(reply), `expected an ok reply, got ${JSON.stringify(reply)}`);
  if (!isInitOk(reply)) return;
  assert.ok(calls.wasmRun > 0, 'an expired relayed record must re-probe with real timing');
  assert.equal(reply.probeRecord.backend, 'wasm');
  assert.notEqual(reply.probeRecord.probedAt, staleRecord.probedAt, 'a fresh probeRecord must carry a new probedAt');
});

// KOM-1101.
test('handleFast: after a mid-search backend fallback the reply carries an updated probeRecord', async () => {
  const request: WorkerRequest = { id: 70, fen: ADVANCED_POSITION, sims: 32, batch: 4 };
  const engine = new WorkerEngine();
  const { factory, now } = makeFakeFactory({ webgpuMsPerPosition: 1, wasmMsPerPosition: 14 });

  const init = await engine.init(
    { id: 69, modelUrl: 'model.onnx', modelSha256: 'sha-fb' },
    {
      sessionFactory: factory,
      gpu: realAdapterGpu(),
      isolated: true,
      hardwareConcurrency: 8,
      now,
      createBatchBackend: throwingWebgpuBackend(1),
    },
  );
  assert.ok(isInitOk(init) && init.backend === 'webgpu', `expected init to pick webgpu, got ${JSON.stringify(init)}`);
  if (!isInitOk(init)) return;

  const reply = await engine.handleFast(request, engine.currentBatchEvaluator!);
  assert.ok(isOk(reply), `expected an ok reply, got ${JSON.stringify(reply)}`);
  if (!isOk(reply)) return;
  assert.equal(reply.stats.backend, 'wasm', 'the retry must land on the fallback backend');
  assert.equal(typeof reply.stats.fallback, 'string', 'a fallback reply must stamp a reason');
  const probeRecord = reply.stats.probeRecord as BackendRecord | undefined;
  assert.ok(probeRecord, 'a mid-search fallback reply must carry an updated probeRecord to persist');
  assert.equal(probeRecord?.backend, 'wasm', 'the record must mark the backend the search fell back to');
});

// KOM-1101 review fix: `writeFallbackCache` must never discard a rung whose
// evaluator already built fine just because `resolveAdapterForCache`'s
// `gpu.requestAdapter()` rejects (the common case: a fallback happens right
// after the GPU backend failed). Before the fix this rejection escaped
// `fallbackLadder`'s `try`, so the loop treated the built wasm evaluator as
// failed and stepped past it to wasm 1 thread.
test('KOM-1101 review fix: a rejecting requestAdapter during a mid-search fallback must not discard the built evaluator', async () => {
  const request: WorkerRequest = { id: 70, fen: ADVANCED_POSITION, sims: 32, batch: 4 };
  const engine = new WorkerEngine();
  const { factory, now } = makeFakeFactory({ webgpuMsPerPosition: 1, wasmMsPerPosition: 14 });
  const gpu = requestAdapterRejectsAfterFirstGpu();
  const { builder: trackedBuilder, builtAt } = trackBuilds(throwingWebgpuBackend(1));

  const init = await engine.init(
    { id: 69, modelUrl: 'model.onnx', modelSha256: 'sha-reject-1' },
    {
      sessionFactory: factory,
      gpu,
      isolated: true,
      hardwareConcurrency: 8,
      now,
      createBatchBackend: trackedBuilder,
    },
  );
  assert.ok(isInitOk(init) && init.backend === 'webgpu', `expected init to pick webgpu, got ${JSON.stringify(init)}`);
  if (!isInitOk(init)) return;
  const webgpuThreads = init.threads;

  const reply = await engine.handleFast(request, engine.currentBatchEvaluator!);
  assert.ok(isOk(reply), `expected an ok reply, got ${JSON.stringify(reply)}`);
  if (!isOk(reply)) return;
  assert.equal(reply.stats.backend, 'wasm', 'the fallback must land on the FIRST next rung, not be stepped past');
  assert.deepEqual(
    builtAt,
    [
      { backend: 'webgpu', threads: webgpuThreads },
      { backend: 'wasm', threads: webgpuThreads },
    ],
    'the first next rung keeps the same thread count as the failed webgpu rung, not stepped past to 1 thread',
  );
  assert.equal(typeof reply.stats.fallback, 'string', 'a fallback reply must stamp a reason');
  assert.equal(reply.stats.probeRecord, undefined, 'the record must be null (absent) when requestAdapter rejected');
});

// KOM-1101 review fix, init's own session-build fallback path (the other
// `fallbackLadder` call site).
test("KOM-1101 review fix: a rejecting requestAdapter during init's own fallback settles on the first next rung, and probeRecord falls back to the probe's own record", async () => {
  const engine = new WorkerEngine();
  const { factory, now } = makeFakeFactory({ webgpuMsPerPosition: 1, wasmMsPerPosition: 14 });
  const gpu = requestAdapterRejectsAfterFirstGpu();

  const reply = await engine.init(
    { id: 71, modelUrl: 'model.onnx', modelSha256: 'sha-reject-2' },
    {
      sessionFactory: factory,
      gpu,
      isolated: true,
      hardwareConcurrency: 8,
      now,
      createBatchBackend: buildThrowsFor('webgpu'),
    },
  );

  assert.ok(isInitOk(reply), `expected an ok reply, got ${JSON.stringify(reply)}`);
  if (!isInitOk(reply)) return;
  assert.equal(reply.backend, 'wasm', 'the ladder must settle on the FIRST next rung, not be stepped past');
  assert.ok(reply.fallback, 'expected a fallback reason on the init reply');
  assert.equal(reply.fallback?.from, 'webgpu');
  assert.ok(reply.probeRecord, "probeRecord must fall back to the probe's own record when the ladder write failed");
  assert.equal(reply.probeRecord?.backend, 'webgpu', "the probe's own record reflects the originally probed backend");
});

// KOM-1101.
test('init: no cachedRecord still uses StorageLike as before, and probeRecord mirrors the stored record', async () => {
  const engine = new WorkerEngine();
  const storage = makeMemoryStorage();
  const { factory, calls, now } = makeFakeFactory({ wasmMsPerPosition: 14 });
  const env: WorkerInitEnv = {
    sessionFactory: factory,
    isolated: true,
    hardwareConcurrency: 8,
    storage,
    now,
    createBatchBackend: cleanBackend(),
  };

  const first = await engine.init({ id: 80, modelUrl: 'model.onnx', modelSha256: 'sha-d' }, env);
  assert.ok(isInitOk(first), `expected an ok reply, got ${JSON.stringify(first)}`);
  if (!isInitOk(first)) return;
  const runsAfterFirst = calls.wasmRun;
  assert.ok(runsAfterFirst > 0, 'the first init must run timing');

  const second = await engine.init({ id: 81, modelUrl: 'model.onnx', modelSha256: 'sha-d' }, env);
  assert.ok(isInitOk(second), `expected an ok reply, got ${JSON.stringify(second)}`);
  if (!isInitOk(second)) return;
  assert.equal(calls.wasmRun, runsAfterFirst, 'a StorageLike cache hit with no cachedRecord must still skip timing');
  assert.deepEqual(second.probeRecord, first.probeRecord, 'probeRecord must mirror the stored record on a cache hit');
});

test('queueInit: an init queued behind a slow search runs only after it', async () => {
  const engine = new WorkerEngine();
  let searchSettled = false;

  const slowEvaluator: MlhBatchEvaluator = async (states) => {
    const leaf = makeFakeMlhEvaluator(ACTION_SIZE);
    await new Promise((resolve) => setTimeout(resolve, 10));
    const rows = states.map((state) => leaf(state));
    searchSettled = true;
    return rows;
  };

  const request: WorkerRequest = { id: 50, fen: ADVANCED_POSITION, sims: 4, batch: 1 };
  const searchPromise = engine.handleFast(request, slowEvaluator);

  const { factory, now } = makeFakeFactory({ wasmMsPerPosition: 14 });
  let searchSettledWhenInitRan = false;
  const initPromise = engine
    .queueInit(
      { id: 51, modelUrl: 'model.onnx', modelSha256: 'sha-d' },
      { sessionFactory: factory, isolated: true, hardwareConcurrency: 8, now, createBatchBackend: cleanBackend() },
    )
    .then((reply) => {
      searchSettledWhenInitRan = searchSettled;
      return reply;
    });

  const [searchReply, initReply] = await Promise.all([searchPromise, initPromise]);
  assert.ok(isOk(searchReply), `expected an ok search reply, got ${JSON.stringify(searchReply)}`);
  assert.ok(isInitOk(initReply), `expected an ok init reply, got ${JSON.stringify(initReply)}`);
  assert.ok(searchSettledWhenInitRan, 'queueInit must not run until the queued search settles');
});
