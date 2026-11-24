// KOM-1214: a threaded wasm session create/warm-up that never settles (the
// nested-worker-start failure verified in the app pane -- an echo worker
// created inside a worker never runs, so the threaded InferenceSession.create
// neither resolves nor rejects) must time out and fail typed, instead of
// hanging forever. Same fake SessionFactory/StorageLike convention as
// backend.test.ts/worker_init.test.ts -- all fakes, no real ORT, so this
// file never skips.

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { ThreadsTimeoutError, backendCacheKey, probeBackend } from '../src/backend.ts';
import type { BackendRecord, ProbeSession, SessionFactory, StorageLike } from '../src/backend.ts';
import { ACTION_SIZE } from '../src/constants.ts';
import { makeFakeMlhEvaluator } from '../src/fake_evaluator.ts';
import type { MlhBatchEvaluator } from '../src/batch_evaluator.ts';
import { WorkerEngine } from '../src/engine.ts';
import type { WorkerInitEnv } from '../src/engine.ts';
import type { WorkerInitReply } from '../src/protocol.ts';

function isInitOk(
  reply: WorkerInitReply,
): reply is WorkerInitReply & { backend: string; threads: number; batch: number } {
  return !('err' in reply);
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

// A fake test timer: `fire()` runs every still-pending callback immediately
// (synchronously, in registration order) instead of waiting on a real
// setTimeout -- so a timeout test doesn't need to actually wait it out.
// Also records every `ms` a timer was requested at, so the threads === 1
// test can assert no timer is ever requested at all.
function makeFakeTimer(): {
  setTimer: (callback: () => void, ms: number) => () => void;
  requestedMs: number[];
  fire: () => void;
} {
  const requestedMs: number[] = [];
  let pending: Array<{ callback: () => void; cancelled: boolean }> = [];
  return {
    requestedMs,
    setTimer: (callback: () => void, ms: number): (() => void) => {
      requestedMs.push(ms);
      const entry = { callback, cancelled: false };
      pending.push(entry);
      return () => {
        entry.cancelled = true;
      };
    },
    fire: (): void => {
      const toFire = pending;
      pending = [];
      for (const entry of toFire) {
        if (!entry.cancelled) entry.callback();
      }
    },
  };
}

interface FakeFactoryOpts {
  // Threaded (numThreads > 1) createWasm never settles; a 1-thread create
  // always succeeds immediately. Mirrors the real nested-worker-start
  // failure (KOM-1214 ticket WHY): neither resolves nor rejects.
  threadedCreateHangs?: boolean;
  // Threaded create succeeds, but only after a real `succeedAfterMs` delay
  // (checks no behaviour change when threads work, just slowly).
  succeedAfterMs?: number;
}

interface FakeFactoryCalls {
  setNumThreadsCalls: number[];
  wasmCreate: number;
}

function makeFakeFactory(opts: FakeFactoryOpts = {}): { factory: SessionFactory; calls: FakeFactoryCalls } {
  const calls: FakeFactoryCalls = { setNumThreadsCalls: [], wasmCreate: 0 };
  let threads = 1;

  function fakeSession(): ProbeSession {
    return {
      async warmup(): Promise<void> {},
      async run(): Promise<void> {},
    };
  }

  const factory: SessionFactory = {
    setNumThreads(t: number): void {
      calls.setNumThreadsCalls.push(t);
      threads = t;
    },
    async createWebgpu(): Promise<ProbeSession> {
      throw new Error('fake: no webgpu adapter offered in this file');
    },
    async createWasm(): Promise<ProbeSession> {
      calls.wasmCreate++;
      if (threads > 1) {
        if (opts.threadedCreateHangs) {
          return new Promise<ProbeSession>(() => {}); // never resolves, never rejects
        }
        if (opts.succeedAfterMs !== undefined) {
          await new Promise((resolve) => setTimeout(resolve, opts.succeedAfterMs));
        }
      }
      return fakeSession();
    },
  };
  return { factory, calls };
}

function fakeBatchEvaluator(): MlhBatchEvaluator {
  const leaf = makeFakeMlhEvaluator(ACTION_SIZE);
  return (states) => states.map((state) => leaf(state));
}

function cleanBackend(): WorkerInitEnv['createBatchBackend'] {
  return async () => fakeBatchEvaluator();
}

test('AC1: probeBackend rejects with ThreadsTimeoutError and writes the failure mark to storage', async () => {
  const { factory } = makeFakeFactory({ threadedCreateHangs: true });
  const storage = makeMemoryStorage();

  // A real (short) timer: the fake `createWasm` genuinely never settles, so
  // this must actually wait out the timeout to see it fire.
  const probePromise = probeBackend({
    factory,
    isolated: true,
    hardwareConcurrency: 8,
    modelSha256: 'sha-timeout',
    storage,
    threadsTimeoutMs: 50,
  });

  await assert.rejects(probePromise, (error: unknown) => {
    assert.ok(error instanceof ThreadsTimeoutError, 'must reject with ThreadsTimeoutError');
    const err = error as ThreadsTimeoutError;
    assert.equal(err.code, 'threads-timeout');
    assert.equal(err.record.backend, 'wasm');
    assert.equal(err.record.threads, 1);
    assert.deepEqual(err.record.msPerPosition, {}, 'a failure mark carries no timings');
    return true;
  });

  const key = backendCacheKey({ modelSha256: 'sha-timeout', adapter: null, hardwareConcurrency: 8, isolated: true });
  const stored = storage.getItem(key);
  assert.ok(stored !== null, 'the failure mark must be persisted to storage');
  const record = JSON.parse(stored as string) as BackendRecord;
  assert.equal(record.backend, 'wasm');
  assert.equal(record.threads, 1);
  assert.deepEqual(record.msPerPosition, {});
});

test('AC1: WorkerEngine.init returns a threads-timeout error reply with a failure-mark probeRecord, within the timeout plus a margin', async () => {
  const engine = new WorkerEngine();
  const { factory } = makeFakeFactory({ threadedCreateHangs: true });

  const started = Date.now();
  const reply = await engine.init(
    { id: 1, modelUrl: 'model.onnx', modelSha256: 'sha-init-timeout', threadsTimeoutMs: 100 },
    { sessionFactory: factory, isolated: true, hardwareConcurrency: 8, createBatchBackend: cleanBackend() },
  );
  const elapsed = Date.now() - started;

  assert.ok(elapsed < 100 + 1000, `expected the reply within 1100ms, took ${elapsed}ms`);
  assert.ok('err' in reply, `expected an error reply, got ${JSON.stringify(reply)}`);
  if (!('err' in reply)) return;
  assert.equal(reply.code, 'threads-timeout');
  const probeRecord = reply.probeRecord as BackendRecord | undefined;
  assert.ok(probeRecord, 'the error reply must carry a probeRecord');
  assert.equal(probeRecord?.backend, 'wasm');
  assert.equal(probeRecord?.threads, 1);
  assert.deepEqual(probeRecord?.msPerPosition, {}, 'a failure mark carries no timings');
});

test('AC2: maxThreads 1 calls setNumThreads(1) exactly once and never takes the threaded (hanging) path', async () => {
  const engine = new WorkerEngine();
  const { factory, calls } = makeFakeFactory({ threadedCreateHangs: true });

  const reply = await engine.init(
    { id: 2, modelUrl: 'model.onnx', modelSha256: 'sha-max-1', maxThreads: 1 },
    { sessionFactory: factory, isolated: true, hardwareConcurrency: 8, createBatchBackend: cleanBackend() },
  );

  assert.ok(isInitOk(reply), `expected an ok reply, got ${JSON.stringify(reply)}`);
  if (!isInitOk(reply)) return;
  assert.equal(reply.backend, 'wasm');
  assert.equal(
    reply.threads,
    1,
    'maxThreads: 1 must force 1 thread even though hardwareConcurrency gives a higher default',
  );
  assert.deepEqual(calls.setNumThreadsCalls, [1], 'setNumThreads must be called exactly once, with 1');
  assert.equal(calls.wasmCreate, 1, 'the threaded (hanging) path must never be taken');
});

test('a threaded create that succeeds (slowly) is unaffected: same record as without the option', async () => {
  const { factory: factoryWithTimeout } = makeFakeFactory({ succeedAfterMs: 10 });
  const resultWithTimeout = await probeBackend({
    factory: factoryWithTimeout,
    isolated: true,
    hardwareConcurrency: 8,
    modelSha256: 'sha-no-change',
    threadsTimeoutMs: 5000,
  });

  const { factory: factoryWithoutOption } = makeFakeFactory({ succeedAfterMs: 10 });
  const resultWithoutOption = await probeBackend({
    factory: factoryWithoutOption,
    isolated: true,
    hardwareConcurrency: 8,
    modelSha256: 'sha-no-change',
  });

  assert.equal(resultWithTimeout.backend, resultWithoutOption.backend);
  assert.equal(resultWithTimeout.threads, resultWithoutOption.threads);
  assert.deepEqual(resultWithTimeout.record.msPerPosition, resultWithoutOption.record.msPerPosition);
});

test('a relayed cachedRecord for wasm at threads 4 whose create hangs fails threads-timeout, no re-probe', async () => {
  const { factory, calls } = makeFakeFactory({ threadedCreateHangs: true });
  const cachedRecord: BackendRecord = {
    backend: 'wasm',
    threads: 4,
    msPerPosition: { 4: 14 },
    chosenBatch: 4,
    probedAt: 0,
  };

  const probePromise = probeBackend({
    factory,
    isolated: true,
    hardwareConcurrency: 8,
    threadsOverride: 4,
    modelSha256: 'sha-cached-hang',
    cachedRecord,
    threadsTimeoutMs: 50,
  });

  await assert.rejects(probePromise, (error: unknown) => error instanceof ThreadsTimeoutError);
  assert.equal(calls.wasmCreate, 1, 'a threads-timeout must never trigger a second (re-probe) session creation');
});

test('AC5: threads === 1 never creates a timer', async () => {
  const { factory } = makeFakeFactory();
  const timer = makeFakeTimer();

  const result = await probeBackend({
    factory,
    isolated: false, // forces 1 thread regardless of hardwareConcurrency
    hardwareConcurrency: 8,
    modelSha256: 'sha-one-thread',
    setTimer: timer.setTimer,
  });

  assert.equal(result.threads, 1);
  assert.deepEqual(timer.requestedMs, [], 'no timer must ever be requested at threads === 1');
});

test('ThreadsTimeoutError is an Error with the typed code and carries the mark', () => {
  const record: BackendRecord = { backend: 'wasm', threads: 1, msPerPosition: {}, chosenBatch: 1, probedAt: 0 };
  const error = new ThreadsTimeoutError(record);
  assert.ok(error instanceof Error);
  assert.equal(error.code, 'threads-timeout');
  assert.equal(error.record, record);
});
