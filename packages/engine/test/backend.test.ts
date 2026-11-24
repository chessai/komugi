// U5: backend probe, session factory, device cache (design doc section 3,
// section 8 Unit U5; ticket KOM-1089's ACCEPTANCE list, which supersedes
// section 8's commands; KOM-1098's operating-batch ladder and relayable
// cache). All fakes -- no real navigator.gpu, no real ORT,
// no real model -- so this file never skips.

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { defaultThreads, probeBackend } from '../src/backend.ts';
import type { BackendRecord, GpuLike, ProbeSession, SessionFactory, StorageLike } from '../src/backend.ts';

const THIRTY_ONE_DAYS_MS = 31 * 24 * 60 * 60 * 1000;

function makeClock(start = 0) {
  let t = start;
  return {
    now: (): number => t,
    advance: (ms: number): void => {
      t += ms;
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

interface FakeFactoryOpts {
  clock: { now: () => number; advance: (ms: number) => void };
  webgpuCreateThrows?: boolean;
  webgpuWarmupThrows?: boolean;
  webgpuMsPerPosition?: number; // default: fast enough to win the ladder
  wasmMsPerPosition?: number; // batch-1 cost
  wasmB4MsPerPosition?: number; // batch-4 cost, only exercised when isolated
  // KOM-1098: per-batch WebGPU cost (overrides webgpuMsPerPosition when set).
  webgpuCost?: (batch: number) => number;
  // KOM-1098: clock cost of every warm-up call, and of the first N runs of
  // each session (shader compile / first-shape allocation).
  warmupMs?: number;
  coldRuns?: number;
  coldRunMs?: number;
  wasmCreateThrows?: boolean;
}

interface FakeFactoryCalls {
  webgpuCreate: number;
  wasmCreate: number;
  webgpuRun: number;
  wasmRun: number;
  warmups: string[];
  setNumThreadsCalls: number[];
}

function makeFakeFactory(opts: FakeFactoryOpts): { factory: SessionFactory; calls: FakeFactoryCalls } {
  const calls: FakeFactoryCalls = {
    webgpuCreate: 0,
    wasmCreate: 0,
    webgpuRun: 0,
    wasmRun: 0,
    warmups: [],
    setNumThreadsCalls: [],
  };
  const webgpuMs = opts.webgpuMsPerPosition ?? 1;
  const wasmMs = opts.wasmMsPerPosition ?? 14;
  const wasmB4Ms = opts.wasmB4MsPerPosition ?? wasmMs;
  const webgpuCost = opts.webgpuCost ?? ((): number => webgpuMs);
  const coldRuns = opts.coldRuns ?? 0;
  const coldRunMs = opts.coldRunMs ?? 0;
  const warmupMs = opts.warmupMs ?? 0;

  const factory: SessionFactory = {
    setNumThreads(threads: number): void {
      calls.setNumThreadsCalls.push(threads);
    },
    async createWebgpu(): Promise<ProbeSession> {
      calls.webgpuCreate++;
      if (opts.webgpuCreateThrows) throw new Error('fake: webgpu session creation failed');
      let runs = 0;
      return {
        async warmup(bucket: number): Promise<void> {
          calls.warmups.push(`webgpu:${bucket}`);
          opts.clock.advance(warmupMs);
          if (opts.webgpuWarmupThrows) throw new Error('fake: webgpu warm-up failed');
        },
        async run(batch: number): Promise<void> {
          calls.webgpuRun++;
          if (runs++ < coldRuns) opts.clock.advance(coldRunMs);
          opts.clock.advance(webgpuCost(batch) * batch);
        },
      };
    },
    async createWasm(): Promise<ProbeSession> {
      calls.wasmCreate++;
      if (opts.wasmCreateThrows) throw new Error('fake: wasm multi-thread session creation failed');
      let runs = 0;
      return {
        async warmup(bucket: number): Promise<void> {
          calls.warmups.push(`wasm:${bucket}`);
          opts.clock.advance(warmupMs);
        },
        async run(batch: number): Promise<void> {
          calls.wasmRun++;
          if (runs++ < coldRuns) opts.clock.advance(coldRunMs);
          const perPosition = batch >= 4 ? wasmB4Ms : wasmMs;
          opts.clock.advance(perPosition * batch);
        },
      };
    },
  };
  return { factory, calls };
}

function realAdapterGpu(): GpuLike {
  return { requestAdapter: async () => ({ isFallbackAdapter: false }) };
}

test('requestAdapter retries once after a null, and picks WebGPU when it wins the ladder', async () => {
  const clock = makeClock();
  let requests = 0;
  const gpu: GpuLike = {
    requestAdapter: async () => {
      requests++;
      return requests === 1 ? null : { isFallbackAdapter: false };
    },
  };
  const { factory } = makeFakeFactory({ clock, webgpuMsPerPosition: 1, wasmMsPerPosition: 14 });

  const result = await probeBackend({
    factory,
    gpu,
    isolated: true,
    hardwareConcurrency: 8,
    modelSha256: 'sha-a',
    now: clock.now,
  });

  assert.equal(requests, 2, 'the first null requestAdapter() call must be retried once');
  assert.equal(result.backend, 'webgpu');
});

test('a fallback/software adapter is never used, even though it is a real non-null response', async () => {
  const clock = makeClock();
  const gpu: GpuLike = { requestAdapter: async () => ({ isFallbackAdapter: true }) };
  const { factory, calls } = makeFakeFactory({ clock, webgpuMsPerPosition: 1, wasmMsPerPosition: 14 });

  const result = await probeBackend({
    factory,
    gpu,
    isolated: true,
    hardwareConcurrency: 8,
    modelSha256: 'sha-a',
    now: clock.now,
  });

  assert.equal(result.backend, 'wasm');
  assert.equal(calls.webgpuCreate, 0, 'a fallback adapter must never reach WebGPU session creation');
});

test('WebGPU slower than wasm at batch 1 loses the ladder to wasm', async () => {
  const clock = makeClock();
  const { factory } = makeFakeFactory({ clock, webgpuMsPerPosition: 20, wasmMsPerPosition: 14 });

  const result = await probeBackend({
    factory,
    gpu: realAdapterGpu(),
    isolated: true,
    hardwareConcurrency: 8,
    modelSha256: 'sha-a',
    now: clock.now,
  });

  assert.equal(result.backend, 'wasm');
});

test('a WebGPU session/warm-up failure falls back to wasm', async () => {
  const clock = makeClock();
  const { factory, calls } = makeFakeFactory({ clock, webgpuWarmupThrows: true, wasmMsPerPosition: 14 });

  const result = await probeBackend({
    factory,
    gpu: realAdapterGpu(),
    isolated: true,
    hardwareConcurrency: 8,
    modelSha256: 'sha-a',
    now: clock.now,
  });

  assert.equal(result.backend, 'wasm');
  assert.equal(calls.wasmCreate, 1, 'the wasm fallback session must still be created');
});

test('non-isolated contexts never use wasm multi-thread, even with a threads override', async () => {
  const clock = makeClock();
  const { factory, calls } = makeFakeFactory({ clock, wasmMsPerPosition: 14 });

  const result = await probeBackend({
    factory,
    gpu: undefined,
    isolated: false,
    hardwareConcurrency: 16,
    threadsOverride: 6,
    modelSha256: 'sha-a',
    now: clock.now,
  });

  assert.equal(result.backend, 'wasm');
  assert.equal(result.threads, 1, 'non-isolated must force 1 thread regardless of override');
  assert.deepEqual(calls.setNumThreadsCalls, [1]);
});

test('thread default table: min(8, max(1, floor(hc / 2))) when isolated, else 1', () => {
  assert.equal(defaultThreads(4, true), 2);
  assert.equal(defaultThreads(8, true), 4);
  assert.equal(defaultThreads(16, true), 8);
  assert.equal(defaultThreads(32, true), 8);
  assert.equal(defaultThreads(4, false), 1);
  assert.equal(defaultThreads(32, false), 1);
});

test('a fresh cache hit skips timing but still warms every bucket', async () => {
  const clock = makeClock();
  const { factory, calls } = makeFakeFactory({ clock, webgpuMsPerPosition: 1, wasmMsPerPosition: 14 });
  const storage = makeMemoryStorage();
  const common = {
    factory,
    gpu: realAdapterGpu(),
    isolated: true,
    hardwareConcurrency: 8,
    modelSha256: 'sha-a',
    storage,
    now: clock.now,
  };

  const first = await probeBackend(common);
  assert.equal(first.fromCache, false);
  assert.equal(first.backend, 'webgpu');
  const runsAfterFirst = calls.webgpuRun + calls.wasmRun;
  assert.ok(runsAfterFirst > 0, 'the first probe must run timing');
  const warmupsAfterFirst = calls.warmups.length;

  const second = await probeBackend(common);
  assert.equal(second.fromCache, true);
  assert.equal(second.backend, 'webgpu');
  assert.equal(calls.webgpuRun + calls.wasmRun, runsAfterFirst, 'a cache hit must run no timing at all');
  // The first probe warms both candidates (it has to time both to run the
  // ladder); a cache hit only creates and warms the chosen backend's session.
  const webgpuBucketCount = calls.warmups.filter((w) => w.startsWith('webgpu:')).length / 2;
  assert.equal(
    calls.warmups.length,
    warmupsAfterFirst + webgpuBucketCount,
    'a cache hit must still warm every bucket of the chosen backend',
  );
});

test('a 31-day-old cache entry re-probes instead of reusing it', async () => {
  const clock = makeClock();
  const { factory, calls } = makeFakeFactory({ clock, webgpuMsPerPosition: 1, wasmMsPerPosition: 14 });
  const storage = makeMemoryStorage();
  const common = {
    factory,
    gpu: realAdapterGpu(),
    isolated: true,
    hardwareConcurrency: 8,
    modelSha256: 'sha-a',
    storage,
    now: clock.now,
  };

  const first = await probeBackend(common);
  assert.equal(first.fromCache, false);
  const runsAfterFirst = calls.webgpuRun + calls.wasmRun;

  clock.advance(THIRTY_ONE_DAYS_MS);
  const second = await probeBackend(common);

  assert.equal(second.fromCache, false, 'a 31-day-old entry must re-probe, not be reused');
  assert.ok(calls.webgpuRun + calls.wasmRun > runsAfterFirst, 're-probing must run timing again');
});

// KOM-1098: dev-box shape -- WebGPU B1 ~12 ms (loses to wasm B1 ~10 ms) but
// B8 ~1.25 ms/position, wasm B4 ~4.4 ms/position (faked as 4.5: binary-
// exact, so the fake clock's subtraction is exact).
function devBoxWebgpuCost(batch: number): number {
  return batch >= 8 ? 1.25 : 12;
}

test('KOM-1098: WebGPU losing at batch 1 but winning per position at batch 8 is chosen', async () => {
  const clock = makeClock();
  const { factory } = makeFakeFactory({
    clock,
    webgpuCost: devBoxWebgpuCost,
    wasmMsPerPosition: 10,
    wasmB4MsPerPosition: 4.5,
  });

  const result = await probeBackend({
    factory,
    gpu: realAdapterGpu(),
    isolated: true,
    hardwareConcurrency: 8,
    modelSha256: 'sha-a',
    now: clock.now,
  });

  assert.equal(result.backend, 'webgpu');
  assert.equal(result.chosenBatch, 8);
  assert.deepEqual(result.timings, {
    webgpu: { batch: 8, msPerPosition: 1.25 },
    wasm: { batch: 4, msPerPosition: 4.5 },
  });
});

test('KOM-1098: WebGPU winning at batch 1 but losing per position at batch 8 loses to wasm B4', async () => {
  const clock = makeClock();
  const { factory } = makeFakeFactory({
    clock,
    webgpuCost: (batch: number): number => (batch >= 8 ? 6 : 2),
    wasmMsPerPosition: 10,
    wasmB4MsPerPosition: 4.5,
  });

  const result = await probeBackend({
    factory,
    gpu: realAdapterGpu(),
    isolated: true,
    hardwareConcurrency: 8,
    modelSha256: 'sha-a',
    now: clock.now,
  });

  assert.equal(result.backend, 'wasm');
  assert.equal(result.chosenBatch, 4);
  assert.deepEqual(result.timings, {
    webgpu: { batch: 8, msPerPosition: 6 },
    wasm: { batch: 4, msPerPosition: 4.5 },
  });
});

test('KOM-1098: wasm on 1 thread is timed at batch 1', async () => {
  const clock = makeClock();
  const { factory } = makeFakeFactory({ clock, wasmMsPerPosition: 10, wasmB4MsPerPosition: 4.4 });

  const result = await probeBackend({
    factory,
    isolated: false,
    hardwareConcurrency: 8,
    modelSha256: 'sha-a',
    now: clock.now,
  });

  assert.equal(result.backend, 'wasm');
  assert.equal(result.chosenBatch, 1);
  assert.deepEqual(result.timings, { wasm: { batch: 1, msPerPosition: 10 } });
});

test('KOM-1098: warm-up runs (bucket warm-up and first cold runs) are excluded from the timing', async () => {
  const clock = makeClock();
  // Every warm-up and the first TWO runs of each session cost 500 ms:
  // counted, they would make WebGPU (1.25 ms/pos at B8) look slower than
  // wasm B4. Two cold runs, so the median of 3 alone cannot hide them --
  // only the untimed discard run in front of the samples does.
  const { factory } = makeFakeFactory({
    clock,
    webgpuCost: devBoxWebgpuCost,
    wasmMsPerPosition: 10,
    wasmB4MsPerPosition: 4.5,
    warmupMs: 500,
    coldRuns: 2,
    coldRunMs: 500,
  });

  const result = await probeBackend({
    factory,
    gpu: realAdapterGpu(),
    isolated: true,
    hardwareConcurrency: 8,
    modelSha256: 'sha-a',
    now: clock.now,
  });

  assert.equal(result.backend, 'webgpu');
  assert.equal(result.timings?.webgpu?.msPerPosition, 1.25);
  assert.equal(result.timings?.wasm?.msPerPosition, 4.5);
});

test('KOM-1098: a relayed record skips timing; an expired one re-probes', async () => {
  const clock = makeClock();
  const { factory, calls } = makeFakeFactory({ clock, webgpuCost: devBoxWebgpuCost, wasmMsPerPosition: 10 });
  const common = {
    factory,
    gpu: realAdapterGpu(),
    isolated: true,
    hardwareConcurrency: 8,
    modelSha256: 'sha-a',
    now: clock.now,
  };

  // No storage at all (a Web Worker): the probe hands back the record to persist.
  const first = await probeBackend(common);
  assert.equal(first.fromCache, false);
  assert.equal(first.record.backend, 'webgpu');
  assert.equal(typeof first.cacheKey, 'string');
  const relayed: BackendRecord = JSON.parse(JSON.stringify(first.record));
  const runsAfterFirst = calls.webgpuRun + calls.wasmRun;

  const second = await probeBackend({ ...common, cachedRecord: relayed });
  assert.equal(second.fromCache, true);
  assert.equal(second.backend, 'webgpu');
  assert.equal(calls.webgpuRun + calls.wasmRun, runsAfterFirst, 'a relayed record must run no timing');

  // A record relayed from another device shape (key mismatch) is a miss.
  const other = await probeBackend({ ...common, hardwareConcurrency: 16, cachedRecord: relayed });
  assert.equal(other.fromCache, false);
  const runsAfterOther = calls.webgpuRun + calls.wasmRun;
  assert.ok(runsAfterOther > runsAfterFirst);

  clock.advance(THIRTY_ONE_DAYS_MS);
  const third = await probeBackend({ ...common, cachedRecord: relayed });
  assert.equal(third.fromCache, false, 'an expired relayed record must re-probe');
  assert.ok(calls.webgpuRun + calls.wasmRun > runsAfterOther, 're-probing must run timing again');
  assert.equal(third.record.probedAt, clock.now());
});

test('KOM-1098: after a multi-thread failure the stored record keeps threads 1', async () => {
  const clock = makeClock();
  const storage = makeMemoryStorage();
  const failing = makeFakeFactory({ clock, wasmCreateThrows: true });
  const common = {
    isolated: true,
    hardwareConcurrency: 8,
    modelSha256: 'sha-a',
    storage,
    now: clock.now,
  };

  // The probe's own wasm multi-thread session fails: a threads=1 mark.
  const first = await probeBackend({ ...common, factory: failing.factory });
  assert.equal(first.backend, 'wasm');
  assert.equal(first.threads, 1);
  assert.equal(first.chosenBatch, 1);
  assert.equal(first.record.threads, 1);

  // Next start (default would be 4 threads): the mark keeps it at 1, both in
  // effect and in the record handed back / stored.
  const next = makeFakeFactory({ clock });
  const second = await probeBackend({ ...common, factory: next.factory });
  assert.equal(second.fromCache, true);
  assert.equal(second.threads, 1);
  assert.equal(second.chosenBatch, 1);
  assert.deepEqual(next.calls.setNumThreadsCalls, [1]);
  assert.equal(second.record.threads, 1);
  assert.equal(JSON.parse(storage.getItem(first.cacheKey) as string).threads, 1);

  // worker.ts's own fallback mark (wasm, threads 1, no timings), relayed.
  const mark: BackendRecord = { backend: 'wasm', threads: 1, msPerPosition: {}, chosenBatch: 1, probedAt: clock.now() };
  const relayedNext = makeFakeFactory({ clock });
  const third = await probeBackend({ ...common, storage: undefined, factory: relayedNext.factory, cachedRecord: mark });
  assert.equal(third.fromCache, true);
  assert.equal(third.threads, 1);
  assert.deepEqual(relayedNext.calls.setNumThreadsCalls, [1]);
  assert.equal(third.record.threads, 1);
});
