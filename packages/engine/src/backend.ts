// Backend selection, session factory and device-probe cache (design doc
// section 3, section 8 Unit U5). THREAD DEFAULT and the selection ladder
// below follow KOM-1089's ticket text, which supersedes section 3.1/3.2's
// min(4, hardwareConcurrency - 1) and the multi-batch estimate. KOM-1098:
// the ladder compares ms PER POSITION at each backend's OPERATING batch
// (WebGPU B8, wasm multi-thread B4, wasm 1 thread B1) -- batch-1 latency
// picked wasm on a box where WebGPU B8 is ~3x cheaper per position. Never
// a fallback/software adapter, wasm multi-thread only when crossOriginIsolated.

import type { OrtModule, OrtSession } from './ort_core.ts';

export type BackendKind = 'webgpu' | 'wasm';

// S-3 (design 3.2): every bucket is warmed untimed, cache hit or not, so a
// session is never cold at the first real move.
const BUCKETS: readonly number[] = [1, 2, 4, 8, 16];

const DEFAULT_MAX_AGE_MS = 30 * 24 * 60 * 60 * 1000;

// KOM-1214: how long a threads > 1 wasm session creation + warm-up is given
// before it is declared hung (the nested-worker-start failure this guards
// against neither resolves nor rejects, so a plain try/catch never sees it).
const DEFAULT_THREADS_TIMEOUT_MS = 10000;

// Structural shape of a WebGPU adapter -- only the two fields this module
// reads (navigator.gpu's real GPUAdapter has many more).
export interface GpuAdapterLike {
  readonly isFallbackAdapter?: boolean;
  readonly info?: { readonly vendor?: string; readonly architecture?: string };
}

export interface GpuLike {
  requestAdapter(): Promise<GpuAdapterLike | null>;
}

// One probe-time session: untimed warm-up at a bucket shape, and a timed
// run at a given batch. The real implementation (makeOrtSessionFactory
// below) backs both with one onnxruntime-web InferenceSession; tests fake
// this directly, so no real model/tensor ever has to exist for U5's own
// acceptance.
export interface ProbeSession {
  warmup(bucket: number): Promise<void>;
  run(batch: number): Promise<void>;
}

// Builds the two candidate sessions. `setNumThreads` is called exactly once,
// before either `createWebgpu` or `createWasm` (S-2: the wasm runtime --
// shared by the WebGPU JSEP backend -- reads numThreads once, at its own
// first initialisation).
export interface SessionFactory {
  setNumThreads(threads: number): void;
  createWebgpu(): Promise<ProbeSession>;
  createWasm(): Promise<ProbeSession>;
}

export interface StorageLike {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
}

// KOM-1098: the operating batch each backend is timed (and then run) at.
export const WEBGPU_BATCH = 8;
export const WASM_MT_BATCH = 4;

// One backend's probe measurement: median ms per position at `batch`.
export interface BackendTiming {
  batch: number;
  msPerPosition: number;
}

export interface BackendRecord {
  backend: BackendKind;
  threads: number;
  msPerPosition: Record<number, number>;
  chosenBatch: number;
  probedAt: number;
  // KOM-1098: per-backend timings of the probe that produced this record
  // (absent on worker.ts's fallback marks, which time nothing).
  timings?: { webgpu?: BackendTiming; wasm?: BackendTiming };
  // KOM-1098: the device-shape key this record was probed under, so a
  // relayed record (`cachedRecord`) from another device/model is a miss.
  key?: string;
}

export interface ProbeResult extends BackendRecord {
  fromCache: boolean;
  // KOM-1098 relayable cache: the record the host should persist (and
  // relay back as `cachedRecord` next start) under `cacheKey`.
  record: BackendRecord;
  cacheKey: string;
}

export interface ProbeOptions {
  factory: SessionFactory;
  gpu?: GpuLike | undefined;
  isolated: boolean;
  hardwareConcurrency: number;
  threadsOverride?: number | undefined;
  // KOM-1214: a hard ceiling applied after the default/override is computed
  // (threads = min(that, maxThreads), never below 1) -- a host's way to force
  // 1 thread (or any lower cap) on the NEXT worker after a threads-timeout,
  // without touching `threadsOverride`'s own meaning.
  maxThreads?: number | undefined;
  modelSha256: string;
  storage?: StorageLike | undefined;
  // KOM-1098: a Web Worker has no localStorage -- the host page reads the
  // record it persisted (ProbeResult.record) and relays it here. Takes
  // precedence over `storage`; same expiry/key checks.
  cachedRecord?: BackendRecord | null | undefined;
  now?: (() => number) | undefined;
  maxAgeMs?: number;
  // KOM-1214: how long a threads > 1 wasm session creation + warm-up may run
  // before it is declared hung. Default DEFAULT_THREADS_TIMEOUT_MS. Never
  // applied at threads === 1 (no timer is even created).
  threadsTimeoutMs?: number | undefined;
  // Test seam: overrides the timer backing the threads-timeout race, so
  // tests can observe/short-circuit it without a real setTimeout. Returns a
  // canceller; the default implementation wraps setTimeout/clearTimeout.
  setTimer?: ((callback: () => void, ms: number) => () => void) | undefined;
}

// Ticket THREAD DEFAULT (supersedes design 3.1's min(4, hc-1), F10 of the
// re-review): min(8, max(1, floor(hc / 2))) when cross-origin isolated,
// else 1 (wasm multi-thread needs SharedArrayBuffer, a hard capability gate
// no override can bypass). The site's device-tier benchmark found threads
// beyond the physical core count slower (4C/8T: 8 threads slower than 4),
// hence the floor(hc/2) rather than hc-1.
export function defaultThreads(hardwareConcurrency: number, isolated: boolean): number {
  if (!isolated) return 1;
  return Math.min(8, Math.max(1, Math.floor(hardwareConcurrency / 2)));
}

function effectiveThreads(opts: {
  hardwareConcurrency: number;
  isolated: boolean;
  threadsOverride?: number | undefined;
  maxThreads?: number | undefined;
}): number {
  if (!opts.isolated) return 1;
  const base = opts.threadsOverride ?? defaultThreads(opts.hardwareConcurrency, opts.isolated);
  const threads = Math.min(8, Math.max(1, Math.floor(base)));
  // KOM-1214: applied AFTER the default/override above -- a host-set ceiling
  // (e.g. maxThreads: 1 on the worker restarted after a threads-timeout),
  // never a replacement for it.
  if (opts.maxThreads === undefined) return threads;
  return Math.max(1, Math.min(threads, opts.maxThreads));
}

// KOM-1214: thrown when a threads > 1 wasm session create/warm-up does not
// settle within `threadsTimeoutMs` -- onnxruntime-web's wasm runtime reads
// numThreads once per worker, so a timed-out worker can never usefully
// retry itself; the host must start a FRESH worker with `maxThreads: 1`.
export class ThreadsTimeoutError extends Error {
  readonly code = 'threads-timeout' as const;
  readonly record: BackendRecord;

  constructor(record: BackendRecord) {
    super(
      'wasm multi-thread session creation/warm-up did not finish within the timeout; ' +
        'onnxruntime-web reads numThreads once per worker, so the host must start a ' +
        'fresh worker with maxThreads: 1',
    );
    this.name = 'ThreadsTimeoutError';
    this.record = record;
  }
}

function defaultSetTimer(callback: () => void, ms: number): () => void {
  const handle = setTimeout(callback, ms);
  return () => clearTimeout(handle);
}

// KOM-1214: races `work` (wasm session creation + warm-up) against a timer
// of `timeoutMs`, but only when `threads > 1` -- at threads === 1 no timer
// is ever created, so that path is byte-identical to before this existed.
// On timeout, builds and persists the KOM-1096 failure mark and rejects with
// `ThreadsTimeoutError` carrying it; the timer is always cleared on the
// `work` path, win or lose, so a settled probe never leaves a timer pending.
async function withThreadsTimeout<T>(
  threads: number,
  timeoutMs: number,
  setTimer: (callback: () => void, ms: number) => () => void,
  key: string,
  now: () => number,
  storage: StorageLike | undefined,
  work: () => Promise<T>,
): Promise<T> {
  if (threads <= 1) return work();
  const timer: { cancel: (() => void) | null } = { cancel: null };
  const timeout = new Promise<never>((_, reject) => {
    timer.cancel = setTimer(() => {
      const record: BackendRecord = { backend: 'wasm', threads: 1, msPerPosition: {}, chosenBatch: 1, probedAt: now(), key };
      writeCache(storage, key, record);
      reject(new ThreadsTimeoutError(record));
    }, timeoutMs);
  });
  try {
    return await Promise.race([work(), timeout]);
  } finally {
    timer.cancel?.();
  }
}

// Cache key (design 3.2): per device shape. Keyed on the real adapter's
// identity when one is available (so a GPU swap invalidates the entry), not
// on whether WebGPU ended up chosen.
export function backendCacheKey(opts: {
  modelSha256: string;
  adapter: GpuAdapterLike | null;
  hardwareConcurrency: number;
  isolated: boolean;
}): string {
  const adapterId = opts.adapter
    ? `${opts.adapter.info?.vendor ?? 'unknown'}/${opts.adapter.info?.architecture ?? 'unknown'}`
    : 'none';
  return `komugi.backend.v1:${opts.modelSha256}:${adapterId}:${opts.hardwareConcurrency}:${opts.isolated}`;
}

// requestAdapter retry (design 3.2 step 2): the first call can return null
// under headless Chromium and some drivers; a second call after a microtask
// has been observed to succeed. A fallback/software adapter is never used
// regardless of which attempt returned it -- it is a real, non-null
// response, just not one the ladder is allowed to pick.
async function requestRealAdapter(gpu: GpuLike | undefined): Promise<GpuAdapterLike | null> {
  if (!gpu) return null;
  for (let attempt = 0; attempt < 2; attempt++) {
    const adapter = await gpu.requestAdapter();
    if (adapter) return adapter.isFallbackAdapter ? null : adapter;
  }
  return null;
}

async function warmupAll(session: ProbeSession): Promise<void> {
  for (const bucket of BUCKETS) await session.warmup(bucket);
}

// Median of 3 per-position timings at `batch` (design 3.2 step 3/4: "times 3
// runs... keeps the median ms per position"). KOM-1098: one extra untimed
// run at the timed batch first -- on top of the bucket warm-up -- so a
// first-run shader compile/allocation at that shape never lands in a sample.
async function medianMsPerPosition(
  session: ProbeSession,
  batch: number,
  now: () => number,
): Promise<number> {
  await session.run(batch);
  const samples: number[] = [];
  for (let i = 0; i < 3; i++) {
    const start = now();
    await session.run(batch);
    samples.push((now() - start) / batch);
  }
  samples.sort((a, b) => a - b);
  return samples[1]!;
}

async function createChosenSession(
  factory: SessionFactory,
  backend: BackendKind,
  adapter: GpuAdapterLike | null,
): Promise<ProbeSession> {
  if (backend === 'webgpu') {
    if (!adapter) throw new Error('backend.ts: cached backend is webgpu but no real adapter is available');
    return factory.createWebgpu();
  }
  return factory.createWasm();
}

// wasm's operating batch: on 1 thread batching buys nothing (design
// preamble), so B4 only when multi-thread is actually in play.
function wasmBatch(threads: number): number {
  return threads > 1 ? WASM_MT_BATCH : 1;
}

// A record that carries no timings is a failure mark (worker.ts's
// `writeFallbackCache`, or this module's own wasm multi-thread failure
// below): its `threads` is a ceiling learned from a failure, not a setting.
function isFailureMark(record: BackendRecord): boolean {
  return Object.keys(record.msPerPosition ?? {}).length === 0;
}

// The ladder itself (KOM-1098): WebGPU on a real adapter if its ms per
// position at B8 beats wasm's at its operating batch; otherwise wasm. wasm
// is always timed (it is always available and is the fallback target),
// WebGPU only when a real adapter was obtained. A wasm failure at threads >
// 1 yields a threads=1 failure mark (numThreads is read once per worker, so
// the 1-thread retry is the builder's / next start's job) unless WebGPU
// already measured fine.
async function runProbe(
  factory: SessionFactory,
  adapter: GpuAdapterLike | null,
  threads: number,
  now: () => number,
  key: string,
  threadsTimeoutMs: number,
  setTimer: (callback: () => void, ms: number) => () => void,
  storage: StorageLike | undefined,
): Promise<BackendRecord> {
  let webgpu: BackendTiming | undefined;
  if (adapter) {
    try {
      const session = await factory.createWebgpu();
      await warmupAll(session);
      webgpu = { batch: WEBGPU_BATCH, msPerPosition: await medianMsPerPosition(session, WEBGPU_BATCH, now) };
    } catch {
      webgpu = undefined; // session creation, warm-up or timing failed -> fall through to wasm
    }
  }

  let wasm: BackendTiming | undefined;
  const batch = wasmBatch(threads);
  try {
    const wasmSession = await withThreadsTimeout(threads, threadsTimeoutMs, setTimer, key, now, storage, async () => {
      const session = await factory.createWasm();
      await warmupAll(session);
      return session;
    });
    wasm = { batch, msPerPosition: await medianMsPerPosition(wasmSession, batch, now) };
  } catch (error) {
    // KOM-1214: a threads-timeout is never swallowed into a fallback choice
    // (even one already measured fine for WebGPU) -- the host must restart
    // in a fresh worker, never retry/re-probe in this one.
    if (error instanceof ThreadsTimeoutError) throw error;
    if (webgpu === undefined) {
      if (threads <= 1) throw error;
      return { backend: 'wasm', threads: 1, msPerPosition: {}, chosenBatch: 1, probedAt: now(), key };
    }
  }

  const timings: { webgpu?: BackendTiming; wasm?: BackendTiming } = {};
  if (webgpu !== undefined) timings.webgpu = webgpu;
  if (wasm !== undefined) timings.wasm = wasm;

  if (webgpu !== undefined && (wasm === undefined || webgpu.msPerPosition < wasm.msPerPosition)) {
    return {
      backend: 'webgpu',
      threads,
      msPerPosition: { [WEBGPU_BATCH]: webgpu.msPerPosition },
      chosenBatch: WEBGPU_BATCH,
      probedAt: now(),
      timings,
      key,
    };
  }

  const wasmTiming = wasm as BackendTiming;
  return {
    backend: 'wasm',
    threads,
    msPerPosition: { [batch]: wasmTiming.msPerPosition },
    chosenBatch: batch,
    probedAt: now(),
    timings,
    key,
  };
}

function readCache(storage: StorageLike | undefined, key: string): BackendRecord | null {
  if (!storage) return null;
  try {
    const raw = storage.getItem(key);
    if (!raw) return null;
    return JSON.parse(raw) as BackendRecord;
  } catch {
    return null; // a corrupt or inaccessible entry is treated as a miss, not a crash
  }
}

function writeCache(storage: StorageLike | undefined, key: string, record: BackendRecord): void {
  if (!storage) return;
  try {
    storage.setItem(key, JSON.stringify(record));
  } catch {
    // localStorage can throw (quota, private mode); losing the cache just costs the next load a re-probe.
  }
}

// A relayed record is only usable if it is well-formed and (when it says)
// was probed under this call's device-shape key.
function relayedRecord(record: BackendRecord | null | undefined, key: string): BackendRecord | null {
  if (!record || typeof record !== 'object') return null;
  if (record.backend !== 'webgpu' && record.backend !== 'wasm') return null;
  if (typeof record.probedAt !== 'number' || typeof record.threads !== 'number') return null;
  if (record.key !== undefined && record.key !== key) return null;
  return record;
}

// The device probe and its cache (design 3.2). `threads` is the value
// actually in effect for this call -- on a cache hit it is recomputed
// fresh, not read back from the stored record, since a "threads" setting
// change is meant to take effect at the next worker start (S-2), which this
// call represents. Exception (KOM-1096 known gap, KOM-1098): a failure
// mark's threads is a ceiling, so after a multi-thread failure the next
// start stays at 1 thread until the mark expires.
export async function probeBackend(opts: ProbeOptions): Promise<ProbeResult> {
  const now = opts.now ?? Date.now;
  const adapter = await requestRealAdapter(opts.gpu);
  const key = backendCacheKey({
    modelSha256: opts.modelSha256,
    adapter,
    hardwareConcurrency: opts.hardwareConcurrency,
    isolated: opts.isolated,
  });

  const maxAge = opts.maxAgeMs ?? DEFAULT_MAX_AGE_MS;
  const relayed = opts.cachedRecord !== undefined;
  const stored = relayed ? relayedRecord(opts.cachedRecord, key) : readCache(opts.storage, key);
  const cached = stored && now() - stored.probedAt < maxAge ? stored : null;

  let threads = effectiveThreads(opts);
  if (cached && isFailureMark(cached)) threads = Math.max(1, Math.min(threads, cached.threads));
  // S-2: exactly once, before any session is created.
  opts.factory.setNumThreads(threads);

  const threadsTimeoutMs = opts.threadsTimeoutMs ?? DEFAULT_THREADS_TIMEOUT_MS;
  const setTimer = opts.setTimer ?? defaultSetTimer;

  if (cached) {
    try {
      const createAndWarm = async (): Promise<ProbeSession> => {
        const session = await createChosenSession(opts.factory, cached.backend, adapter);
        await warmupAll(session);
        return session;
      };
      // KOM-1214: only wasm's session creation races the timeout -- a cached
      // webgpu backend never does, same as runProbe only ever times wasm.
      if (cached.backend === 'wasm') {
        await withThreadsTimeout(threads, threadsTimeoutMs, setTimer, key, now, opts.storage, createAndWarm);
      } else {
        await createAndWarm();
      }
      const record: BackendRecord = { ...cached, key };
      // wasm's batch follows the threads in effect now, not the probe's.
      const chosenBatch = cached.backend === 'wasm' ? wasmBatch(threads) : cached.chosenBatch;
      return { ...cached, threads, chosenBatch, fromCache: true, record, cacheKey: key };
    } catch (error) {
      // KOM-1214: a threads-timeout on the cached-record path never falls
      // through to a re-probe either -- same "fresh worker only" rule.
      if (error instanceof ThreadsTimeoutError) throw error;
      // The cached backend no longer works on this device/session -- re-probe fully, below.
    }
  }

  const record = await runProbe(opts.factory, adapter, threads, now, key, threadsTimeoutMs, setTimer, opts.storage);
  writeCache(opts.storage, key, record);
  return { ...record, fromCache: false, record, cacheKey: key };
}

// The real SessionFactory: one onnxruntime-web model, two execution-provider
// sessions (design 3.1/2.2's "one dynamic ONNX model"). `ort` is loaded by
// the caller exactly as ort_browser.ts's loadOrtSessionBrowser already does
// (a caller-supplied browser ESM URL, no vendor filesystem path, no
// bundler); this module only ever receives the already-imported module, so
// it stays loadable as a worker script as-is.
export function makeOrtSessionFactory(
  ort: OrtModule,
  modelUrl: string,
  obsShape: readonly number[],
  // KOM-1226: forwarded to every InferenceSession.create below; absent
  // defaults to 3 (errors only), same as before this field existed.
  options?: { logSeverityLevel?: number },
): SessionFactory {
  const logSeverityLevel = options?.logSeverityLevel ?? 3;
  const rowSize = obsShape.reduce((a, b) => a * b, 1);

  function probeSessionOf(session: OrtSession): ProbeSession {
    return {
      async warmup(bucket: number): Promise<void> {
        const data = new Float32Array(rowSize * bucket);
        const tensor = new ort.Tensor('float32', data, [bucket, ...obsShape]);
        await session.run({ obs: tensor });
      },
      async run(batch: number): Promise<void> {
        const data = new Float32Array(rowSize * batch);
        const tensor = new ort.Tensor('float32', data, [batch, ...obsShape]);
        await session.run({ obs: tensor });
      },
    };
  }

  return {
    setNumThreads(threads: number): void {
      ort.env.wasm.numThreads = threads;
    },
    async createWebgpu(): Promise<ProbeSession> {
      const session = await ort.InferenceSession.create(modelUrl, { executionProviders: ['webgpu'], logSeverityLevel });
      return probeSessionOf(session);
    },
    async createWasm(): Promise<ProbeSession> {
      const session = await ort.InferenceSession.create(modelUrl, { executionProviders: ['wasm'], logSeverityLevel });
      return probeSessionOf(session);
    },
  };
}
