// KOM-1185 unit 4b2 (design docs/design/ts-engine-library.md section 5.2
// "One ORT delivery model", section 7.1): `installWorkerHost`'s real
// `makeInitEnv` -- model fetch, sha256 verification, and `ort_core.ts`'s
// `makeOrtBackend` -- driven with an injected `fetch`/`loadOrt` and a FAKE
// `OrtModule` that logs every call/assignment in order. No real
// onnxruntime-web, no network, no browser.

import assert from 'node:assert/strict';
import { test } from 'node:test';

import type { BackendKind, GpuLike } from '../src/backend.ts';
import { ACTION_SIZE, ADVANCED_POSITION } from '../src/constants.ts';
import type { OrtModule, OrtSession } from '../src/ort_core.ts';
import type {
  WorkerInitReply,
  WorkerInitRequest,
  WorkerPonderNowReply,
  WorkerPonderSettingsReply,
  WorkerReply,
  WorkerRequest,
} from '../src/protocol.ts';
import { installWorkerHost, isOrtModule } from '../src/worker_host.ts';
import type { WorkerHostOptions } from '../src/worker_host.ts';

type Posted = WorkerReply | WorkerInitReply | WorkerPonderSettingsReply | WorkerPonderNowReply;

// Same fake worker global as worker_host.test.ts: `send` plays the role of
// a real `postMessage` arriving at the worker, dispatching synchronously.
function makeFakeScope(): {
  scope: { addEventListener: (type: 'message', listener: (event: { data: unknown }) => void) => void; postMessage: (reply: Posted) => void };
  posted: Posted[];
  send: (data: unknown) => void;
} {
  const listeners: Array<(event: { data: unknown }) => void> = [];
  const posted: Posted[] = [];
  const scope = {
    addEventListener(_type: 'message', listener: (event: { data: unknown }) => void): void {
      listeners.push(listener);
    },
    postMessage(reply: Posted): void {
      posted.push(reply);
    },
  };
  const send = (data: unknown): void => {
    for (const listener of listeners) listener({ data });
  };
  return { scope, posted, send };
}

// Unlike worker_host.test.ts's own fakes (pure promise chains), the real
// `makeInitEnv` here awaits `globalThis.crypto.subtle.digest`, which
// resolves via Node's threadpool -- a macrotask, not a microtask. A pure
// `await Promise.resolve()` poll starves the event loop's macrotask phase
// forever (the digest's completion callback never gets a turn), so this
// yields through a real timer tick each iteration instead.
async function flushUntil(predicate: () => boolean, maxTicks = 2000): Promise<void> {
  for (let i = 0; i < maxTicks; i++) {
    if (predicate()) return;
    await new Promise<void>((resolve) => setTimeout(resolve, 0));
  }
}

async function sha256Hex(bytes: Uint8Array): Promise<string> {
  const digest = await globalThis.crypto.subtle.digest('SHA-256', bytes);
  return Array.from(new Uint8Array(digest))
    .map((byte) => byte.toString(16).padStart(2, '0'))
    .join('');
}

const MODEL_BYTES = new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
const ORT_ASSETS_WASM_ONLY = { wasm: { mjs: 'wasm.mjs', wasm: 'wasm.wasm' } };
const ORT_ASSETS_WITH_WEBGPU = { ...ORT_ASSETS_WASM_ONLY, webgpu: { mjs: 'webgpu.mjs', wasm: 'webgpu.wasm' } };

type OrtEvent =
  | { type: 'set-wasmPaths'; value: unknown }
  | { type: 'set-proxy'; value: unknown }
  | { type: 'set-numThreads'; value: unknown }
  | { type: 'create'; bytes: unknown; executionProviders: string[] }
  | { type: 'run' };

// A FAKE OrtModule: `env.wasm`'s three fields are accessor properties that
// log every assignment, `InferenceSession.create` logs its call and the
// bytes it received, and the returned session's `run` logs each call and
// returns zero-filled float32 outputs at the names/sizes `runOrtBatch`
// (batch_evaluator.ts) reads -- `policy` (ACTION_SIZE per row), `wdl` (3 per
// row), `moves_left` (1 per row), sized off the request tensor's own bucket
// dimension so warm-up/timing calls at any bucket produce a legal shape.
function makeFakeOrtModule(log: OrtEvent[]): OrtModule {
  let numThreads = 1;
  let wasmPaths: string | { mjs?: string; wasm?: string } | undefined;
  let proxy: boolean | undefined;
  const wasm = {
    get numThreads(): number {
      return numThreads;
    },
    set numThreads(value: number) {
      numThreads = value;
      log.push({ type: 'set-numThreads', value });
    },
    get wasmPaths(): string | { mjs?: string; wasm?: string } | undefined {
      return wasmPaths;
    },
    set wasmPaths(value: string | { mjs?: string; wasm?: string } | undefined) {
      wasmPaths = value;
      log.push({ type: 'set-wasmPaths', value });
    },
    get proxy(): boolean | undefined {
      return proxy;
    },
    set proxy(value: boolean | undefined) {
      proxy = value;
      log.push({ type: 'set-proxy', value });
    },
  };

  class FakeTensor {
    readonly kind: string;
    readonly data: Float32Array;
    readonly dims: readonly number[];
    constructor(kind: string, data: Float32Array, dims: readonly number[]) {
      this.kind = kind;
      this.data = data;
      this.dims = dims;
    }
  }

  return {
    env: { wasm },
    InferenceSession: {
      async create(pathOrBytes: string | Uint8Array, opts: { executionProviders: string[] }): Promise<OrtSession> {
        log.push({ type: 'create', bytes: pathOrBytes, executionProviders: opts.executionProviders });
        return {
          async run(feeds: Record<string, unknown>): Promise<Record<string, { data: Float32Array }>> {
            log.push({ type: 'run' });
            const obsTensor = feeds.obs as { dims: readonly number[] };
            const bucket = obsTensor.dims[0];
            return {
              policy: { data: new Float32Array(bucket * ACTION_SIZE) },
              wdl: { data: new Float32Array(bucket * 3) },
              moves_left: { data: new Float32Array(bucket) },
            };
          },
        };
      },
    },
    Tensor: FakeTensor,
  };
}

function fakeFetchOk(bytes: Uint8Array): WorkerHostOptions['fetch'] {
  return async (_url: string) => ({ ok: true, status: 200, arrayBuffer: async () => bytes.buffer as ArrayBuffer });
}

function isOkInitReply(reply: Posted | undefined): reply is WorkerInitReply & { backend: BackendKind; threads: number } {
  return reply !== undefined && !('err' in reply);
}

test('worker_host_ort init: correct sha wires wasmPaths/proxy/numThreads before the first create', async () => {
  const log: OrtEvent[] = [];
  const loadOrtCalls: BackendKind[] = [];
  const loadOrt = async (kind: BackendKind): Promise<OrtModule> => {
    loadOrtCalls.push(kind);
    return makeFakeOrtModule(log);
  };
  const sha = await sha256Hex(MODEL_BYTES);
  const options: WorkerHostOptions = { loadOrt, fetch: fakeFetchOk(MODEL_BYTES) };
  const { scope, posted, send } = makeFakeScope();
  installWorkerHost(scope, options);

  const initRequest: WorkerInitRequest = {
    id: 'init-1',
    type: 'init',
    modelUrl: 'model.onnx',
    modelSha256: sha,
    ortAssets: ORT_ASSETS_WASM_ONLY,
  };
  send(initRequest);
  await flushUntil(() => posted.length >= 1);

  assert.equal(posted.length, 1, `expected exactly one reply, got ${JSON.stringify(posted)}`);
  const [reply] = posted;
  assert.ok(isOkInitReply(reply), `init should succeed, got ${JSON.stringify(reply)}`);
  if (!isOkInitReply(reply)) return;
  assert.equal(reply.backend, 'wasm');
  assert.equal(typeof reply.threads, 'number');
  assert.deepEqual(loadOrtCalls, ['wasm'], 'loadOrt must be called exactly once, for wasm only');

  const createIndex = log.findIndex((event) => event.type === 'create');
  assert.ok(createIndex >= 0, 'a session must have been created');
  const before = log.slice(0, createIndex);
  const wasmPathsEvent = before.find((event): event is { type: 'set-wasmPaths'; value: unknown } => event.type === 'set-wasmPaths');
  const proxyEvent = before.find((event): event is { type: 'set-proxy'; value: unknown } => event.type === 'set-proxy');
  const numThreadsEvent = before.find((event): event is { type: 'set-numThreads'; value: unknown } => event.type === 'set-numThreads');
  assert.ok(wasmPathsEvent, 'wasmPaths must be set before the first create');
  assert.ok(proxyEvent, 'proxy must be set before the first create');
  assert.ok(numThreadsEvent, 'numThreads must be set before the first create');
  assert.deepEqual(wasmPathsEvent?.value, ORT_ASSETS_WASM_ONLY.wasm);
  assert.equal(proxyEvent?.value, false);

  const createEvent = log[createIndex] as { type: 'create'; bytes: unknown; executionProviders: string[] };
  assert.ok(createEvent.bytes instanceof Uint8Array, 'create must receive the model bytes as a Uint8Array');
  assert.deepEqual(Array.from(createEvent.bytes as Uint8Array), Array.from(MODEL_BYTES));
});

test('worker_host_ort init: wrong modelSha256 rejects sha-mismatch without loading or creating anything', async () => {
  const log: OrtEvent[] = [];
  let loadOrtCalls = 0;
  const loadOrt = async (): Promise<OrtModule> => {
    loadOrtCalls++;
    return makeFakeOrtModule(log);
  };
  const options: WorkerHostOptions = { loadOrt, fetch: fakeFetchOk(MODEL_BYTES) };
  const { scope, posted, send } = makeFakeScope();
  installWorkerHost(scope, options);

  send({
    id: 'init-1',
    type: 'init',
    modelUrl: 'model.onnx',
    modelSha256: 'deadbeef00000000000000000000000000000000000000000000000000000',
    ortAssets: ORT_ASSETS_WASM_ONLY,
  });
  await flushUntil(() => posted.length >= 1);

  const [reply] = posted;
  assert.ok(reply !== undefined && 'err' in reply, `expected an error reply, got ${JSON.stringify(reply)}`);
  if (reply === undefined || !('err' in reply)) return;
  assert.equal(reply.code, 'sha-mismatch');
  assert.equal(loadOrtCalls, 0, 'loadOrt must never be called on a sha mismatch');
  assert.equal(log.filter((event) => event.type === 'create').length, 0, 'no session must ever be created on a sha mismatch');
});

test('worker_host_ort init: a rejected fetch replies model-fetch', async () => {
  const options: WorkerHostOptions = {
    loadOrt: async () => makeFakeOrtModule([]),
    fetch: async () => {
      throw new Error('network down');
    },
  };
  const { scope, posted, send } = makeFakeScope();
  installWorkerHost(scope, options);

  send({ id: 'init-1', type: 'init', modelUrl: 'model.onnx', modelSha256: 'sha', ortAssets: ORT_ASSETS_WASM_ONLY });
  await flushUntil(() => posted.length >= 1);

  const [reply] = posted;
  assert.ok(reply !== undefined && 'err' in reply, `expected an error reply, got ${JSON.stringify(reply)}`);
  if (reply === undefined || !('err' in reply)) return;
  assert.equal(reply.code, 'model-fetch');
});

test('worker_host_ort init: a non-ok fetch response replies model-fetch', async () => {
  const options: WorkerHostOptions = {
    loadOrt: async () => makeFakeOrtModule([]),
    fetch: async () => ({ ok: false, status: 404, arrayBuffer: async () => new ArrayBuffer(0) }),
  };
  const { scope, posted, send } = makeFakeScope();
  installWorkerHost(scope, options);

  send({ id: 'init-1', type: 'init', modelUrl: 'model.onnx', modelSha256: 'sha', ortAssets: ORT_ASSETS_WASM_ONLY });
  await flushUntil(() => posted.length >= 1);

  const [reply] = posted;
  assert.ok(reply !== undefined && 'err' in reply, `expected an error reply, got ${JSON.stringify(reply)}`);
  if (reply === undefined || !('err' in reply)) return;
  assert.equal(reply.code, 'model-fetch');
});

test('worker_host_ort: a search after a successful init reaches the fake session, never the fake leaf evaluator', async () => {
  const log: OrtEvent[] = [];
  const loadOrt = async (): Promise<OrtModule> => makeFakeOrtModule(log);
  const sha = await sha256Hex(MODEL_BYTES);
  const options: WorkerHostOptions = { loadOrt, fetch: fakeFetchOk(MODEL_BYTES) };
  const { scope, posted, send } = makeFakeScope();
  installWorkerHost(scope, options);

  send({ id: 'init-1', type: 'init', modelUrl: 'model.onnx', modelSha256: sha, ortAssets: ORT_ASSETS_WASM_ONLY });
  await flushUntil(() => posted.length >= 1);
  assert.ok(isOkInitReply(posted[0]), `init should succeed, got ${JSON.stringify(posted[0])}`);

  const runsAfterInit = log.filter((event) => event.type === 'run').length;

  const searchRequest: WorkerRequest = { id: 'search-1', fen: ADVANCED_POSITION, sims: 4, batch: 1 };
  send(searchRequest);
  await flushUntil(() => posted.length >= 2);

  const runsAfterSearch = log.filter((event) => event.type === 'run').length;
  assert.ok(runsAfterSearch > runsAfterInit, 'the search must have run the fake session at least once');

  const [, searchReply] = posted;
  assert.ok(searchReply !== undefined && !('err' in searchReply), `search should succeed, got ${JSON.stringify(searchReply)}`);
  if (searchReply === undefined || 'err' in searchReply) return;
  assert.equal(typeof searchReply.action, 'number', 'a successful search reply carries a legal action id');
});

test('worker_host_ort: no gpu given never loads webgpu', async () => {
  const log: OrtEvent[] = [];
  const loadOrtCalls: BackendKind[] = [];
  const loadOrt = async (kind: BackendKind): Promise<OrtModule> => {
    loadOrtCalls.push(kind);
    return makeFakeOrtModule(log);
  };
  const sha = await sha256Hex(MODEL_BYTES);
  const options: WorkerHostOptions = { loadOrt, fetch: fakeFetchOk(MODEL_BYTES) };
  const { scope, posted, send } = makeFakeScope();
  installWorkerHost(scope, options);

  send({ id: 'init-1', type: 'init', modelUrl: 'model.onnx', modelSha256: sha, ortAssets: ORT_ASSETS_WITH_WEBGPU });
  await flushUntil(() => posted.length >= 1);

  assert.ok(isOkInitReply(posted[0]), `init should succeed, got ${JSON.stringify(posted[0])}`);
  assert.ok(!loadOrtCalls.includes('webgpu'), 'loadOrt must never be called for webgpu when no gpu was supplied');
});

test('worker_host_ort: ortAssets without webgpu plus a working gpu adapter still ends on wasm', async () => {
  const log: OrtEvent[] = [];
  const loadOrtCalls: BackendKind[] = [];
  const loadOrt = async (kind: BackendKind): Promise<OrtModule> => {
    loadOrtCalls.push(kind);
    return makeFakeOrtModule(log);
  };
  const sha = await sha256Hex(MODEL_BYTES);
  const gpu: GpuLike = {
    requestAdapter: async () => ({ isFallbackAdapter: false, info: { vendor: 'fake', architecture: 'fake' } }),
  };
  const options: WorkerHostOptions = { loadOrt, fetch: fakeFetchOk(MODEL_BYTES), gpu };
  const { scope, posted, send } = makeFakeScope();
  installWorkerHost(scope, options);

  send({ id: 'init-1', type: 'init', modelUrl: 'model.onnx', modelSha256: sha, ortAssets: ORT_ASSETS_WASM_ONLY });
  await flushUntil(() => posted.length >= 1);

  const [reply] = posted;
  assert.ok(isOkInitReply(reply), `init should succeed via the BackendError fallback, got ${JSON.stringify(reply)}`);
  if (!isOkInitReply(reply)) return;
  assert.equal(reply.backend, 'wasm', 'webgpu has no ortAssets entry, so the probe must fall back to wasm');
  assert.ok(!loadOrtCalls.includes('webgpu'), 'the webgpu BackendError must fire before loadOrt is ever called for it');
});

test('worker_host_ort: the default init path never reads globalThis.localStorage', async () => {
  const originalDescriptor = Object.getOwnPropertyDescriptor(globalThis, 'localStorage');
  let accessed = false;
  Object.defineProperty(globalThis, 'localStorage', {
    configurable: true,
    get(): never {
      accessed = true;
      throw new Error('worker_host.ts must never read globalThis.localStorage -- section 5.2/8 bans storage access in the worker');
    },
  });
  try {
    const log: OrtEvent[] = [];
    const loadOrt = async (): Promise<OrtModule> => makeFakeOrtModule(log);
    const sha = await sha256Hex(MODEL_BYTES);
    const options: WorkerHostOptions = { loadOrt, fetch: fakeFetchOk(MODEL_BYTES) };
    const { scope, posted, send } = makeFakeScope();
    installWorkerHost(scope, options);

    send({ id: 'init-1', type: 'init', modelUrl: 'model.onnx', modelSha256: sha, ortAssets: ORT_ASSETS_WASM_ONLY });
    await flushUntil(() => posted.length >= 1);

    assert.ok(isOkInitReply(posted[0]), `init should succeed, got ${JSON.stringify(posted[0])}`);
    assert.equal(accessed, false, 'installWorkerHost must never touch globalThis.localStorage');
  } finally {
    if (originalDescriptor) Object.defineProperty(globalThis, 'localStorage', originalDescriptor);
    else delete (globalThis as { localStorage?: unknown }).localStorage;
  }
});

// KOM-1225: real onnxruntime-web 1.31.1 exports InferenceSession as a
// CLASS (typeof 'function'), not an object literal -- the fake modules
// above are object-shaped, so they never exercised this path.
class ClassInferenceSession {
  static async create(): Promise<OrtSession> {
    return { run: async () => ({}) } as unknown as OrtSession;
  }
}
class ClassTensor {}

test('isOrtModule: a class-shaped InferenceSession (real onnxruntime-web shape) is accepted', () => {
  const classShaped = { env: { wasm: {} }, InferenceSession: ClassInferenceSession, Tensor: ClassTensor };
  assert.equal(isOrtModule(classShaped), true, 'a class-shaped InferenceSession with a callable create must be accepted');
});

test('isOrtModule: the same class-shaped module behind a bare default export is accepted via the fallback', () => {
  const classShaped = { env: { wasm: {} }, InferenceSession: ClassInferenceSession, Tensor: ClassTensor };
  const wrapped = { default: classShaped };
  assert.equal(isOrtModule(wrapped), false, 'the namespace itself (lacking InferenceSession/env/Tensor) must not satisfy isOrtModule directly');
  assert.equal(isOrtModule((wrapped as { default: unknown }).default), true, 'the default export must satisfy isOrtModule -- the same fallback defaultLoadOrt takes');
});

test('isOrtModule: InferenceSession without a callable create is rejected', () => {
  const noCreate = { env: { wasm: {} }, InferenceSession: class NoCreate {}, Tensor: ClassTensor };
  assert.equal(isOrtModule(noCreate), false);
});

test('isOrtModule: env without wasm is rejected', () => {
  const noWasm = { env: {}, InferenceSession: ClassInferenceSession, Tensor: ClassTensor };
  assert.equal(isOrtModule(noWasm), false);
});

test('isOrtModule: Tensor missing is rejected', () => {
  const noTensor = { env: { wasm: {} }, InferenceSession: ClassInferenceSession };
  assert.equal(isOrtModule(noTensor), false);
});

test('worker_host_ort init: missing ortAssets replies init-failed', async () => {
  const options: WorkerHostOptions = { loadOrt: async () => makeFakeOrtModule([]), fetch: fakeFetchOk(MODEL_BYTES) };
  const { scope, posted, send } = makeFakeScope();
  installWorkerHost(scope, options);

  send({ id: 'init-1', type: 'init', modelUrl: 'model.onnx', modelSha256: 'sha' });
  await flushUntil(() => posted.length >= 1);

  const [reply] = posted;
  assert.ok(reply !== undefined && 'err' in reply, `expected an error reply, got ${JSON.stringify(reply)}`);
  if (reply === undefined || !('err' in reply)) return;
  assert.equal(reply.code, 'init-failed');
});
