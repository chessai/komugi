// KOM-1200 units 1-2 (design docs/design/kom1200-model-loading.md section
// 8/9): loadModel's fetch/verify/progress/compatibility behaviour (unit 1),
// plus Cache Storage read/write/LRU/quota-retry (unit 2). No real network,
// no real Cache Storage -- a StreamFetchLike mock built from Node's global
// ReadableStream/Headers-shaped object, a CacheStorageLike mock with
// real-Cache body semantics, and cards built inline. Unit 3 adds the
// end-to-end tests: loadModel's blob: URL through the real worker host
// (fake ORT, REAL global fetch) and KomugiClient, plus a README check.

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';

import { KomugiClient } from '../src/client.ts';
import type { KomugiModel, KomugiWorkerLike } from '../src/client.ts';
import { ACTION_ENCODING_VERSION, ACTION_SIZE, ADVANCED_POSITION } from '../src/constants.ts';
import { loadModel } from '../src/model_loader.ts';
import type { StreamFetchLike, CacheLike, CacheStorageLike } from '../src/model_loader.ts';
import type { OrtModule, OrtSession } from '../src/ort_core.ts';
import type { WorkerInitReply, WorkerInitRequest, WorkerPonderNowReply, WorkerPonderSettingsReply, WorkerReply } from '../src/protocol.ts';
import { installWorkerHost } from '../src/worker_host.ts';

type FetchResponse = Awaited<ReturnType<StreamFetchLike>>;
type AssetInput = { path: string; sha256: string; bytes?: number };

// ---------------------------------------------------------------------------
// Helpers.

// "never rejects" (section 7) checked at every call site: a reject fails
// the test instead of surfacing as a rejected loadModel() promise.
async function expectResolves<T>(p: Promise<T>): Promise<T> {
  try {
    return await p;
  } catch (error) {
    assert.fail(`loadModel rejected: ${error instanceof Error ? (error.stack ?? error.message) : String(error)}`);
  }
}

async function sha256Hex(bytes: Uint8Array): Promise<string> {
  const digest = await globalThis.crypto.subtle.digest('SHA-256', bytes);
  return Array.from(new Uint8Array(digest))
    .map((byte) => byte.toString(16).padStart(2, '0'))
    .join('');
}

function cardId(sha: string): string {
  return `komugi-c1-s1-${sha.slice(0, 8)}`;
}

function makeCard(id: string, onnx: AssetInput, overrides: { planeMapVersion?: unknown; actionEncodingVersion?: unknown } = {}): Record<string, unknown> {
  const onnxAsset: Record<string, unknown> = { path: onnx.path, sha256: onnx.sha256 };
  if (onnx.bytes !== undefined) onnxAsset.bytes = onnx.bytes;
  return {
    id,
    campaign: 'c1',
    iteration: 1,
    step: 1,
    compatibilityKey: {
      planeMapVersion: overrides.planeMapVersion ?? 4,
      actionEncodingVersion: overrides.actionEncodingVersion ?? ACTION_ENCODING_VERSION,
      rulesProfile: 'komugi',
      architectureTag: 'b10c128',
    },
    modesTrained: ['advanced'],
    assets: {
      pt: { path: 'model.pt', sha256: 'a'.repeat(64) },
      onnx: onnxAsset,
    },
    sourceCheckpoint: { name: 'checkpoint.pt', sha256: 'b'.repeat(64) },
    exportOpset: 17,
    exportParity: { n: 1, maxAbsDelta: 0, atol: 0, rtol: 0 },
    evalPointers: [],
    license: 'MIT, Copyright (c) 2026 chessai',
    createdAt: '2026-11-22T00:00:00Z',
  };
}

function makeIndex(id: string, onnxSha256: string): Record<string, unknown> {
  return { schemaVersion: 1, recommended: {}, models: { [id]: { card: `${id}.card.json`, onnxSha256 } } };
}

function headersFrom(h: Record<string, string> = {}): { get(name: string): string | null } {
  const map = new Map(Object.entries(h).map(([k, v]) => [k.toLowerCase(), v] as const));
  return { get: (name: string) => map.get(name.toLowerCase()) ?? null };
}

function bufferResponse(bytes: Uint8Array, headers: Record<string, string> = {}, status = 200): FetchResponse {
  return {
    ok: status >= 200 && status < 300,
    status,
    headers: headersFrom(headers),
    body: null,
    arrayBuffer: async () => bytes.slice().buffer as ArrayBuffer,
    text: async () => new TextDecoder().decode(bytes),
  };
}

function streamResponse(parts: (Uint8Array | 'error')[], headers: Record<string, string> = {}, status = 200): FetchResponse {
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      void (async () => {
        for (const part of parts) {
          if (part === 'error') {
            controller.error(new Error('stream error'));
            return;
          }
          controller.enqueue(part);
        }
        controller.close();
      })();
    },
  });
  return {
    ok: status >= 200 && status < 300,
    status,
    headers: headersFrom(headers),
    body,
    arrayBuffer: async () => {
      throw new Error('arrayBuffer should not be called when body is streamable');
    },
    text: async () => {
      throw new Error('text should not be called for a streamed asset response');
    },
  };
}

function textResponse(text: string, status = 200): FetchResponse {
  return {
    ok: status >= 200 && status < 300,
    status,
    headers: headersFrom(),
    body: null,
    arrayBuffer: async () => new ArrayBuffer(0),
    text: async () => text,
  };
}

function fetchSpy(handler?: (url: string) => FetchResponse | Promise<FetchResponse>): { fetch: StreamFetchLike; calls: string[] } {
  const calls: string[] = [];
  const fetch: StreamFetchLike = async (url: string) => {
    calls.push(url);
    if (handler === undefined) throw new Error(`unexpected fetch: ${url}`);
    return handler(url);
  };
  return { fetch, calls };
}

function captureProgress(): { onProgress: (loaded: number, total: number | null) => void; events: Array<[number, number | null]> } {
  const events: Array<[number, number | null]> = [];
  return { onProgress: (loaded, total) => events.push([loaded, total]), events };
}

const UNUSED_RESOLVE_URL = (): string => 'https://x.invalid/m.onnx';

const MODEL_PREFIX_TEST = 'https://komugi.invalid/model/';

// ---------------------------------------------------------------------------
// CacheStorageLike mock (section 8): stores BYTES, not Responses; match()
// returns a fresh Response per call; put() consumes its body and rejects a
// re-used Response, exactly as a real Cache; keys() is insertion order.

interface CacheMock extends CacheLike {
  putCalls: string[];
  modelPutCalls: string[];
  deleteCalls: string[];
  seedRaw(key: string, bytes: Uint8Array): void;
  readIndex(): Promise<string[]>;
}

function makeCacheMock(opts: { erroringMatchBody?: boolean; quotaAlwaysOnModelPut?: boolean; quotaFailOnPutNumber?: number } = {}): { caches: CacheStorageLike; cache: CacheMock } {
  const store = new Map<string, Uint8Array>();
  const insertionOrder: string[] = [];
  const putCalls: string[] = [];
  const modelPutCalls: string[] = [];
  const deleteCalls: string[] = [];
  let modelPutAttempt = 0;

  const cache: CacheMock = {
    async match(key: string) {
      if (opts.erroringMatchBody) {
        return new Response(
          new ReadableStream<Uint8Array>({
            start(controller) {
              controller.error(new Error('body read error'));
            },
          }),
        );
      }
      const bytes = store.get(key);
      if (bytes === undefined) return undefined;
      return new Response(bytes.slice());
    },
    async put(key: string, response: Response) {
      if (response.bodyUsed) throw new TypeError('response body already used');
      const buf = await response.arrayBuffer();
      putCalls.push(key);
      const isModelKey = key.startsWith(MODEL_PREFIX_TEST);
      if (isModelKey) {
        modelPutCalls.push(key);
        modelPutAttempt++;
        if (opts.quotaAlwaysOnModelPut || opts.quotaFailOnPutNumber === modelPutAttempt) {
          throw new DOMException('quota exceeded', 'QuotaExceededError');
        }
      }
      if (!store.has(key)) insertionOrder.push(key);
      store.set(key, new Uint8Array(buf));
    },
    async delete(key: string) {
      deleteCalls.push(key);
      const had = store.has(key);
      store.delete(key);
      const idx = insertionOrder.indexOf(key);
      if (idx >= 0) insertionOrder.splice(idx, 1);
      return had;
    },
    async keys() {
      return insertionOrder.map((url) => ({ url }));
    },
    putCalls,
    modelPutCalls,
    deleteCalls,
    seedRaw(key: string, bytes: Uint8Array) {
      if (!store.has(key)) insertionOrder.push(key);
      store.set(key, bytes);
    },
    async readIndex() {
      const bytes = store.get('https://komugi.invalid/lru');
      if (bytes === undefined) return [];
      return JSON.parse(new TextDecoder().decode(bytes)) as string[];
    },
  };

  const caches: CacheStorageLike = {
    async open() {
      return cache;
    },
  };
  return { caches, cache };
}

async function modelFixture(bytes: Uint8Array): Promise<{ card: Record<string, unknown>; resolveUrl: (asset: { path: string }) => string; url: string; sha: string }> {
  const sha = await sha256Hex(bytes);
  const url = `https://x.invalid/${sha}.onnx`;
  const card = makeCard(cardId(sha), { path: 'm.onnx', sha256: sha, bytes: bytes.byteLength });
  return { card, resolveUrl: () => url, url, sha };
}

// ---------------------------------------------------------------------------
// AC2: sha mismatch.

test('AC2 sha mismatch fails and never caches', async () => {
  const good = new Uint8Array([1, 2, 3, 4]);
  const bad = new Uint8Array([9, 9, 9, 9]);
  const sha = await sha256Hex(good);
  const card = makeCard(cardId(sha), { path: 'm.onnx', sha256: sha, bytes: bad.byteLength });
  const { fetch, calls } = fetchSpy(() => bufferResponse(bad));
  const result = await expectResolves(loadModel({ card, resolveUrl: UNUSED_RESOLVE_URL }, { fetch }));
  assert.equal(result.status, 'failed');
  assert.ok(result.status === 'failed' && result.failure.code === 'sha-mismatch');
  // unit 1 wires no Cache Storage at all, so there is no put to make.
  assert.equal(calls.length, 1);
});

test('AC2 sha mismatch never caches (cache on)', async () => {
  const good = new Uint8Array([1, 2, 3, 4]);
  const bad = new Uint8Array([9, 9, 9, 9]);
  const sha = await sha256Hex(good);
  const card = makeCard(cardId(sha), { path: 'm.onnx', sha256: sha, bytes: bad.byteLength });
  const { fetch, calls } = fetchSpy(() => bufferResponse(bad));
  const { caches, cache } = makeCacheMock();
  const result = await expectResolves(loadModel({ card, resolveUrl: UNUSED_RESOLVE_URL }, { fetch, caches }));
  assert.equal(result.status, 'failed');
  assert.ok(result.status === 'failed' && result.failure.code === 'sha-mismatch');
  assert.equal(calls.length, 1);
  assert.equal(cache.modelPutCalls.length, 0);
});

// ---------------------------------------------------------------------------
// AC3: compatibility, checked before any asset fetch.

test('AC3 actionEncodingVersion mismatch fetches nothing', async () => {
  const chunk = new Uint8Array([1, 2, 3]);
  const sha = await sha256Hex(chunk);
  const card = makeCard(cardId(sha), { path: 'm.onnx', sha256: sha, bytes: chunk.byteLength }, { actionEncodingVersion: ACTION_ENCODING_VERSION + 1 });
  const { fetch, calls } = fetchSpy();
  const result = await expectResolves(loadModel({ card, resolveUrl: UNUSED_RESOLVE_URL }, { fetch }));
  assert.equal(result.status, 'failed');
  assert.ok(result.status === 'failed' && result.failure.code === 'incompatible-model');
  assert.equal(calls.length, 0);
});

test('AC3 planeMapVersion 5 fails isModelCard and fetches nothing', async () => {
  const chunk = new Uint8Array([1, 2, 3]);
  const sha = await sha256Hex(chunk);
  const card = makeCard(cardId(sha), { path: 'm.onnx', sha256: sha, bytes: chunk.byteLength }, { planeMapVersion: 5 });
  const { fetch, calls } = fetchSpy();
  const result = await expectResolves(loadModel({ card, resolveUrl: UNUSED_RESOLVE_URL }, { fetch }));
  assert.equal(result.status, 'failed');
  assert.ok(result.status === 'failed' && result.failure.code === 'incompatible-model');
  assert.equal(calls.length, 0);
});

// ---------------------------------------------------------------------------
// id form: dispatch on `id` before `card`.

test('id form unknown id is model-not-found and fetches nothing', async () => {
  const chunk = new Uint8Array([1, 2, 3]);
  const sha = await sha256Hex(chunk);
  const id = cardId(sha);
  const card = makeCard(id, { path: 'm.onnx', sha256: sha, bytes: chunk.byteLength });
  const index = makeIndex(id, sha);
  const { fetch, calls } = fetchSpy();
  const result = await expectResolves(loadModel({ id: 'komugi-other-s1-00000000', index, card, resolveUrl: UNUSED_RESOLVE_URL }, { fetch }));
  assert.equal(result.status, 'failed');
  assert.ok(result.status === 'failed' && result.failure.code === 'model-not-found');
  assert.equal(calls.length, 0);
});

test('id form stale index sha is incompatible-model', async () => {
  const chunk = new Uint8Array([1, 2, 3]);
  const sha = await sha256Hex(chunk);
  const id = cardId(sha);
  const card = makeCard(id, { path: 'm.onnx', sha256: sha, bytes: chunk.byteLength });
  const staleIndex = makeIndex(id, 'f'.repeat(64));
  const { fetch, calls } = fetchSpy();
  const result = await expectResolves(loadModel({ id, index: staleIndex, card, resolveUrl: UNUSED_RESOLVE_URL }, { fetch }));
  assert.equal(result.status, 'failed');
  assert.ok(result.status === 'failed' && result.failure.code === 'incompatible-model');
  assert.equal(calls.length, 0);
});

// ---------------------------------------------------------------------------
// manifestUrl form.

test('manifestUrl relative resolves against baseUrl', async () => {
  const wanted = new Uint8Array([1, 2, 3]);
  const sha = await sha256Hex(wanted);
  const card = makeCard(cardId(sha), { path: 'model.onnx', sha256: sha, bytes: wanted.byteLength });
  const manifestUrl = '/models/card.json';
  const baseUrl = 'https://h.test/play';
  const cardUrl = new URL(manifestUrl, baseUrl).toString();
  const assetUrl = new URL('model.onnx', cardUrl).toString();
  const { fetch, calls } = fetchSpy((url) => {
    if (url === cardUrl) return textResponse(JSON.stringify(card));
    if (url === assetUrl) return bufferResponse(wanted);
    throw new Error(`unexpected fetch ${url}`);
  });
  const result = await expectResolves(loadModel({ manifestUrl }, { fetch, baseUrl }));
  assert.equal(result.status, 'ok');
  assert.deepEqual(calls, [cardUrl, assetUrl]);
});

test('manifestUrl relative resolves against location', async () => {
  const original = Object.getOwnPropertyDescriptor(globalThis, 'location');
  Object.defineProperty(globalThis, 'location', { value: { href: 'https://loc.test/play' }, configurable: true });
  try {
    const wanted = new Uint8Array([1, 2, 3]);
    const sha = await sha256Hex(wanted);
    const card = makeCard(cardId(sha), { path: 'model.onnx', sha256: sha, bytes: wanted.byteLength });
    const manifestUrl = '/models/card.json';
    const cardUrl = new URL(manifestUrl, 'https://loc.test/play').toString();
    const assetUrl = new URL('model.onnx', cardUrl).toString();
    const { fetch, calls } = fetchSpy((url) => {
      if (url === cardUrl) return textResponse(JSON.stringify(card));
      if (url === assetUrl) return bufferResponse(wanted);
      throw new Error(`unexpected fetch ${url}`);
    });
    const result = await expectResolves(loadModel({ manifestUrl }, { fetch }));
    assert.equal(result.status, 'ok');
    assert.deepEqual(calls, [cardUrl, assetUrl]);
  } finally {
    if (original) Object.defineProperty(globalThis, 'location', original);
    else delete (globalThis as { location?: unknown }).location;
  }
});

test('manifestUrl relative without a base fails typed', async () => {
  const original = Object.getOwnPropertyDescriptor(globalThis, 'location');
  delete (globalThis as { location?: unknown }).location;
  try {
    const { fetch, calls } = fetchSpy();
    const result = await expectResolves(loadModel({ manifestUrl: '/models/card.json' }, { fetch }));
    assert.equal(result.status, 'failed');
    assert.ok(result.status === 'failed' && result.failure.code === 'model-fetch');
    assert.equal(calls.length, 0);
  } finally {
    if (original) Object.defineProperty(globalThis, 'location', original);
  }
});

test('manifestUrl non-JSON manifest is incompatible-model', async () => {
  const baseUrl = 'https://h.test/play';
  const manifestUrl = '/models/card.json';
  const cardUrl = new URL(manifestUrl, baseUrl).toString();
  const { fetch, calls } = fetchSpy((url) => {
    if (url === cardUrl) return textResponse('not json');
    throw new Error(`unexpected fetch ${url}`);
  });
  const result = await expectResolves(loadModel({ manifestUrl }, { fetch, baseUrl }));
  assert.equal(result.status, 'failed');
  assert.ok(result.status === 'failed' && result.failure.code === 'incompatible-model');
  assert.deepEqual(calls, [cardUrl]);
});

// ---------------------------------------------------------------------------
// Never rejects.

test('stream error mid-body is model-fetch', async () => {
  const chunk = new Uint8Array([1, 2, 3]);
  const sha = await sha256Hex(chunk);
  const card = makeCard(cardId(sha), { path: 'm.onnx', sha256: sha, bytes: chunk.byteLength * 2 });
  const { fetch } = fetchSpy(() => streamResponse([chunk, 'error']));
  const result = await expectResolves(loadModel({ card, resolveUrl: UNUSED_RESOLVE_URL }, { fetch }));
  assert.equal(result.status, 'failed');
  assert.ok(result.status === 'failed' && result.failure.code === 'model-fetch');
});

test('throwing onProgress still loads', async () => {
  const originalQueueMicrotask = globalThis.queueMicrotask;
  const seen: unknown[] = [];
  globalThis.queueMicrotask = ((fn: () => void) => {
    void Promise.resolve().then(() => {
      try {
        fn();
      } catch (error) {
        seen.push(error);
      }
    });
  }) as typeof globalThis.queueMicrotask;
  try {
    const wanted = new Uint8Array([1, 2, 3, 4]);
    const sha = await sha256Hex(wanted);
    const card = makeCard(cardId(sha), { path: 'm.onnx', sha256: sha, bytes: wanted.byteLength });
    const { fetch } = fetchSpy(() => bufferResponse(wanted));
    let eventCount = 0;
    const onProgress = (): void => {
      eventCount++;
      throw new Error('progress callback boom');
    };
    const result = await expectResolves(loadModel({ card, resolveUrl: UNUSED_RESOLVE_URL }, { fetch, onProgress }));
    assert.equal(result.status, 'ok');
    // Let the deferred microtask-stub callbacks run before asserting.
    await new Promise((resolve) => setTimeout(resolve, 0));
    assert.equal(seen.length, eventCount);
    assert.ok(eventCount > 0);
  } finally {
    globalThis.queueMicrotask = originalQueueMicrotask;
  }
});

test('missing crypto.subtle is a typed failure', async () => {
  const originalCryptoDescriptor = Object.getOwnPropertyDescriptor(globalThis, 'crypto');
  Object.defineProperty(globalThis, 'crypto', { value: {}, configurable: true });
  const originalCreateObjectUrl = URL.createObjectURL;
  let createObjectUrlCalled = false;
  URL.createObjectURL = ((blob: Blob) => {
    createObjectUrlCalled = true;
    return originalCreateObjectUrl(blob);
  }) as typeof URL.createObjectURL;
  try {
    const wanted = new Uint8Array([1, 2, 3]);
    const card = makeCard('komugi-c1-s1-deadbeef', { path: 'm.onnx', sha256: 'd'.repeat(64), bytes: wanted.byteLength });
    const { fetch } = fetchSpy(() => bufferResponse(wanted));
    const result = await expectResolves(loadModel({ card, resolveUrl: UNUSED_RESOLVE_URL }, { fetch }));
    assert.equal(result.status, 'failed');
    assert.ok(result.status === 'failed' && result.failure.code === 'init-failed');
    assert.equal(createObjectUrlCalled, false);
  } finally {
    URL.createObjectURL = originalCreateObjectUrl;
    if (originalCryptoDescriptor) Object.defineProperty(globalThis, 'crypto', originalCryptoDescriptor);
  }
});

// ---------------------------------------------------------------------------
// AC4: progress.

test('AC4 progress strictly increasing ends at card bytes', async () => {
  const c1 = new Uint8Array([1, 2, 3]);
  const c2 = new Uint8Array([4, 5, 6, 7]);
  const c3 = new Uint8Array([8, 9]);
  const empty = new Uint8Array(0);
  const all = new Uint8Array([...c1, ...c2, ...c3]);
  const sha = await sha256Hex(all);
  const card = makeCard(cardId(sha), { path: 'm.onnx', sha256: sha, bytes: all.byteLength });
  const { fetch } = fetchSpy(() => streamResponse([c1, empty, c2, c3]));
  const { onProgress, events } = captureProgress();
  const result = await expectResolves(loadModel({ card, resolveUrl: UNUSED_RESOLVE_URL }, { fetch, onProgress }));
  assert.equal(result.status, 'ok');
  assert.deepEqual(events[0], [0, all.byteLength]);
  assert.deepEqual(events[events.length - 1], [all.byteLength, all.byteLength]);
  for (let i = 1; i < events.length; i++) {
    const prev = events[i - 1] as [number, number | null];
    const cur = events[i] as [number, number | null];
    assert.ok(cur[0] > prev[0], `events not strictly increasing at ${i}`);
  }
});

test('AC4 card without bytes uses Content-Length', async () => {
  const c1 = new Uint8Array([1, 2, 3]);
  const c2 = new Uint8Array([4, 5]);
  const all = new Uint8Array([...c1, ...c2]);
  const sha = await sha256Hex(all);
  const card = makeCard(cardId(sha), { path: 'm.onnx', sha256: sha });
  const { fetch } = fetchSpy(() => streamResponse([c1, c2], { 'content-length': String(all.byteLength) }));
  const { onProgress, events } = captureProgress();
  const result = await expectResolves(loadModel({ card, resolveUrl: UNUSED_RESOLVE_URL }, { fetch, onProgress }));
  assert.equal(result.status, 'ok');
  assert.deepEqual(events[0], [0, all.byteLength]);
  assert.deepEqual(events[events.length - 1], [all.byteLength, all.byteLength]);
});

test('AC4 no bytes and no Content-Length is indeterminate', async () => {
  const c1 = new Uint8Array([1, 2, 3]);
  const c2 = new Uint8Array([4, 5]);
  const all = new Uint8Array([...c1, ...c2]);
  const sha = await sha256Hex(all);
  const card = makeCard(cardId(sha), { path: 'm.onnx', sha256: sha });
  const response = streamResponse([c1, c2]);
  assert.equal(response.headers.get('content-length'), null);
  const { fetch } = fetchSpy(() => response);
  const { onProgress, events } = captureProgress();
  const result = await expectResolves(loadModel({ card, resolveUrl: UNUSED_RESOLVE_URL }, { fetch, onProgress }));
  assert.equal(result.status, 'ok');
  assert.ok(events.every(([, total]) => total === null));
  for (let i = 1; i < events.length; i++) {
    const prev = events[i - 1] as [number, number | null];
    const cur = events[i] as [number, number | null];
    assert.ok(cur[0] > prev[0]);
  }
});

test('AC4 oversize body stops at total and fails sha', async () => {
  const wanted = new Uint8Array([1, 2, 3]);
  const sha = await sha256Hex(wanted);
  const card = makeCard(cardId(sha), { path: 'm.onnx', sha256: sha, bytes: wanted.byteLength });
  const oversizeChunk = new Uint8Array([9, 9, 9, 9, 9]);
  const { fetch } = fetchSpy(() => streamResponse([oversizeChunk]));
  const { onProgress, events } = captureProgress();
  const result = await expectResolves(loadModel({ card, resolveUrl: UNUSED_RESOLVE_URL }, { fetch, onProgress }));
  assert.equal(result.status, 'failed');
  assert.ok(result.status === 'failed' && result.failure.code === 'sha-mismatch');
  for (const [loaded] of events) assert.ok(loaded <= wanted.byteLength);
});

test('AC4 overrun aligned at total repeats no event', async () => {
  const wanted = new Uint8Array([1, 2, 3, 4]);
  const sha = await sha256Hex(wanted);
  const card = makeCard(cardId(sha), { path: 'm.onnx', sha256: sha, bytes: wanted.byteLength });
  const extra = new Uint8Array([9, 9, 9]);
  const { fetch } = fetchSpy(() => streamResponse([wanted, extra]));
  const { onProgress, events } = captureProgress();
  const result = await expectResolves(loadModel({ card, resolveUrl: UNUSED_RESOLVE_URL }, { fetch, onProgress }));
  assert.equal(result.status, 'ok');
  assert.deepEqual(events, [
    [0, 4],
    [4, 4],
  ]);
});

test('AC4 Content-Length never truncates', async () => {
  const wanted = new Uint8Array([1, 2, 3, 4, 5]);
  const sha = await sha256Hex(wanted);
  const card = makeCard(cardId(sha), { path: 'm.onnx', sha256: sha });
  const { fetch } = fetchSpy(() => streamResponse([wanted], { 'content-length': '3' }));
  const { onProgress, events } = captureProgress();
  const result = await expectResolves(loadModel({ card, resolveUrl: UNUSED_RESOLVE_URL }, { fetch, onProgress }));
  assert.equal(result.status, 'ok');
  for (let i = 1; i < events.length; i++) {
    const prev = events[i - 1] as [number, number | null];
    const cur = events[i] as [number, number | null];
    assert.ok(cur[0] > prev[0]);
  }
  assert.deepEqual(events[events.length - 1], [5, null]);
});

test('AC4 body null reports two events', async () => {
  const wanted = new Uint8Array([1, 2, 3, 4]);
  const sha = await sha256Hex(wanted);
  const card = makeCard(cardId(sha), { path: 'm.onnx', sha256: sha, bytes: wanted.byteLength });
  const { fetch } = fetchSpy(() => bufferResponse(wanted));
  const { onProgress, events } = captureProgress();
  const result = await expectResolves(loadModel({ card, resolveUrl: UNUSED_RESOLVE_URL }, { fetch, onProgress }));
  assert.equal(result.status, 'ok');
  assert.deepEqual(events, [
    [0, 4],
    [4, 4],
  ]);
});

// ---------------------------------------------------------------------------
// compatibility survives the URL swap.

test('loadModel keeps compatibility through the URL swap', async () => {
  const wanted = new Uint8Array([1, 2, 3]);
  const sha = await sha256Hex(wanted);
  const card = makeCard(cardId(sha), { path: 'm.onnx', sha256: sha, bytes: wanted.byteLength }, { planeMapVersion: 4 });
  const { fetch } = fetchSpy(() => bufferResponse(wanted));
  const result = await expectResolves(loadModel({ card, resolveUrl: UNUSED_RESOLVE_URL }, { fetch }));
  assert.equal(result.status, 'ok');
  assert.ok(result.status === 'ok' && result.model.compatibility?.planeMapVersion === 4);
  assert.ok(result.status === 'ok' && result.model.sha256 === sha);
});

// ---------------------------------------------------------------------------
// Cache Storage (unit 2, section 5).

test('CacheLike accepts a DOM-shaped Cache', () => {
  const domCache: {
    match(request: Request | string): Promise<Response | undefined>;
    put(request: Request | string, response: Response): Promise<void>;
    delete(request: Request | string): Promise<boolean>;
    keys(): Promise<readonly Request[]>;
  } = {
    match: async () => undefined,
    put: async () => undefined,
    delete: async () => false,
    keys: async () => [] as readonly Request[],
  };
  const asCacheLike: CacheLike = domCache;

  const domCaches: { open(name: string): Promise<typeof domCache> } = {
    open: async () => domCache,
  };
  const asCacheStorageLike: CacheStorageLike = domCaches;

  assert.ok(asCacheLike !== undefined);
  assert.ok(asCacheStorageLike !== undefined);
});

test('cache mock matches real Response semantics', async () => {
  const { cache } = makeCacheMock();
  const bytes = new Uint8Array([1, 2, 3]);
  const key = `${MODEL_PREFIX_TEST}deadbeef`;
  const res = new Response(bytes.slice());

  await cache.put(key, res);
  assert.equal(res.bodyUsed, true);
  await assert.rejects(() => cache.put(key, res));

  const match1 = await cache.match(key);
  assert.ok(match1 !== undefined);
  assert.deepEqual(new Uint8Array(await match1.arrayBuffer()), bytes);
  const match2 = await cache.match(key);
  assert.ok(match2 !== undefined);
  assert.deepEqual(new Uint8Array(await match2.arrayBuffer()), bytes);

  // Sanity: the mock isn't hiding the single-read rule a plain Response has.
  const plain = new Response(bytes.slice());
  await plain.arrayBuffer();
  await assert.rejects(() => plain.arrayBuffer());
});

test('AC1 first load fetches once and caches', async () => {
  const bytes = new Uint8Array([1, 2, 3]);
  const m = await modelFixture(bytes);
  const { caches, cache } = makeCacheMock();
  const { fetch, calls } = fetchSpy(() => bufferResponse(bytes));
  const result = await expectResolves(loadModel({ card: m.card, resolveUrl: m.resolveUrl }, { fetch, caches }));
  assert.equal(result.status, 'ok');
  assert.equal(calls.length, 1);
  assert.equal(cache.modelPutCalls.length, 1);
});

test('AC1 repeat cache hits make zero asset fetches', async () => {
  const bytes = new Uint8Array([1, 2, 3]);
  const m = await modelFixture(bytes);
  const { caches, cache } = makeCacheMock();
  const first = fetchSpy(() => bufferResponse(bytes));
  const seeded = await expectResolves(loadModel({ card: m.card, resolveUrl: m.resolveUrl }, { fetch: first.fetch, caches }));
  assert.equal(seeded.status, 'ok');
  const putsAfterFirst = cache.modelPutCalls.length;

  for (let i = 0; i < 3; i++) {
    const { fetch, calls } = fetchSpy();
    const result = await expectResolves(loadModel({ card: m.card, resolveUrl: m.resolveUrl }, { fetch, caches }));
    assert.equal(result.status, 'ok');
    assert.equal(calls.length, 0);
    assert.equal(cache.modelPutCalls.length, putsAfterFirst);
  }
});

test('AC1 corrupted cache entry is evicted and refetched', async () => {
  const bytes = new Uint8Array([1, 2, 3]);
  const m = await modelFixture(bytes);
  const { caches, cache } = makeCacheMock();
  cache.seedRaw(`${MODEL_PREFIX_TEST}${m.sha}`, new Uint8Array([9, 9, 9]));
  const { fetch, calls } = fetchSpy(() => bufferResponse(bytes));
  const result = await expectResolves(loadModel({ card: m.card, resolveUrl: m.resolveUrl }, { fetch, caches }));
  assert.equal(result.status, 'ok');
  assert.equal(calls.length, 1);
  assert.ok(cache.deleteCalls.includes(`${MODEL_PREFIX_TEST}${m.sha}`));
  assert.equal(cache.modelPutCalls.length, 1);
});

test('AC1 unreadable cache body falls back to network', async () => {
  const bytes = new Uint8Array([1, 2, 3]);
  const m = await modelFixture(bytes);
  const { caches, cache } = makeCacheMock({ erroringMatchBody: true });
  const { fetch, calls } = fetchSpy(() => bufferResponse(bytes));
  const result = await expectResolves(loadModel({ card: m.card, resolveUrl: m.resolveUrl }, { fetch, caches }));
  assert.equal(result.status, 'ok');
  assert.equal(calls.length, 1);
  assert.equal(cache.modelPutCalls.length, 1);
});

test('AC4 cache hit reports one full-size event', async () => {
  const bytes = new Uint8Array([1, 2, 3, 4]);
  const m = await modelFixture(bytes);
  const { caches } = makeCacheMock();
  const seed = fetchSpy(() => bufferResponse(bytes));
  const seeded = await expectResolves(loadModel({ card: m.card, resolveUrl: m.resolveUrl }, { fetch: seed.fetch, caches }));
  assert.equal(seeded.status, 'ok');

  const { fetch } = fetchSpy();
  const { onProgress, events } = captureProgress();
  const result = await expectResolves(loadModel({ card: m.card, resolveUrl: m.resolveUrl }, { fetch, caches, onProgress }));
  assert.equal(result.status, 'ok');
  assert.deepEqual(events, [[bytes.byteLength, bytes.byteLength]]);
});

test('eviction keeps current plus most recent other', async () => {
  const { caches, cache } = makeCacheMock();
  const bytesA = new Uint8Array([1, 1, 1]);
  const bytesB = new Uint8Array([2, 2, 2]);
  const bytesC = new Uint8Array([3, 3, 3]);
  const A = await modelFixture(bytesA);
  const B = await modelFixture(bytesB);
  const C = await modelFixture(bytesC);

  async function load(m: Awaited<ReturnType<typeof modelFixture>>, bytes: Uint8Array, expectFetch: boolean): Promise<void> {
    const { fetch, calls } = fetchSpy(expectFetch ? () => bufferResponse(bytes) : undefined);
    const result = await expectResolves(loadModel({ card: m.card, resolveUrl: m.resolveUrl }, { fetch, caches }));
    assert.equal(result.status, 'ok');
    assert.equal(calls.length, expectFetch ? 1 : 0);
  }

  await load(A, bytesA, true);
  await load(B, bytesB, true);
  await load(A, bytesA, false);
  await load(C, bytesC, true);

  const modelKeys = (await cache.keys())
    .filter((k) => k.url.startsWith(MODEL_PREFIX_TEST))
    .map((k) => k.url.slice(MODEL_PREFIX_TEST.length));
  assert.deepEqual(new Set(modelKeys), new Set([C.sha, A.sha]));
  assert.deepEqual(await cache.readIndex(), [C.sha, A.sha]);
});

test('eviction touches only komugi-models-v1 and komugi keys', async () => {
  const { cache } = makeCacheMock();
  cache.seedRaw('https://other.invalid/unrelated', new Uint8Array([7]));
  const opens: string[] = [];
  const caches: CacheStorageLike = {
    async open(name: string) {
      opens.push(name);
      return cache;
    },
  };
  const bytes = new Uint8Array([1, 2, 3]);
  const m = await modelFixture(bytes);
  const { fetch } = fetchSpy(() => bufferResponse(bytes));
  const result = await expectResolves(loadModel({ card: m.card, resolveUrl: m.resolveUrl }, { fetch, caches }));
  assert.equal(result.status, 'ok');
  assert.deepEqual(opens, ['komugi-models-v1']);
  const unrelated = await cache.match('https://other.invalid/unrelated');
  assert.ok(unrelated !== undefined);
});

test('quota error evicts and retries once', async () => {
  {
    const { caches, cache } = makeCacheMock({ quotaFailOnPutNumber: 3 });
    const bytesA = new Uint8Array([1, 1, 1]);
    const bytesB = new Uint8Array([2, 2, 2]);
    const bytesC = new Uint8Array([3, 3, 3]);
    const A = await modelFixture(bytesA);
    const B = await modelFixture(bytesB);
    const C = await modelFixture(bytesC);

    async function load(m: Awaited<ReturnType<typeof modelFixture>>, bytes: Uint8Array): Promise<void> {
      const { fetch } = fetchSpy(() => bufferResponse(bytes));
      const result = await expectResolves(loadModel({ card: m.card, resolveUrl: m.resolveUrl }, { fetch, caches }));
      assert.equal(result.status, 'ok');
    }

    await load(A, bytesA);
    await load(B, bytesB);
    const before = cache.modelPutCalls.length;
    await load(C, bytesC);
    assert.equal(cache.modelPutCalls.length - before, 2);

    const modelKeys = (await cache.keys())
      .filter((k) => k.url.startsWith(MODEL_PREFIX_TEST))
      .map((k) => k.url.slice(MODEL_PREFIX_TEST.length));
    assert.deepEqual(modelKeys, [C.sha]);
  }

  {
    const { caches, cache } = makeCacheMock({ quotaAlwaysOnModelPut: true });
    const bytesC = new Uint8Array([4, 4, 4]);
    const C = await modelFixture(bytesC);
    const { fetch } = fetchSpy(() => bufferResponse(bytesC));
    const result = await expectResolves(loadModel({ card: C.card, resolveUrl: C.resolveUrl }, { fetch, caches }));
    assert.equal(result.status, 'ok');
    assert.equal(cache.modelPutCalls.length, 2);
    const modelKeys = (await cache.keys())
      .filter((k) => k.url.startsWith(MODEL_PREFIX_TEST))
      .map((k) => k.url.slice(MODEL_PREFIX_TEST.length));
    assert.ok(!modelKeys.includes(C.sha));
  }
});

test('cache unavailable still loads', async () => {
  const bytes = new Uint8Array([1, 2, 3]);
  const sha = await sha256Hex(bytes);
  const card = makeCard(cardId(sha), { path: 'm.onnx', sha256: sha, bytes: bytes.byteLength });

  {
    const { fetch } = fetchSpy(() => bufferResponse(bytes));
    const caches: CacheStorageLike = {
      open: async () => {
        throw new Error('open boom');
      },
    };
    const result = await expectResolves(loadModel({ card, resolveUrl: UNUSED_RESOLVE_URL }, { fetch, caches }));
    assert.equal(result.status, 'ok');
  }

  {
    const original = Object.getOwnPropertyDescriptor(globalThis, 'caches');
    Object.defineProperty(globalThis, 'caches', {
      configurable: true,
      get() {
        throw new Error('caches getter boom');
      },
    });
    try {
      const { fetch } = fetchSpy(() => bufferResponse(bytes));
      const result = await expectResolves(loadModel({ card, resolveUrl: UNUSED_RESOLVE_URL }, { fetch }));
      assert.equal(result.status, 'ok');
    } finally {
      if (original) Object.defineProperty(globalThis, 'caches', original);
      else delete (globalThis as { caches?: unknown }).caches;
    }
  }
});

// ---------------------------------------------------------------------------
// End-to-end (unit 3, section 8): loadModel's blob: URL through the real
// worker host with a fake ORT module and the REAL globalThis.fetch (no
// `fetch` option on installWorkerHost), then through KomugiClient.
// makeFakeScope/flushUntil/makeFakeOrtModule are COPIED from
// worker_host_ort.test.ts (never imported).

type Posted = WorkerReply | WorkerInitReply | WorkerPonderSettingsReply | WorkerPonderNowReply;

// Copy of worker_host_ort.test.ts's fake worker global, with an optional
// onPost hook so an adapter can relay the worker's posts to a client.
function makeFakeScope(onPost?: (reply: Posted) => void): {
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
      if (onPost !== undefined) onPost(reply);
    },
  };
  const send = (data: unknown): void => {
    for (const listener of listeners) listener({ data });
  };
  return { scope, posted, send };
}

// Real crypto.subtle.digest and real fetch resolve as macrotasks, so this
// yields through a real timer tick each iteration.
async function flushUntil(predicate: () => boolean, maxTicks = 2000): Promise<void> {
  for (let i = 0; i < maxTicks; i++) {
    if (predicate()) return;
    await new Promise<void>((resolve) => setTimeout(resolve, 0));
  }
}

// Copy of worker_host_ort.test.ts's FAKE OrtModule, without the event log.
function makeFakeOrtModule(): OrtModule {
  let numThreads = 1;
  let wasmPaths: string | { mjs?: string; wasm?: string } | undefined;
  let proxy: boolean | undefined;
  const wasm = {
    get numThreads(): number {
      return numThreads;
    },
    set numThreads(value: number) {
      numThreads = value;
    },
    get wasmPaths(): string | { mjs?: string; wasm?: string } | undefined {
      return wasmPaths;
    },
    set wasmPaths(value: string | { mjs?: string; wasm?: string } | undefined) {
      wasmPaths = value;
    },
    get proxy(): boolean | undefined {
      return proxy;
    },
    set proxy(value: boolean | undefined) {
      proxy = value;
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
      async create(): Promise<OrtSession> {
        return {
          async run(feeds: Record<string, unknown>): Promise<Record<string, { data: Float32Array }>> {
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

const E2E_MODEL_BYTES = new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
const E2E_ORT_ASSETS = { wasm: { mjs: 'wasm.mjs', wasm: 'wasm.wasm' } };

// loadModel over a mocked asset fetch, cache off: the result's model.url is
// a real blob: URL from URL.createObjectURL.
async function loadE2eModel(): Promise<{ model: KomugiModel; sha: string }> {
  const sha = await sha256Hex(E2E_MODEL_BYTES);
  const card = makeCard(cardId(sha), { path: 'm.onnx', sha256: sha, bytes: E2E_MODEL_BYTES.byteLength }, { planeMapVersion: 4 });
  const { fetch } = fetchSpy(() => bufferResponse(E2E_MODEL_BYTES));
  const result = await expectResolves(loadModel({ card, resolveUrl: UNUSED_RESOLVE_URL }, { fetch, cache: false }));
  assert.equal(result.status, 'ok', `loadModel failed: ${JSON.stringify(result)}`);
  if (result.status !== 'ok') throw new Error('unreachable');
  assert.ok(result.model.url.startsWith('blob:'), `model.url must be a blob: URL, got ${result.model.url}`);
  return { model: result.model, sha };
}

function isErr(reply: unknown): boolean {
  return typeof reply === 'object' && reply !== null && 'err' in reply;
}

// In-process KomugiWorkerLike adapter: each createWorker() builds a fresh
// fake scope with the real worker host (real fetch). Init messages are
// forwarded; search messages are recorded and HELD, so a play() stays
// pending until stop(). The scope's posts are relayed to the client.
function makeE2eAdapter(): {
  createWorker: () => KomugiWorkerLike;
  spawns: () => number;
  initReplies: unknown[];
  held: Array<Record<string, unknown>>;
  crash: () => void;
} {
  let spawns = 0;
  const initReplies: unknown[] = [];
  const held: Array<Record<string, unknown>> = [];
  let crashCurrent: () => void = () => undefined;
  const createWorker = (): KomugiWorkerLike => {
    spawns += 1;
    const messageListeners: Array<(event: { data: unknown }) => void> = [];
    const errorListeners: Array<(event: unknown) => void> = [];
    const initIds = new Set<string>();
    let terminated = false;
    const { scope, send } = makeFakeScope((reply) => {
      if (terminated) return;
      if (initIds.has(String((reply as { id?: unknown }).id))) initReplies.push(reply);
      for (const listener of messageListeners) listener({ data: reply });
    });
    installWorkerHost(scope, { loadOrt: async () => makeFakeOrtModule() });
    crashCurrent = () => {
      for (const listener of errorListeners) listener(new Error('idle crash'));
    };
    return {
      postMessage(message: unknown): void {
        const record = message as Record<string, unknown>;
        if (record.type === 'init') {
          initIds.add(String(record.id));
          send(message);
        } else {
          held.push(record);
        }
      },
      terminate(): void {
        terminated = true;
      },
      addEventListener(type: 'message' | 'error' | 'messageerror', listener: (event: { data: unknown }) => void): void {
        if (type === 'message') messageListeners.push(listener);
        else errorListeners.push(listener as (event: unknown) => void);
      },
    };
  };
  return { createWorker, spawns: () => spawns, initReplies, held, crash: () => crashCurrent() };
}

test('e2e blob URL init via real fetch', async () => {
  const { model, sha } = await loadE2eModel();
  try {
    const { scope, posted, send } = makeFakeScope();
    installWorkerHost(scope, { loadOrt: async () => makeFakeOrtModule() });
    const initRequest: WorkerInitRequest = { id: 'init-1', type: 'init', modelUrl: model.url, modelSha256: sha, ortAssets: E2E_ORT_ASSETS };
    send(initRequest);
    await flushUntil(() => posted.length >= 1);
    assert.equal(posted.length, 1, `expected exactly one reply, got ${JSON.stringify(posted)}`);
    assert.ok(!isErr(posted[0]), `init over the blob: URL should succeed, got ${JSON.stringify(posted[0])}`);
    assert.equal((posted[0] as { id: string }).id, 'init-1');
  } finally {
    model.release?.();
  }
});

test('e2e revoked blob URL is model-fetch', async () => {
  const { model, sha } = await loadE2eModel();
  URL.revokeObjectURL(model.url);
  const { scope, posted, send } = makeFakeScope();
  installWorkerHost(scope, { loadOrt: async () => makeFakeOrtModule() });
  const initRequest: WorkerInitRequest = { id: 'init-1', type: 'init', modelUrl: model.url, modelSha256: sha, ortAssets: E2E_ORT_ASSETS };
  send(initRequest);
  await flushUntil(() => posted.length >= 1);
  assert.equal(posted.length, 1, `expected exactly one reply, got ${JSON.stringify(posted)}`);
  assert.ok(isErr(posted[0]), `init over a revoked blob: URL must fail, got ${JSON.stringify(posted[0])}`);
  assert.equal((posted[0] as { code?: unknown }).code, 'model-fetch');
});

test('e2e client restarts reuse the blob until dispose', async () => {
  const originalRevoke = URL.revokeObjectURL;
  const revoked: string[] = [];
  URL.revokeObjectURL = (url: string): void => {
    revoked.push(url);
    originalRevoke.call(URL, url);
  };
  try {
    const { model } = await loadE2eModel();
    const adapter = makeE2eAdapter();
    const client = new KomugiClient({ createWorker: adapter.createWorker, model, ortAssets: E2E_ORT_ASSETS });

    // 1. init() is ready (spawn 1).
    const ready = await client.init();
    assert.equal(ready.status, 'ready', `init failed: ${JSON.stringify(ready)}`);
    assert.equal(adapter.spawns(), 1);

    // 2. idle crash: onCrash -> toIdle.
    adapter.crash();
    assert.equal(client.state, 'idle');

    // 3. play(): spawn 2 inits ok from the same blob URL; the search is held.
    const play1 = client.play({ fen: ADVANCED_POSITION, sims: 1 });
    await flushUntil(() => adapter.held.length >= 1);
    assert.equal(adapter.held.length, 1, 'the first play must reach a search message');
    assert.equal(adapter.spawns(), 2);

    // 4. stop() while that play is pending.
    client.stop();
    assert.deepEqual(await play1, { status: 'stopped' });

    // 5. play() again: spawn 3 inits ok.
    const play2 = client.play({ fen: ADVANCED_POSITION, sims: 1 });
    await flushUntil(() => adapter.held.length >= 2);
    assert.equal(adapter.held.length, 2, 'the second play must reach a search message');
    assert.equal(adapter.spawns(), 3);

    // 6. stop(), then dispose().
    client.stop();
    assert.deepEqual(await play2, { status: 'stopped' });
    assert.deepEqual(revoked, [], 'nothing may revoke the URL before dispose()');
    client.dispose();

    assert.equal(adapter.initReplies.length, 3, `expected three init replies, got ${JSON.stringify(adapter.initReplies)}`);
    for (const reply of adapter.initReplies) assert.ok(!isErr(reply), `init reply must be ok, got ${JSON.stringify(reply)}`);
    assert.deepEqual(revoked, [model.url], 'dispose() must release the blob URL exactly once');
    await assert.rejects(globalThis.fetch(model.url), 'the blob URL must be revoked after dispose()');
  } finally {
    URL.revokeObjectURL = originalRevoke;
  }
});

test('e2e v4 card posts planeMapVersion 4', async () => {
  const { model } = await loadE2eModel();
  assert.equal(model.compatibility?.planeMapVersion, 4);
  const adapter = makeE2eAdapter();
  const client = new KomugiClient({ createWorker: adapter.createWorker, model, ortAssets: E2E_ORT_ASSETS });
  const play = client.play({ fen: ADVANCED_POSITION, sims: 1 });
  await flushUntil(() => adapter.held.length >= 1);
  assert.equal(adapter.held.length, 1, 'the play must reach a search message');
  assert.equal(adapter.held[0].planeMapVersion, 4);
  client.stop();
  assert.deepEqual(await play, { status: 'stopped' });
  client.dispose();
});

// ---------------------------------------------------------------------------
// README (unit 3).

test('README Loading a model mentions blob: connect-src and release on dispose', () => {
  const text = readFileSync(new URL('../README.md', import.meta.url), 'utf8');
  const start = text.indexOf('\n## Loading a model\n');
  assert.ok(start >= 0, 'README has a "Loading a model" section');
  const next = text.indexOf('\n## ', start + 1);
  const section = next >= 0 ? text.slice(start, next) : text.slice(start);
  assert.ok(section.includes('connect-src'), 'section mentions connect-src');
  assert.ok(section.includes('blob:'), 'section mentions blob:');
  assert.ok(section.includes('dispose()'), 'section mentions dispose()');
  assert.doesNotMatch(text, /KOM-[0-9]+/);
});
