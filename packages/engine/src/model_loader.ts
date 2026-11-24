// KOM-1200 units 1-2 (design docs/design/kom1200-model-loading.md sections
// 2-7): `loadModel` resolves a KOM-1188 model card (by manifest URL, parsed
// card, or models.json id) into a ready-to-use `KomugiModel`, streaming and
// sha256-verifying the `.onnx` asset with progress reporting, and (unit 2,
// section 5) persisting verified bytes in Cache Storage keyed by sha256,
// with an LRU of two Komugi entries and one quota retry.
// Never rejects: every path resolves to a `LoadModelResult` (section 7).

import type { ModelAsset, ModelCard, ModelResolution, ResolveAssetUrl, EngineCompatibilityKey } from './model_registry.ts';
import { resolveModelCard, resolveModelById } from './model_registry.ts';
import type { KomugiModel } from './client.ts';
import type { KomugiFailure } from './client_failures.ts';

export interface StreamFetchLike {
  (url: string): Promise<{
    ok: boolean;
    status: number;
    headers: { get(name: string): string | null };
    body: ReadableStream<Uint8Array> | null;
    arrayBuffer(): Promise<ArrayBuffer>;
    text(): Promise<string>;
  }>;
}

// Narrower than DOM's real Cache/CacheStorage -- only the four members this
// module calls. keys() returns a READONLY array, because a real Cache's
// keys() is Promise<readonly Request[]>, which is not assignable to a
// mutable { url: string }[] under strict tsc (TS2322). With readonly, a
// real Cache/CacheStorage is assignable without a cast.
export interface CacheLike {
  match(key: string): Promise<Response | undefined>;
  put(key: string, response: Response): Promise<void>;
  delete(key: string): Promise<boolean>;
  keys(): Promise<readonly { url: string }[]>;
}
export interface CacheStorageLike {
  open(name: string): Promise<CacheLike>;
}

export type ModelLocator =
  | { manifestUrl: string; resolveUrl?: ResolveAssetUrl }
  | { card: unknown; resolveUrl: ResolveAssetUrl }
  | { id: string; index: unknown; card: unknown; resolveUrl: ResolveAssetUrl };

export interface LoadModelOptions {
  cache?: boolean; // default true -- try Cache Storage; false skips it entirely
  onProgress?: (loaded: number, total: number | null) => void;
  fetch?: StreamFetchLike; // default: globalThis.fetch wrapped
  caches?: CacheStorageLike; // default: globalThis.caches, read inside the try
  baseUrl?: string; // manifestUrl base; default globalThis.location?.href
  expected?: EngineCompatibilityKey; // default: ENGINE_COMPATIBILITY_KEY (model_registry.ts)
}

export type LoadModelResult = { status: 'ok'; model: KomugiModel; card: ModelCard } | { status: 'failed'; failure: KomugiFailure };

// `crypto.subtle.digest`/`Blob` need a concrete ArrayBuffer-backed view, not
// the wider ArrayBufferLike a bare `Uint8Array` (or a stream chunk) carries.
type Bytes = Uint8Array<ArrayBuffer>;

type AssetResult = { status: 'ok'; bytes: Bytes } | { status: 'failed'; failure: KomugiFailure };

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function modelFetchFailure(message: string): { status: 'failed'; failure: KomugiFailure } {
  return { status: 'failed', failure: { code: 'model-fetch', stage: 'init', message } };
}

function incompatibleFailure(message: string): { status: 'failed'; failure: KomugiFailure } {
  return { status: 'failed', failure: { code: 'incompatible-model', stage: 'init', message } };
}

// Copied from client.ts's callHost (not exported there): a throwing host
// callback is re-thrown on a microtask instead of aborting the caller.
// `globalThis.queueMicrotask` is read at call time, never captured, so a
// test can swap in a capturing stub.
function callHost(callback: () => void): void {
  try {
    callback();
  } catch (error) {
    globalThis.queueMicrotask(() => {
      throw error;
    });
  }
}

function makeReport(onProgress: LoadModelOptions['onProgress']): (loaded: number, total: number | null) => void {
  if (onProgress === undefined) return () => undefined;
  return (loaded, total) => callHost(() => onProgress(loaded, total));
}

function makeRelease(url: string): () => void {
  let revoked = false;
  return () => {
    if (revoked) return;
    revoked = true;
    URL.revokeObjectURL(url);
  };
}

async function sha256Hex(bytes: Bytes): Promise<string> {
  const digest = await globalThis.crypto.subtle.digest('SHA-256', bytes);
  return Array.from(new Uint8Array(digest))
    .map((byte) => byte.toString(16).padStart(2, '0'))
    .join('');
}

async function verifyBytes(bytes: Bytes, asset: ModelAsset): Promise<AssetResult> {
  if (globalThis.crypto?.subtle === undefined) {
    return { status: 'failed', failure: { code: 'init-failed', stage: 'init', message: 'model verification needs crypto.subtle, which requires a secure context' } };
  }
  const digest = await sha256Hex(bytes);
  const expected = asset.sha256.toLowerCase();
  if (digest !== expected) {
    return { status: 'failed', failure: { code: 'sha-mismatch', stage: 'init', message: `model sha256 mismatch: expected ${expected}, got ${digest}` } };
  }
  return { status: 'ok', bytes };
}

function contentLengthTotal(response: { headers: { get(name: string): string | null } }): number | null {
  const encoding = response.headers.get('content-encoding');
  if (encoding !== null && encoding !== 'identity') return null;
  const raw = response.headers.get('content-length');
  if (raw === null) return null;
  const parsed = Number(raw);
  return Number.isInteger(parsed) && parsed > 0 ? parsed : null;
}

function concat(chunks: Uint8Array[]): Bytes {
  let size = 0;
  for (const chunk of chunks) size += chunk.byteLength;
  const out = new Uint8Array(size);
  let offset = 0;
  for (const chunk of chunks) {
    out.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return out;
}

// Section 4's streamed fetch: `cardTotal` (asset.bytes) is the only size
// that may truncate a body or fail a short read; `Content-Length` drives
// progress only and is abandoned (set to null) on any overrun. Every fetch/
// read/stream error is caught here and mapped to 'model-fetch', never a
// rejection of loadModel.
async function fetchAsset(doFetch: StreamFetchLike, url: string, asset: ModelAsset, report: (loaded: number, total: number | null) => void): Promise<AssetResult> {
  let response: Awaited<ReturnType<StreamFetchLike>>;
  try {
    response = await doFetch(url);
  } catch (error) {
    return modelFetchFailure(`model fetch failed: ${errorMessage(error)}`);
  }
  if (!response.ok) return modelFetchFailure(`status ${response.status}`);

  const cardTotal = asset.bytes ?? null;
  let total = cardTotal ?? contentLengthTotal(response);
  const reader = response.body?.getReader();

  if (reader === undefined) {
    let buf: Bytes;
    try {
      buf = new Uint8Array(await response.arrayBuffer());
    } catch (error) {
      return modelFetchFailure(`model read failed: ${errorMessage(error)}`);
    }
    if (cardTotal !== null && buf.byteLength < cardTotal) return modelFetchFailure('short read');
    if (cardTotal !== null && buf.byteLength > cardTotal) buf = buf.subarray(0, cardTotal);
    if (total !== null && buf.byteLength !== total) total = null;
    if (total !== null) report(0, total);
    report(buf.byteLength, total);
    return verifyBytes(buf, asset);
  }

  const chunks: Uint8Array[] = [];
  let loaded = 0;
  if (total !== null) report(0, total);
  for (;;) {
    let next: ReadableStreamReadResult<Uint8Array>;
    try {
      next = await reader.read();
    } catch (error) {
      return modelFetchFailure(`model stream failed: ${errorMessage(error)}`);
    }
    if (next.done) break;
    const value = next.value;
    if (value.byteLength === 0) continue;
    if (cardTotal !== null && loaded + value.byteLength > cardTotal) {
      await reader.cancel().catch(() => undefined);
      const rest = cardTotal - loaded;
      if (rest > 0) {
        chunks.push(value.subarray(0, rest));
        loaded = cardTotal;
        report(loaded, cardTotal);
      }
      return verifyBytes(concat(chunks), asset);
    }
    if (total !== null && loaded + value.byteLength > total) total = null;
    chunks.push(value);
    loaded += value.byteLength;
    report(loaded, total);
  }
  if (cardTotal !== null && loaded !== cardTotal) return modelFetchFailure(`got ${loaded} of ${cardTotal} bytes`);
  return verifyBytes(concat(chunks), asset);
}

// Section 5: a single, versioned cache name; keys are the asset's own
// sha256 under a map-only, unresolvable `.invalid` host (never fetched).
const CACHE_NAME = 'komugi-models-v1';
const MODEL_PREFIX = 'https://komugi.invalid/model/';
const LRU_KEY = 'https://komugi.invalid/lru';

function cacheKey(sha256: string): string {
  return `${MODEL_PREFIX}${sha256}`;
}

function isQuotaExceeded(error: unknown): boolean {
  return error instanceof DOMException && error.name === 'QuotaExceededError';
}

async function readLruIndex(cache: CacheLike): Promise<string[]> {
  try {
    const res = await cache.match(LRU_KEY);
    if (res === undefined) return [];
    const parsed: unknown = JSON.parse(new TextDecoder().decode(await res.arrayBuffer()));
    if (Array.isArray(parsed) && parsed.every((entry) => typeof entry === 'string')) return parsed;
    return [];
  } catch {
    return [];
  }
}

// Keeps the current model plus the most recently used *other* Komugi
// entry; runs on every successful load, hit or miss, before any write
// (section 5). Every step is caught and ignored.
async function touchLru(cache: CacheLike, sha256: string): Promise<void> {
  const index = await readLruIndex(cache);
  const other = index.find((entry) => entry !== sha256);
  const keep = other !== undefined ? [sha256, other] : [sha256];
  try {
    const keys = await cache.keys();
    for (const key of keys) {
      if (!key.url.startsWith(MODEL_PREFIX)) continue;
      const sha = key.url.slice(MODEL_PREFIX.length);
      if (!keep.includes(sha)) await cache.delete(key.url).catch(() => undefined);
    }
  } catch {
    // ignored: eviction never fails a load
  }
  try {
    await cache.put(LRU_KEY, new Response(JSON.stringify(keep)));
  } catch {
    // ignored: the index write never fails a load
  }
}

// Evicts every *other* Komugi model entry, used only as the one quota
// retry's eviction step (section 5): the normal touchLru eviction above
// already ran before this put, so at most one other entry remains.
async function evictOtherModelKeys(cache: CacheLike, keepKey: string): Promise<void> {
  const keys = await cache.keys();
  for (const key of keys) {
    if (key.url.startsWith(MODEL_PREFIX) && key.url !== keepKey) {
      await cache.delete(key.url).catch(() => undefined);
    }
  }
}

// Section 5 write path: evicts first (touchLru, by the caller), then
// writes. A QuotaExceededError evicts the remaining other entry and
// retries once; any other rejection, or a second failure, is ignored.
// A failed write never fails the load.
async function putModel(cache: CacheLike, key: string, bytes: Bytes): Promise<void> {
  try {
    await cache.put(key, new Response(bytes));
    return;
  } catch (error) {
    if (!isQuotaExceeded(error)) return;
  }
  try {
    await evictOtherModelKeys(cache, key);
  } catch {
    // ignored
  }
  try {
    await cache.put(key, new Response(bytes));
  } catch {
    // ignored: the load continues uncached
  }
}

// Section 5 read path: a cache hit is re-hashed (never trusted blindly); a
// damaged or non-matching entry is evicted and treated as a miss, never a
// failure.
async function readCacheHit(cache: CacheLike, asset: ModelAsset, report: (loaded: number, total: number | null) => void): Promise<Bytes | undefined> {
  const key = cacheKey(asset.sha256);
  const cached = await cache.match(key).catch(() => undefined);
  if (cached === undefined) return undefined;
  const buf = await cached.arrayBuffer().catch(() => undefined);
  if (buf !== undefined) {
    const bytes = new Uint8Array(buf) as Bytes;
    const digest = await sha256Hex(bytes);
    if (digest === asset.sha256.toLowerCase()) {
      report(bytes.byteLength, bytes.byteLength);
      return bytes;
    }
  }
  await cache.delete(key).catch(() => undefined);
  return undefined;
}

function buildModel(resolution: { model: KomugiModel }, bytes: Bytes): KomugiModel {
  const blobUrl = URL.createObjectURL(new Blob([bytes]));
  return { ...resolution.model, url: blobUrl, release: makeRelease(blobUrl) };
}

// Never rejects: every path below resolves to a LoadModelResult (section 7).
export async function loadModel(locator: ModelLocator, options: LoadModelOptions = {}): Promise<LoadModelResult> {
  try {
    const doFetch = options.fetch ?? (globalThis.fetch as unknown as StreamFetchLike);
    const report = makeReport(options.onProgress);
    const expected = options.expected;

    let resolution: ModelResolution;
    if ('id' in locator) {
      resolution = resolveModelById(locator.id, locator.index, locator.card, locator.resolveUrl, expected);
    } else if ('card' in locator) {
      resolution = resolveModelCard(locator.card, locator.resolveUrl, expected);
    } else {
      const baseUrl = options.baseUrl ?? (globalThis as { location?: { href: string } }).location?.href;
      let cardUrl: string;
      try {
        cardUrl = new URL(locator.manifestUrl, baseUrl).toString();
      } catch (error) {
        return modelFetchFailure(`manifestUrl: ${errorMessage(error)}`);
      }
      let text: string;
      try {
        const response = await doFetch(cardUrl);
        if (!response.ok) return modelFetchFailure(`manifest fetch status ${response.status}`);
        text = await response.text();
      } catch (error) {
        return modelFetchFailure(`manifest fetch failed: ${errorMessage(error)}`);
      }
      let parsed: unknown;
      try {
        parsed = JSON.parse(text);
      } catch {
        return incompatibleFailure('manifest body is not valid JSON');
      }
      const resolveUrl: ResolveAssetUrl = locator.resolveUrl ?? ((asset) => new URL(asset.path, cardUrl).toString());
      resolution = resolveModelCard(parsed, resolveUrl, expected);
    }

    if (resolution.status === 'failed') return { status: 'failed', failure: resolution.failure };

    const asset = resolution.card.assets.onnx;

    let cache: CacheLike | undefined;
    if (options.cache !== false) {
      let storage: CacheStorageLike | undefined;
      try {
        storage = options.caches ?? globalThis.caches;
      } catch {
        storage = undefined;
      }
      cache = await storage?.open(CACHE_NAME).catch(() => undefined);
    }

    if (cache !== undefined) {
      const hit = await readCacheHit(cache, asset, report);
      if (hit !== undefined) {
        await touchLru(cache, asset.sha256);
        return { status: 'ok', model: buildModel(resolution, hit), card: resolution.card };
      }
    }

    const fetched = await fetchAsset(doFetch, resolution.model.url, asset, report);
    if (fetched.status === 'failed') return fetched;

    if (cache !== undefined) {
      await touchLru(cache, asset.sha256);
      await putModel(cache, cacheKey(asset.sha256), fetched.bytes);
    }

    return { status: 'ok', model: buildModel(resolution, fetched.bytes), card: resolution.card };
  } catch (error) {
    return { status: 'failed', failure: { code: 'init-failed', stage: 'init', message: errorMessage(error) } };
  }
}
