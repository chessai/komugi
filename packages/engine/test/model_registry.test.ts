// KOM-1188 unit 2 (design docs/design/kom1188-model-releases.md sections 5,
// 6 AC4/AC5, 7 unit 2 acceptance 3-7): model_registry.ts's guard, the
// compatibility check, resolve-by-card/by-id, and an end-to-end resolve into
// `installWorkerHost`'s real `makeInitEnv`. Fakes only: no real model, no
// network. Planted invalid cards are built inline (no fixture file).

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';

import type { BackendKind } from '../src/backend.ts';
import { ACTION_ENCODING_VERSION, ACTION_SIZE } from '../src/constants.ts';
import { CLIENT_ERROR_CODES } from '../src/client_failures.ts';
import type { OrtModule, OrtSession } from '../src/ort_core.ts';
import type { WorkerInitReply, WorkerInitRequest, WorkerReply } from '../src/protocol.ts';
import { installWorkerHost } from '../src/worker_host.ts';
import type { WorkerHostOptions } from '../src/worker_host.ts';
import {
  ENGINE_COMPATIBILITY_KEY,
  checkCompatibility,
  isModelCard,
  resolveModelById,
  resolveModelCard,
} from '../src/model_registry.ts';
import type { ModelAsset, ModelCard, ModelResolution } from '../src/model_registry.ts';

const REGISTRY_SOURCE = new URL('../src/model_registry.ts', import.meta.url);
const README = new URL('../README.md', import.meta.url);

// ---------------------------------------------------------------------------
// Helpers copied from worker_host_ort.test.ts (file-local there; section 6
// AC4 row: copied, not imported).

type Posted = WorkerReply | WorkerInitReply;

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

// The real makeInitEnv awaits crypto.subtle.digest (a macrotask): yield
// through real timer ticks.
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

function makeFakeOrtModule(): OrtModule {
  const wasm = { numThreads: 1, wasmPaths: undefined as string | { mjs?: string; wasm?: string } | undefined, proxy: undefined as boolean | undefined };
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
            const bucket = (feeds.obs as { dims: readonly number[] }).dims[0] as number;
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

const ORT_ASSETS_WASM_ONLY = { wasm: { mjs: 'wasm.mjs', wasm: 'wasm.wasm' } };

// ---------------------------------------------------------------------------
// Inline cards.

const ONNX_BYTES = new Uint8Array([11, 22, 33, 44, 55, 66, 77, 88]);
const ONNX_SHA = await sha256Hex(ONNX_BYTES);
const PT_SHA = await sha256Hex(new Uint8Array([9, 9, 9]));
const SRC_SHA = 'c'.repeat(64);
const ID = `komugi-c6-s580800-${ONNX_SHA.slice(0, 8)}`;

function makeCard(overrides: { planeMapVersion?: unknown; rulesProfile?: unknown; actionEncodingVersion?: unknown } = {}): Record<string, unknown> {
  return {
    id: ID,
    campaign: 'c6',
    iteration: 1936,
    step: 580800,
    compatibilityKey: {
      planeMapVersion: overrides.planeMapVersion ?? 3,
      actionEncodingVersion: overrides.actionEncodingVersion ?? ACTION_ENCODING_VERSION,
      rulesProfile: overrides.rulesProfile ?? 'komugi',
      architectureTag: 'b10c128',
    },
    modesTrained: ['intro', 'beginner', 'intermediate', 'advanced'],
    assets: {
      pt: { path: `${ID}.pt`, sha256: PT_SHA },
      onnx: { path: `${ID}.onnx`, sha256: ONNX_SHA },
    },
    sourceCheckpoint: { name: 'checkpoint-000580800-0123456789abcdef.pt', sha256: SRC_SHA },
    exportOpset: 17,
    exportParity: { n: 4, maxAbsDelta: 6.2e-6, atol: 3.9e-4, rtol: 1.1e-4 },
    evalPointers: ['evals/c6-s580800-gauntlet.json'],
    license: 'MIT, Copyright (c) 2026 chessai',
    createdAt: '2026-11-22T00:00:00Z',
  };
}

function makeIndex(card: Record<string, unknown> = makeCard()): Record<string, unknown> {
  const id = card.id as string;
  const onnx = (card.assets as { onnx: ModelAsset }).onnx;
  return {
    schemaVersion: 1,
    recommended: { advanced: id },
    models: { [id]: { card: `${id}.card.json`, onnxSha256: onnx.sha256 } },
  };
}

const resolveUrl = (asset: ModelAsset): string => `https://example.invalid/release/${asset.path}`;

function without(obj: Record<string, unknown>, key: string): Record<string, unknown> {
  const copy = { ...obj };
  delete copy[key];
  return copy;
}

function expectFailed(result: ModelResolution, code: string): string {
  assert.equal(result.status, 'failed', `expected failed, got ${JSON.stringify(result)}`);
  assert.ok(result.status === 'failed' && result.failure.code === code, `expected ${code}, got ${JSON.stringify(result)}`);
  return result.status === 'failed' ? result.failure.message : '';
}

// ---------------------------------------------------------------------------
// isModelCard (acceptance 6).

test('isModelCard accepts the reference card', () => {
  assert.equal(isModelCard(makeCard()), true);
});

test('isModelCard rejects a card missing each required field, without throwing', () => {
  const card = makeCard();
  for (const key of Object.keys(card)) {
    assert.equal(isModelCard(without(card, key)), false, `card without ${key} must be rejected`);
  }
  assert.equal(isModelCard(without(card, 'compatibilityKey')), false);
  assert.equal(isModelCard(without(card, 'sourceCheckpoint')), false);
  const key = card.compatibilityKey as Record<string, unknown>;
  for (const field of Object.keys(key)) {
    assert.equal(isModelCard({ ...card, compatibilityKey: without(key, field) }), false, `compatibilityKey without ${field}`);
  }
});

test('isModelCard rejects malformed values and non-objects', () => {
  const card = makeCard();
  const assets = card.assets as { pt: ModelAsset; onnx: ModelAsset };
  const planted: Array<[string, unknown]> = [
    ['null', null],
    ['array', []],
    ['string', 'card'],
    ['planeMapVersion 5', makeCard({ planeMapVersion: 5 })],
    ['rulesProfile bogus', makeCard({ rulesProfile: 'chess' })],
    ['createdAt yesterday', { ...card, createdAt: 'yesterday' }],
    ['createdAt bare date', { ...card, createdAt: '2026-11-22' }],
    ['license other', { ...card, license: 'GPL' }],
    ['absolute pt path', { ...card, assets: { ...assets, pt: { path: '/abs/model.pt', sha256: PT_SHA } } }],
    ['uppercase sha', { ...card, assets: { ...assets, onnx: { path: 'm.onnx', sha256: ONNX_SHA.toUpperCase() } } }],
    ['bad id', { ...card, id: 'komugi-C6-s1-zz' }],
    ['empty modesTrained', { ...card, modesTrained: [] }],
    ['negative step', { ...card, step: -1 }],
    ['sourceCheckpoint path', { ...card, sourceCheckpoint: { name: 'a/b.pt', sha256: SRC_SHA } }],
  ];
  for (const [label, value] of planted) assert.equal(isModelCard(value), false, label);
});

// ---------------------------------------------------------------------------
// Optional asset `bytes` (design kom1200-model-loading.md section 2/9 unit 0).

test('isModelCard accepts a card without asset bytes', () => {
  const card = makeCard();
  const assets = card.assets as { pt: ModelAsset; onnx: ModelAsset };
  assert.equal('bytes' in assets.pt, false);
  assert.equal('bytes' in assets.onnx, false);
  assert.equal(isModelCard(card), true);
});

test('isModelCard accepts positive integer asset bytes', () => {
  const card = makeCard();
  const assets = card.assets as { pt: ModelAsset; onnx: ModelAsset };
  const withBytes = { ...card, assets: { pt: { ...assets.pt, bytes: 1234 }, onnx: { ...assets.onnx, bytes: 5678 } } };
  assert.equal(isModelCard(withBytes), true);
});

test('isModelCard rejects asset bytes 0', () => {
  const card = makeCard();
  const assets = card.assets as { pt: ModelAsset; onnx: ModelAsset };
  const zeroBytes = { ...card, assets: { ...assets, onnx: { ...assets.onnx, bytes: 0 } } };
  assert.equal(isModelCard(zeroBytes), false);
});

test('isModelCard rejects non-integer asset bytes', () => {
  const card = makeCard();
  const assets = card.assets as { pt: ModelAsset; onnx: ModelAsset };
  const fractionalBytes = { ...card, assets: { ...assets, onnx: { ...assets.onnx, bytes: 1.5 } } };
  assert.equal(isModelCard(fractionalBytes), false);
});

// ---------------------------------------------------------------------------
// Compatibility key (acceptance 3).

test('incompatible key: planeMapVersion outside {3, 4} fails incompatible-model', () => {
  const result = resolveModelCard(makeCard({ planeMapVersion: 5 }), resolveUrl);
  assert.ok(result.status === 'failed' && result.failure.code === 'incompatible-model');
  // checkCompatibility alone, on a cast (unguarded) card.
  const failure = checkCompatibility(makeCard({ planeMapVersion: 2 }) as unknown as ModelCard);
  assert.equal(failure?.code, 'incompatible-model');
  assert.equal(failure?.stage, 'init');
  assert.match(failure?.message ?? '', /^planeMapVersion: /);
});

test('incompatible key: wrong actionEncodingVersion fails incompatible-model', () => {
  const result = resolveModelCard(makeCard({ actionEncodingVersion: ACTION_ENCODING_VERSION + 1 }), resolveUrl);
  assert.ok(result.status === 'failed' && result.failure.code === 'incompatible-model');
  assert.equal(
    result.status === 'failed' ? result.failure.message : '',
    `actionEncodingVersion: card says ${ACTION_ENCODING_VERSION + 1}, engine expects ${ACTION_ENCODING_VERSION}`,
  );
  const custom = resolveModelCard(makeCard(), resolveUrl, { actionEncodingVersion: 99 });
  assert.ok(custom.status === 'failed' && custom.failure.code === 'incompatible-model');
});

test('ENGINE_COMPATIBILITY_KEY carries only ACTION_ENCODING_VERSION', () => {
  assert.deepEqual(ENGINE_COMPATIBILITY_KEY, { actionEncodingVersion: ACTION_ENCODING_VERSION });
});

test('rulesProfile and architectureTag never fail the check (recorded only)', () => {
  for (const rulesProfile of ['komugi', 'gungijs', 'ligungi']) {
    assert.equal(resolveModelCard(makeCard({ rulesProfile }), resolveUrl).status, 'ok', rulesProfile);
  }
  const odd = { ...makeCard(), compatibilityKey: { ...(makeCard().compatibilityKey as object), architectureTag: 'b99c7' } };
  assert.equal(resolveModelCard(odd, resolveUrl).status, 'ok');
});

test('resolveModelCard: a malformed card is incompatible-model naming the field', () => {
  const message = expectFailed(resolveModelCard(without(makeCard(), 'compatibilityKey'), resolveUrl), 'incompatible-model');
  assert.equal(message, 'card: compatibilityKey missing or malformed');
});

test('resolveModelCard: a matching card builds the KomugiModel from the onnx asset', () => {
  const result = resolveModelCard(makeCard({ planeMapVersion: 4 }), resolveUrl);
  assert.equal(result.status, 'ok');
  if (result.status !== 'ok') return;
  assert.deepEqual(result.model, { url: `https://example.invalid/release/${ID}.onnx`, sha256: ONNX_SHA, compatibility: { planeMapVersion: 4 } });
  assert.notEqual(result.model.sha256, PT_SHA);
  assert.equal(result.card.id, ID);
});

// ---------------------------------------------------------------------------
// resolveModelById (acceptance 5).

test('by id: matching id/index/card resolves ok', () => {
  const result = resolveModelById(ID, makeIndex(), makeCard(), resolveUrl);
  assert.equal(result.status, 'ok');
  assert.ok(result.status === 'ok' && result.model.sha256 === ONNX_SHA);
});

test('by id: unknown id is model-not-found', () => {
  expectFailed(resolveModelById('komugi-c6-s1-00000000', makeIndex(), makeCard(), resolveUrl), 'model-not-found');
  // Prototype keys are not index entries.
  expectFailed(resolveModelById('toString', makeIndex(), makeCard(), resolveUrl), 'model-not-found');
});

test('by id: malformed index is model-not-found', () => {
  for (const index of [null, {}, { schemaVersion: 2, recommended: {}, models: {} }, { schemaVersion: 1, recommended: {}, models: { [ID]: { card: '/a', onnxSha256: ONNX_SHA } } }]) {
    expectFailed(resolveModelById(ID, index, makeCard(), resolveUrl), 'model-not-found');
  }
});

test('by id: a card whose id disagrees with the requested id is incompatible-model', () => {
  const otherId = `komugi-c7-s1-${ONNX_SHA.slice(0, 8)}`;
  const index = makeIndex();
  (index.models as Record<string, unknown>)[otherId] = { card: 'x.card.json', onnxSha256: ONNX_SHA };
  const message = expectFailed(resolveModelById(otherId, index, makeCard(), resolveUrl), 'incompatible-model');
  assert.match(message, /^id: /);
});

test('by id: a stale index sha is incompatible-model', () => {
  const index = makeIndex();
  (index.models as Record<string, { card: string; onnxSha256: string }>)[ID] = { card: `${ID}.card.json`, onnxSha256: 'd'.repeat(64) };
  const message = expectFailed(resolveModelById(ID, index, makeCard(), resolveUrl), 'incompatible-model');
  assert.match(message, /^onnxSha256: /);
});

test('by id: an id whose sha8 is not the onnx sha prefix is incompatible-model', () => {
  const badId = 'komugi-c6-s580800-00000000';
  const card = { ...makeCard(), id: badId };
  const message = expectFailed(resolveModelById(badId, makeIndex(card), card, resolveUrl), 'incompatible-model');
  assert.match(message, /sha8/);
});

test('by id: a card missing compatibilityKey is incompatible-model (guard runs first)', () => {
  const message = expectFailed(resolveModelById(ID, makeIndex(), without(makeCard(), 'compatibilityKey'), resolveUrl), 'incompatible-model');
  assert.equal(message, 'card: compatibilityKey missing or malformed');
  // Guard before index: a bad card with a bad index is still incompatible-model.
  expectFailed(resolveModelById(ID, null, without(makeCard(), 'compatibilityKey'), resolveUrl), 'incompatible-model');
});

// ---------------------------------------------------------------------------
// End to end through the real worker host (acceptance 4).

async function initThroughHost(model: { url: string; sha256: string }, bytes: Uint8Array): Promise<{ reply: Posted | undefined; fetched: string[] }> {
  const fetched: string[] = [];
  const fetchOk = fakeFetchOk(bytes);
  const options: WorkerHostOptions = {
    loadOrt: async () => makeFakeOrtModule(),
    fetch: async (url: string) => {
      fetched.push(url);
      return fetchOk!(url);
    },
  };
  const { scope, posted, send } = makeFakeScope();
  installWorkerHost(scope, options);
  const init: WorkerInitRequest = { id: 'init-1', type: 'init', modelUrl: model.url, modelSha256: model.sha256, ortAssets: ORT_ASSETS_WASM_ONLY };
  send(init);
  await flushUntil(() => posted.length >= 1);
  assert.equal(posted.length, 1);
  return { reply: posted[0], fetched };
}

test('e2e: resolveModelCard(card).model initializes the real makeInitEnv on wasm', async () => {
  const result = resolveModelCard(makeCard(), resolveUrl);
  assert.equal(result.status, 'ok');
  if (result.status !== 'ok') return;
  const { reply, fetched } = await initThroughHost(result.model, ONNX_BYTES);
  assert.ok(isOkInitReply(reply), `init should succeed, got ${JSON.stringify(reply)}`);
  assert.equal(reply.backend, 'wasm');
  assert.deepEqual(fetched, [`https://example.invalid/release/${ID}.onnx`]);
});

test('e2e: a model carrying the pt sha (mutant) is red at the real host', async () => {
  const result = resolveModelCard(makeCard(), resolveUrl);
  assert.equal(result.status, 'ok');
  if (result.status !== 'ok') return;
  const { reply } = await initThroughHost({ ...result.model, sha256: PT_SHA }, ONNX_BYTES);
  assert.ok(reply !== undefined && 'err' in reply && reply.code === 'sha-mismatch', JSON.stringify(reply));
});

test('e2e: fetched bytes whose sha256 differs from the card yield sha-mismatch', async () => {
  const result = resolveModelCard(makeCard(), resolveUrl);
  assert.equal(result.status, 'ok');
  if (result.status !== 'ok') return;
  const { reply } = await initThroughHost(result.model, new Uint8Array([1, 2, 3]));
  assert.ok(reply !== undefined && 'err' in reply, `expected an error reply, got ${JSON.stringify(reply)}`);
  assert.equal(reply !== undefined && 'err' in reply ? reply.code : undefined, 'sha-mismatch');
});

// ---------------------------------------------------------------------------
// Failure codes + README (acceptance 6), import lines (acceptance 7).

test('CLIENT_ERROR_CODES contains incompatible-model and model-not-found', () => {
  const codes: readonly string[] = CLIENT_ERROR_CODES;
  assert.ok(codes.includes('incompatible-model'));
  assert.ok(codes.includes('model-not-found'));
});

test("README's Failure codes section names every CLIENT_ERROR_CODES member", () => {
  const text = readFileSync(README, 'utf8');
  const start = text.indexOf('\n## Failure codes\n');
  assert.ok(start >= 0, 'README has a Failure codes section');
  const next = text.indexOf('\n## ', start + 1);
  const section = text.slice(start, next < 0 ? undefined : next);
  for (const code of CLIENT_ERROR_CODES) assert.ok(section.includes(`\`${code}\``), `README Failure codes section misses ${code}`);
});

test('model_registry.ts imports only the leaf modules, type-only from client/client_failures', () => {
  const source = readFileSync(REGISTRY_SOURCE, 'utf8');
  const imports = source.split('\n').filter((line) => /^\s*import\b/.test(line));
  assert.ok(imports.length > 0);
  const allowedValue = new Set(['./rule_features.ts', './rules_config.ts', './constants.ts']);
  const typeOnly = new Set(['./client.ts', './client_failures.ts']);
  for (const line of imports) {
    const spec = /from '([^']+)';$/.exec(line)?.[1];
    assert.ok(spec !== undefined, `unparsed import: ${line}`);
    if (typeOnly.has(spec)) assert.match(line, /^import type /, `value import from ${spec}: ${line}`);
    else assert.ok(allowedValue.has(spec), `disallowed import: ${line}`);
  }
  assert.doesNotMatch(source, /\bimport\s*\(/, 'no dynamic import');
});
