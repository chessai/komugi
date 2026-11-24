// Round-trip: the assets CLI's output, served under a fake origin, loads
// through loadModel({ manifestUrl }) unchanged, and the ort-assets.json
// paths resolve against the URL it was served from. Fakes only: a
// StreamFetchLike over the output dir, no ORT session, no network.
import assert from 'node:assert/strict';
import { createHash, randomBytes } from 'node:crypto';
import { existsSync, mkdirSync, mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, normalize } from 'node:path';
import { test } from 'node:test';
import { main } from '../bin/komugi-assets.mjs';
import { ACTION_ENCODING_VERSION } from '../src/constants.ts';
import { loadModel } from '../src/index.ts';
import type { StreamFetchLike } from '../src/index.ts';

const ORIGIN = 'http://assets.test/';
const sha256 = (b: Uint8Array): string => createHash('sha256').update(b).digest('hex');
const tmp = (): string => mkdtempSync(join(tmpdir(), 'komugi-assets-loadmodel-'));

// A minimal installed onnxruntime-web: the './wasm' export the CLI resolves
// plus the wasm glue pair it copies. Returns a resolveFrom anchor beside it.
function ortLayout(): string {
  const root = tmp();
  const ort = join(root, 'node_modules', 'onnxruntime-web');
  mkdirSync(join(ort, 'dist'), { recursive: true });
  const pkg = { name: 'onnxruntime-web', version: '1.31.1', exports: { './wasm': { require: './dist/ort.wasm.min.js' } } };
  writeFileSync(join(ort, 'package.json'), JSON.stringify(pkg));
  writeFileSync(join(ort, 'dist', 'ort.wasm.min.js'), 'module.exports = {};\n');
  writeFileSync(join(ort, 'dist', 'ort-wasm-simd-threaded.mjs'), 'export default function stubGlue() {}\n');
  writeFileSync(join(ort, 'dist', 'ort-wasm-simd-threaded.wasm'), 'stub wasm binary\n');
  return join(root, 'anchor.mjs');
}

// A one-model release dir: models.json + <id>.card.json + <id>.onnx.
function release(): { dir: string; id: string; onnx: Buffer } {
  const dir = tmp();
  const onnx = randomBytes(512);
  const sha = sha256(onnx);
  const id = `komugi-rt-s300-${sha.slice(0, 8)}`;
  const card = {
    id,
    campaign: 'rt',
    iteration: 1,
    step: 300,
    compatibilityKey: { planeMapVersion: 4, actionEncodingVersion: ACTION_ENCODING_VERSION, rulesProfile: 'komugi', architectureTag: 'b10c128' },
    modesTrained: ['advanced'],
    assets: { pt: { path: `${id}.pt`, sha256: 'a'.repeat(64) }, onnx: { path: `${id}.onnx`, sha256: sha, bytes: onnx.length } },
    sourceCheckpoint: { name: 'checkpoint_1.pt', sha256: 'b'.repeat(64) },
    exportOpset: 17,
    exportParity: { n: 8, maxAbsDelta: 0, atol: 1e-4, rtol: 1e-3 },
    evalPointers: [],
    license: 'MIT, Copyright (c) 2026 chessai',
    createdAt: '2026-11-22T00:00:00Z',
  };
  writeFileSync(join(dir, `${id}.onnx`), onnx);
  writeFileSync(join(dir, `${id}.card.json`), `${JSON.stringify(card, null, 2)}\n`);
  const index = { schemaVersion: 1, recommended: { advanced: id }, models: { [id]: { card: `${id}.card.json`, onnxSha256: sha } } };
  writeFileSync(join(dir, 'models.json'), `${JSON.stringify(index, null, 2)}\n`);
  return { dir, id, onnx };
}

async function runCli(): Promise<{ pub: string; id: string; onnx: Buffer }> {
  const { dir, id, onnx } = release();
  const pub = join(tmp(), 'public');
  const err: string[] = [];
  const code = await main(['assets', pub, '--release-dir', dir], { stdout: () => {}, stderr: (l: string) => err.push(l), resolveFrom: ortLayout() });
  assert.equal(code, 0, err.join('\n'));
  return { pub, id, onnx };
}

// Serves <pub> under `base` (default ORIGIN); records every URL requested.
function serve(pub: string, base: string = ORIGIN): { fetch: StreamFetchLike; requested: string[] } {
  const requested: string[] = [];
  const fetch: StreamFetchLike = async (url: string) => {
    requested.push(url);
    if (!url.startsWith(base)) return new Response('not found', { status: 404 });
    const rel = normalize(decodeURIComponent(url.slice(base.length)));
    const file = join(pub, rel);
    if (rel.startsWith('..') || rel.startsWith('/') || !existsSync(file)) return new Response('not found', { status: 404 });
    const bytes = readFileSync(file);
    return new Response(bytes, { status: 200, headers: { 'content-length': String(bytes.length) } });
  };
  return { fetch, requested };
}

test('loadModel({ manifestUrl: <id>.card.json }) with baseUrl loads the CLI output: status ok', async () => {
  const { pub, id, onnx } = await runCli();
  const { fetch, requested } = serve(pub);
  const result = await loadModel({ manifestUrl: `${id}.card.json` }, { baseUrl: ORIGIN, fetch, cache: false });
  assert.equal(result.status, 'ok', result.status === 'failed' ? `${result.failure.code}: ${result.failure.message}` : '');
  if (result.status !== 'ok') return;
  assert.equal(result.card.id, id);
  assert.equal(result.model.sha256.toLowerCase(), sha256(onnx));
  assert.deepEqual(requested, [`${ORIGIN}${id}.card.json`, `${ORIGIN}${id}.onnx`]);
  result.model.release?.();
});

test('loadModel with a relative manifestUrl and no baseUrl fails model-fetch', async () => {
  const { pub, id } = await runCli();
  const { fetch, requested } = serve(pub);
  assert.equal((globalThis as { location?: unknown }).location, undefined);
  const result = await loadModel({ manifestUrl: `${id}.card.json` }, { fetch, cache: false });
  assert.equal(result.status, 'failed');
  if (result.status !== 'failed') return;
  assert.equal(result.failure.code, 'model-fetch');
  assert.deepEqual(requested, []);
});

test('ort-assets.json paths and the card URL resolve against the URL ort-assets.json was served from', async () => {
  const { pub, id } = await runCli();
  // Served from a nested mount: every path must resolve under it, not the
  // origin root (the fake 404s anything outside the mount).
  const mount = `${ORIGIN}engine/`;
  const { fetch } = serve(pub, mount);
  const ortAssetsUrl = new URL('/engine/ort-assets.json', ORIGIN).href;
  const response = await fetch(ortAssetsUrl);
  assert.equal(response.status, 200);
  const ort = JSON.parse(await response.text()) as { wasm: Record<'mjs' | 'wasm', { path: string; sha256: string; bytes: number }> };
  const abs = (p: string) => new URL(p, ortAssetsUrl).href;
  for (const role of ['mjs', 'wasm'] as const) {
    const entry = ort.wasm[role];
    const url = abs(entry.path);
    assert.ok(url.startsWith(mount), url);
    const got = await fetch(url);
    assert.equal(got.status, 200, url);
    const bytes = new Uint8Array(await got.arrayBuffer());
    assert.equal(sha256(bytes), entry.sha256);
    assert.equal(bytes.length, entry.bytes);
  }
  const cardUrl = abs(`${id}.card.json`);
  assert.equal(cardUrl, `${mount}${id}.card.json`);
  const result = await loadModel({ manifestUrl: cardUrl }, { fetch, cache: false });
  assert.equal(result.status, 'ok');
  if (result.status === 'ok') result.model.release?.();
});
