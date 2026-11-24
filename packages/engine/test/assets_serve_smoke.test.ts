// KOM-1201 unit B (design docs/design/kom1201-asset-kit.md section 6 AC3):
// a static-server smoke test over a CLI-produced public dir. Runs the real
// assets CLI (bin/komugi-assets.mjs) into a fresh temp public dir, serves
// it over a real node:http server on an ephemeral loopback port with the
// documented headers, resolves every ort-assets.json path and the model
// files the way the guide tells a host to (N1), and checks every response
// against the guide's own parsed tables (N8) rather than against the
// DOCUMENTED_HEADERS constant directly, so a drift between the two fails
// here. No network beyond 127.0.0.1, no browser, fakes only.

import assert from 'node:assert/strict';
import { createHash, randomBytes } from 'node:crypto';
import { createServer } from 'node:http';
import type { Server } from 'node:http';
import { existsSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, extname, join, resolve } from 'node:path';
import { after, before, test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { main } from '../bin/komugi-assets.mjs';
import { ACTION_ENCODING_VERSION } from '../src/constants.ts';
import { DOCUMENTED_HEADERS, parseGuideHeaders } from './support/documented_headers.ts';

const HERE = dirname(fileURLToPath(import.meta.url));
const ENGINE_DIR = resolve(HERE, '..');
const GUIDE_PATH = join(ENGINE_DIR, 'docs', 'integration.md');

const sha256 = (b: Uint8Array): string => createHash('sha256').update(b).digest('hex');

function tmp(): string {
  return mkdtempSync(join(tmpdir(), 'komugi-assets-smoke-'));
}

// A minimal onnxruntime-web 1.31.1 install: just enough for `req.resolve
// ('onnxruntime-web/wasm')` to find it and for the CLI's fixed glue table
// to resolve under its dist/.
function makeOrtInstall(): string {
  const root = tmp();
  const ortDir = join(root, 'node_modules', 'onnxruntime-web');
  mkdirSync(join(ortDir, 'dist'), { recursive: true });
  writeFileSync(
    join(ortDir, 'package.json'),
    `${JSON.stringify({ name: 'onnxruntime-web', version: '1.31.1', exports: { './wasm': './dist/ort.wasm.min.js' } }, null, 2)}\n`,
  );
  writeFileSync(join(ortDir, 'dist', 'ort.wasm.min.js'), "module.exports = { name: 'ort.wasm' };\n");
  writeFileSync(join(ortDir, 'dist', 'ort-wasm-simd-threaded.mjs'), "export default function stubGlue() { return 'ort-wasm-simd-threaded.wasm'; }\n");
  writeFileSync(join(ortDir, 'dist', 'ort-wasm-simd-threaded.wasm'), 'stub wasm binary\n');
  // Resolution starts from a file under root/node_modules/..., walking up
  // to find the sibling onnxruntime-web; the file itself need not exist.
  return join(root, 'node_modules', '@komugi-ai', 'engine', 'bin', 'komugi-assets.mjs');
}

interface Model {
  id: string;
  onnx: Buffer;
}

// A release dir: models.json + <id>.card.json + <id>.onnx.
function makeRelease(): { dir: string; model: Model } {
  const dir = tmp();
  const onnx = randomBytes(256);
  const onnxSha = sha256(onnx);
  const id = `komugi-smoke-s300-${onnxSha.slice(0, 8)}`;
  const card = {
    id,
    campaign: 'smoke',
    iteration: 1,
    step: 300,
    compatibilityKey: { planeMapVersion: 4, actionEncodingVersion: ACTION_ENCODING_VERSION, rulesProfile: 'komugi', architectureTag: 'b10c128' },
    modesTrained: ['advanced'],
    assets: { pt: { path: `${id}.pt`, sha256: 'a'.repeat(64) }, onnx: { path: `${id}.onnx`, sha256: onnxSha, bytes: onnx.length } },
    sourceCheckpoint: { name: 'checkpoint_1.pt', sha256: 'b'.repeat(64) },
    exportOpset: 17,
    exportParity: { n: 8, maxAbsDelta: 0, atol: 1e-4, rtol: 1e-3 },
    evalPointers: [],
    license: 'MIT, Copyright (c) 2026 chessai',
    createdAt: '2026-11-22T00:00:00Z',
  };
  const index = {
    schemaVersion: 1,
    recommended: { advanced: id },
    models: { [id]: { card: `${id}.card.json`, onnxSha256: onnxSha } },
  };
  writeFileSync(join(dir, `${id}.onnx`), onnx);
  writeFileSync(join(dir, `${id}.card.json`), `${JSON.stringify(card, null, 2)}\n`);
  writeFileSync(join(dir, 'models.json'), `${JSON.stringify(index, null, 2)}\n`);
  return { dir, model: { id, onnx } };
}

function walkExtensions(dir: string): Set<string> {
  const exts = new Set<string>();
  const entries = readdirSync(dir, { recursive: true, withFileTypes: true });
  for (const entry of entries) {
    if (!entry.isFile()) continue;
    const ext = extname(entry.name);
    if (ext) exts.add(ext);
  }
  return exts;
}

let server: Server;
let baseUrl: string;
let publicDir: string;
let model: Model;

before(async () => {
  const resolveFrom = makeOrtInstall();
  const release = makeRelease();
  model = release.model;
  publicDir = join(tmp(), 'public');
  const errLines: string[] = [];
  const code = await main(['assets', publicDir, '--release-dir', release.dir], { resolveFrom, stdout: () => {}, stderr: (l: string) => errLines.push(l) });
  assert.equal(code, 0, `CLI failed: ${errLines.join('\n')}`);

  server = createServer((req, res) => {
    const reqPath = (req.url ?? '/').replace(/^\/+/, '');
    const filePath = join(publicDir, decodeURIComponent(reqPath));
    const ext = extname(filePath) as keyof typeof DOCUMENTED_HEADERS.extensions;
    const documented = DOCUMENTED_HEADERS.extensions[ext];
    if (!documented) {
      res.writeHead(500).end('no documented Content-Type for this extension');
      return;
    }
    if (!existsSync(filePath)) {
      res.writeHead(404).end('not found');
      return;
    }
    res.writeHead(200, { 'Content-Type': documented.contentType, 'Cache-Control': documented.cacheControl });
    res.end(readFileSync(filePath));
  });
  await new Promise<void>((done) => server.listen(0, '127.0.0.1', done));
  const addr = server.address();
  if (addr === null || typeof addr === 'string') throw new Error('server did not bind to a port');
  baseUrl = `http://127.0.0.1:${addr.port}/`;
});

after(async () => {
  await new Promise<void>((done) => server.close(() => done()));
});

test('every file the CLI wrote is exactly one of .mjs, .wasm, .onnx, .json', () => {
  const exts = walkExtensions(publicDir);
  assert.deepEqual([...exts].sort(), ['.json', '.mjs', '.onnx', '.wasm']);
});

test('resolving ort-assets.json paths the way the guide tells a host to (N1) reaches real files', async () => {
  const ortAssetsUrl = new URL('ort-assets.json', baseUrl).href;
  const ortRes = await fetch(ortAssetsUrl);
  assert.equal(ortRes.status, 200);
  const ort = (await ortRes.json()) as { wasm: { mjs: { path: string }; wasm: { path: string } }; webgpu?: { mjs: { path: string }; wasm: { path: string } } };
  const abs = (p: string) => new URL(p, ortAssetsUrl).href;

  const resolvedUrls = [abs(ort.wasm.mjs.path), abs(ort.wasm.wasm.path), abs(`${model.id}.card.json`), abs(`${model.id}.onnx`), abs('models.json')];
  if (ort.webgpu) resolvedUrls.push(abs(ort.webgpu.mjs.path), abs(ort.webgpu.wasm.path));

  for (const url of resolvedUrls) {
    assert.ok(/^https?:\/\//.test(url), `resolved URL is not absolute: ${url}`);
    const res = await fetch(url);
    assert.equal(res.status, 200, `GET ${url}`);
  }
});

test("every response's Content-Type/Cache-Control matches the guide's own parsed table, not the constant (N8)", async () => {
  const guideHeaders = parseGuideHeaders(readFileSync(GUIDE_PATH, 'utf8'));
  // A drift between DOCUMENTED_HEADERS (what the server actually serves)
  // and the guide's documented values would fail here first.
  assert.deepEqual(guideHeaders, DOCUMENTED_HEADERS);

  const ortAssetsUrl = new URL('ort-assets.json', baseUrl).href;
  const ort = (await (await fetch(ortAssetsUrl)).json()) as { wasm: { mjs: { path: string }; wasm: { path: string } } };
  const abs = (p: string) => new URL(p, ortAssetsUrl).href;

  const cases: [string, keyof typeof guideHeaders.extensions][] = [
    [abs(ort.wasm.mjs.path), '.mjs'],
    [abs(ort.wasm.wasm.path), '.wasm'],
    [abs(`${model.id}.onnx`), '.onnx'],
    [abs('models.json'), '.json'],
    [abs(`${model.id}.card.json`), '.json'],
    [ortAssetsUrl, '.json'],
  ];
  for (const [url, ext] of cases) {
    const res = await fetch(url);
    const expected = guideHeaders.extensions[ext];
    assert.equal(res.headers.get('content-type')?.split(';')[0]?.trim(), expected.contentType, `Content-Type for ${url}`);
    assert.equal(res.headers.get('cache-control'), expected.cacheControl, `Cache-Control for ${url}`);
  }
});

test("the MIME lookup fails closed: an undocumented extension gets 500, never a default type", async () => {
  writeFileSync(join(publicDir, 'x.txt'), 'planted file with an undocumented extension\n');
  const res = await fetch(new URL('x.txt', baseUrl).href);
  assert.equal(res.status, 500);
});
