// The assets CLI (bin/komugi-assets.mjs): copies the ORT glue pairs and one
// released model into a public dir, atomically, resolving onnxruntime-web
// from the engine's own location. Fixtures are hand-built: a release dir
// with a random-bytes .onnx, and installed node_modules layouts built at
// test time from a stub template written by buildStub. Opt-in real install:
// KOM_ORT_WEB_DIR=<onnxruntime-web package dir>.

import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { createHash, randomBytes } from 'node:crypto';
import * as nodeFs from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { main } from '../bin/komugi-assets.mjs';
import { ACTION_ENCODING_VERSION } from '../src/constants.ts';

const { cpSync, existsSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, rmSync, symlinkSync, writeFileSync } = nodeFs;

const HERE = dirname(fileURLToPath(import.meta.url));
const ENGINE_DIR = resolve(HERE, '..');
const BIN = join(ENGINE_DIR, 'bin', 'komugi-assets.mjs');
const STUB_VERSION = '1.31.1';
const WASM_PAIR = ['ort-wasm-simd-threaded.mjs', 'ort-wasm-simd-threaded.wasm'];
const WEBGPU_PAIR = ['ort-wasm-simd-threaded.asyncify.mjs', 'ort-wasm-simd-threaded.asyncify.wasm'];

const sha256 = (b: Uint8Array): string => createHash('sha256').update(b).digest('hex');

function tmp(): string {
  return mkdtempSync(join(tmpdir(), 'komugi-assets-test-'));
}

// The onnxruntime-web 1.31.1 exports map, copied verbatim.
const ortEntry = (name: string): Record<string, unknown> => ({
  types: './types.d.ts',
  import: { 'onnxruntime-web-use-extern-wasm': `./dist/ort.${name}.min.mjs`, default: `./dist/ort.${name}.bundle.min.mjs` },
  require: `./dist/ort.${name}.min.js`,
});
const STUB_EXPORTS: Record<string, unknown> = {
  '.': {
    types: './types.d.ts',
    node: { import: './dist/ort.node.min.mjs', require: './dist/ort.node.min.js' },
    import: { 'onnxruntime-web-use-extern-wasm': './dist/ort.min.mjs', default: './dist/ort.bundle.min.mjs' },
    require: './dist/ort.min.js',
  },
  './all': ortEntry('all'),
  './wasm': ortEntry('wasm'),
  './webgl': { types: './types.d.ts', import: './dist/ort.webgl.min.mjs', require: './dist/ort.webgl.min.js' },
  './webgpu': ortEntry('webgpu'),
  './jspi': ortEntry('jspi'),
  ...Object.fromEntries(
    ['wasm', 'mjs'].flatMap((ext) =>
      ['', '.jsep', '.jspi', '.asyncify'].map((v) => [`./ort-wasm-simd-threaded${v}.${ext}`, `./dist/ort-wasm-simd-threaded${v}.${ext}`]),
    ),
  ),
};

// The stub template (package.json + stub dist files), built once per run.
// The require-condition entries carry decoy tokens a regex-scraping copier
// would pick up.
function buildStub(): string {
  const dir = join(tmp(), 'onnxruntime-web');
  mkdirSync(join(dir, 'dist'), { recursive: true });
  const pkg = {
    name: 'onnxruntime-web',
    description: 'test stub: the onnxruntime-web 1.31.1 exports map, with stub dist files',
    version: STUB_VERSION,
    license: 'MIT',
    main: 'dist/ort.node.min.js',
    types: './types.d.ts',
    exports: STUB_EXPORTS,
  };
  writeFileSync(join(dir, 'package.json'), `${JSON.stringify(pkg, null, 2)}\n`);
  const files: Record<string, string> = {
    'ort.min.js':
      "// stub require-condition entry for onnxruntime-web. Decoy tokens:\n// x.wasm, env.wasm, ort.wasm.\nmodule.exports = { env: { wasm: { wasmPaths: 'x.wasm' } }, name: 'ort' };\n",
    'ort.wasm.min.js':
      '// stub require-condition entry for onnxruntime-web/wasm. Decoy tokens a\n// regex-scraping copier would pick up: x.wasm, env.wasm, ort.wasm,\n// "ort-wasm-decoy.wasm", new URL("decoy.wasm", import.meta.url).\n' +
      "module.exports = { env: { wasm: { wasmPaths: 'x.wasm' } }, name: 'ort.wasm' };\n",
    'ort.webgpu.min.js':
      '// stub require-condition entry for onnxruntime-web/webgpu. Decoy tokens:\n// x.wasm, env.wasm, ort.wasm, "ort-wasm-simd.jsep.wasm".\n' +
      "module.exports = { env: { wasm: { wasmPaths: 'x.wasm' } }, name: 'ort.webgpu' };\n",
    'ort-wasm-simd-threaded.mjs':
      "// stub Emscripten glue (wasm pair)\nexport default function stubGlue() { return 'ort-wasm-simd-threaded.wasm'; }\n",
    'ort-wasm-simd-threaded.wasm': 'stub wasm binary (wasm pair)\n',
    'ort-wasm-simd-threaded.asyncify.mjs':
      "// stub Emscripten glue (asyncify pair)\nexport default function stubGlue() { return 'ort-wasm-simd-threaded.asyncify.wasm'; }\n",
    'ort-wasm-simd-threaded.asyncify.wasm': 'stub wasm binary (asyncify pair)\n',
  };
  for (const [name, body] of Object.entries(files)) writeFileSync(join(dir, 'dist', name), body);
  return dir;
}
const STUB = buildStub();

interface Model {
  id: string;
  onnx: Buffer;
  card: Record<string, unknown>;
}

function makeModel(tag: string): Model {
  const onnx = randomBytes(256);
  const sha = sha256(onnx);
  const id = `komugi-${tag}-s300-${sha.slice(0, 8)}`;
  const card: Record<string, unknown> = {
    id,
    campaign: tag,
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
  return { id, onnx, card };
}

interface ReleaseOpts {
  extra?: boolean;
  recommended?: (ids: string[]) => Record<string, string>;
  mutate?: (m: Model, dir: string, index: { models: Record<string, { card: string; onnxSha256: string }> }) => void;
}

// A release dir: models.json + <id>.card.json + <id>.onnx (one or two models).
function makeRelease(opts: ReleaseOpts = {}): { dir: string; models: Model[] } {
  const dir = tmp();
  const models = [makeModel('alpha')];
  if (opts.extra) models.push(makeModel('beta'));
  const ids = models.map((m) => m.id);
  const index = {
    schemaVersion: 1,
    recommended: opts.recommended ? opts.recommended(ids) : { advanced: ids[0] as string },
    models: Object.fromEntries(models.map((m) => [m.id, { card: `${m.id}.card.json`, onnxSha256: sha256(m.onnx) }])),
  };
  for (const m of models) {
    writeFileSync(join(dir, `${m.id}.onnx`), m.onnx);
    if (opts.mutate && m === models[0]) opts.mutate(m, dir, index);
    writeFileSync(join(dir, `${m.id}.card.json`), `${JSON.stringify(m.card, null, 2)}\n`);
  }
  writeFileSync(join(dir, 'models.json'), `${JSON.stringify(index, null, 2)}\n`);
  return { dir, models };
}

// An installed layout: <root>/node_modules/onnxruntime-web (stub copy or a
// symlink to a real install) and <root>/node_modules/@komugi-ai/engine with
// a COPY of the bin and `src` symlinked to the real src (type stripping
// refuses files whose realpath is under node_modules).
function makeLayout(ort: { version?: string; dropWebgpu?: boolean; linkTo?: string } = {}): { root: string; bin: string } {
  const root = tmp();
  const ortDir = join(root, 'node_modules', 'onnxruntime-web');
  if (ort.linkTo !== undefined) {
    mkdirSync(dirname(ortDir), { recursive: true });
    symlinkSync(ort.linkTo, ortDir, 'dir');
  } else {
    placeStub(ortDir, ort.version, ort.dropWebgpu);
  }
  const engine = join(root, 'node_modules', '@komugi-ai', 'engine');
  mkdirSync(join(engine, 'bin'), { recursive: true });
  cpSync(BIN, join(engine, 'bin', 'komugi-assets.mjs'));
  symlinkSync(join(ENGINE_DIR, 'src'), join(engine, 'src'), 'dir');
  return { root, bin: join(engine, 'bin', 'komugi-assets.mjs') };
}

function placeStub(dest: string, version?: string, dropWebgpu?: boolean): void {
  cpSync(STUB, dest, { recursive: true });
  if (version !== undefined) {
    const pkgPath = join(dest, 'package.json');
    const pkg = JSON.parse(readFileSync(pkgPath, 'utf8')) as { version: string };
    pkg.version = version;
    writeFileSync(pkgPath, JSON.stringify(pkg, null, 2));
  }
  if (dropWebgpu) for (const f of WEBGPU_PAIR) rmSync(join(dest, 'dist', f));
}

function spawnBin(bin: string, args: string[], opts: { cwd?: string; env?: Record<string, string> } = {}) {
  const env = { ...process.env, ...opts.env };
  delete env.NODE_TEST_CONTEXT;
  return spawnSync(process.execPath, [bin, ...args], { cwd: opts.cwd ?? ENGINE_DIR, env, encoding: 'utf8' });
}

async function runIn(args: string[], extra: Record<string, unknown> = {}): Promise<{ code: number; err: string }> {
  const lines: string[] = [];
  const code = await main(args, { stdout: () => {}, stderr: (l: string) => lines.push(l), ...extra });
  return { code, err: lines.join('\n') };
}

function stages(dir: string): string[] {
  return existsSync(dir) ? readdirSync(dir).filter((n) => n.startsWith('.komugi-assets-')) : [];
}

// The full AC1 assertion set over a successful output dir.
function assertOutput(pub: string, model: Model, version: string, webgpu: boolean, srcCard?: string): Record<string, unknown> {
  const ortDir = join(pub, `ort-${version}`);
  for (const f of WASM_PAIR) assert.ok(existsSync(join(ortDir, f)), `missing ${f}`);
  for (const f of WEBGPU_PAIR) assert.equal(existsSync(join(ortDir, f)), webgpu, `${f} presence`);
  assert.deepEqual(readdirSync(ortDir).sort(), (webgpu ? [...WASM_PAIR, ...WEBGPU_PAIR] : WASM_PAIR).slice().sort());
  if (srcCard !== undefined) assert.deepEqual(readFileSync(join(pub, `${model.id}.card.json`)), readFileSync(srcCard));
  assert.deepEqual(readFileSync(join(pub, `${model.id}.onnx`)), model.onnx);
  const index = JSON.parse(readFileSync(join(pub, 'models.json'), 'utf8')) as {
    schemaVersion: number;
    recommended: Record<string, string>;
    models: Record<string, { card: string; onnxSha256: string }>;
  };
  assert.equal(index.schemaVersion, 1);
  assert.deepEqual(Object.keys(index.models), [model.id]);
  assert.equal(index.models[model.id]?.card, `${model.id}.card.json`);
  assert.ok(existsSync(join(pub, index.models[model.id]?.card as string)));
  for (const v of Object.values(index.recommended)) assert.ok(v in index.models, `recommended names unlisted ${v}`);
  const manifest = JSON.parse(readFileSync(join(pub, 'ort-assets.json'), 'utf8')) as Record<string, unknown>;
  assert.equal(manifest.schemaVersion, 1);
  assert.equal(manifest.ortVersion, version);
  assert.equal(typeof manifest.generatedAt, 'string');
  assert.equal('webgpu' in manifest, webgpu);
  const kinds: [string, string[]][] = [['wasm', WASM_PAIR]];
  if (webgpu) kinds.push(['webgpu', WEBGPU_PAIR]);
  for (const [kind, pair] of kinds) {
    const entry = manifest[kind] as Record<string, { path: string; sha256: string; bytes: number }>;
    for (const [role, name] of [['mjs', pair[0]], ['wasm', pair[1]]] as const) {
      const file = entry[role];
      assert.ok(file, `${kind}.${role}`);
      assert.equal(file.path, `ort-${version}/${name}`);
      assert.ok(file.path.includes(version));
      const bytes = readFileSync(join(pub, file.path));
      assert.equal(file.sha256, sha256(bytes));
      assert.equal(file.bytes, bytes.length);
    }
  }
  assert.deepEqual(stages(pub), []);
  return index;
}

test('release dir: copies the wasm pair, card, onnx, models.json and ort-assets.json', async () => {
  const { dir, models } = makeRelease();
  const layout = makeLayout();
  const pub = join(tmp(), 'public');
  const r = await runIn(['assets', pub, '--release-dir', dir], { resolveFrom: layout.bin });
  assert.equal(r.code, 0, r.err);
  const m = models[0] as Model;
  const index = assertOutput(pub, m, STUB_VERSION, false, join(dir, `${m.id}.card.json`));
  assert.deepEqual(index.recommended, { advanced: m.id });
});

test('--webgpu also copies the asyncify pair', async () => {
  const { dir, models } = makeRelease();
  const layout = makeLayout();
  const pub = join(tmp(), 'public');
  const r = await runIn(['assets', pub, '--release-dir', dir, '--webgpu'], { resolveFrom: layout.bin });
  assert.equal(r.code, 0, r.err);
  assertOutput(pub, models[0] as Model, STUB_VERSION, true);
});

test('--model selects one entry; recommended keeps only modes naming it', async () => {
  const { dir, models } = makeRelease({ extra: true, recommended: (ids) => ({ advanced: ids[1] as string, beginner: ids[0] as string }) });
  const layout = makeLayout();
  const pub = join(tmp(), 'public');
  const m = models[0] as Model;
  const r = await runIn(['assets', pub, '--release-dir', dir, '--model', m.id], { resolveFrom: layout.bin });
  assert.equal(r.code, 0, r.err);
  const index = assertOutput(pub, m, STUB_VERSION, false, join(dir, `${m.id}.card.json`));
  assert.deepEqual(index.recommended, { beginner: m.id });
  assert.equal(existsSync(join(pub, `${(models[1] as Model).id}.onnx`)), false);
});

test('--model naming the other entry drops every recommended mode to {}', async () => {
  const { dir, models } = makeRelease({ extra: true, recommended: (ids) => ({ advanced: ids[0] as string }) });
  const layout = makeLayout();
  const pub = join(tmp(), 'public');
  const m = models[1] as Model;
  const r = await runIn(['assets', pub, '--release-dir', dir, '--model', m.id], { resolveFrom: layout.bin });
  assert.equal(r.code, 0, r.err);
  const index = assertOutput(pub, m, STUB_VERSION, false);
  assert.deepEqual(index.recommended, {});
});

test('--card mode synthesizes a one-entry index, with or without --index', async () => {
  const { dir, models } = makeRelease();
  const m = models[0] as Model;
  const layout = makeLayout();
  const cardPath = join(dir, `${m.id}.card.json`);
  for (const extra of [[], ['--index', join(dir, 'models.json')]]) {
    const pub = join(tmp(), 'public');
    const r = await runIn(['assets', pub, '--card', cardPath, ...extra], { resolveFrom: layout.bin });
    assert.equal(r.code, 0, r.err);
    const index = assertOutput(pub, m, STUB_VERSION, false, cardPath);
    assert.deepEqual(index, { schemaVersion: 1, recommended: {}, models: { [m.id]: { card: `${m.id}.card.json`, onnxSha256: sha256(m.onnx) } } });
  }
});

test('--card with a mismatched --index refuses', async () => {
  const { dir, models } = makeRelease();
  const m = models[0] as Model;
  const other = makeRelease();
  const layout = makeLayout();
  const pub = join(tmp(), 'public');
  const r = await runIn(['assets', pub, '--card', join(dir, `${m.id}.card.json`), '--index', join(other.dir, 'models.json')], { resolveFrom: layout.bin });
  assert.notEqual(r.code, 0);
  assert.equal(existsSync(pub), false);
});

async function assertRefusal(args: (pub: string) => string[], resolveFrom: string, pattern: RegExp): Promise<void> {
  const pub = join(tmp(), 'public');
  mkdirSync(pub);
  writeFileSync(join(pub, 'keep.txt'), 'pre-existing');
  const r = await runIn(args(pub), { resolveFrom });
  assert.notEqual(r.code, 0, 'expected a non-zero exit');
  assert.match(r.err, pattern);
  assert.deepEqual(readdirSync(pub), ['keep.txt'], 'refusal left output or a stage behind');
}

test('refuses an .onnx with one byte flipped after its sha256 was recorded', async () => {
  const { dir, models } = makeRelease();
  const m = models[0] as Model;
  const path = join(dir, `${m.id}.onnx`);
  const bytes = readFileSync(path);
  bytes[17] = (bytes[17] as number) ^ 0xff;
  writeFileSync(path, bytes);
  await assertRefusal((pub) => ['assets', pub, '--release-dir', dir], makeLayout().bin, /assets\.onnx\.sha256/);
});

test('refuses a card whose id does not match its index key', async () => {
  const { dir } = makeRelease({
    mutate: (m) => {
      m.card.id = (m.card.id as string).replace('alpha', 'gamma');
    },
  });
  await assertRefusal((pub) => ['assets', pub, '--release-dir', dir], makeLayout().bin, /id/);
});

test('refuses assets.onnx.path "../x.onnx"', async () => {
  const { dir } = makeRelease({
    mutate: (m, d) => {
      (m.card.assets as { onnx: { path: string } }).onnx.path = '../x.onnx';
      writeFileSync(join(dirname(d), 'x.onnx'), m.onnx);
    },
  });
  await assertRefusal((pub) => ['assets', pub, '--release-dir', dir], makeLayout().bin, /assets\.onnx\.path/);
});

test('refuses assets.onnx.path ".."', async () => {
  const { dir } = makeRelease({
    mutate: (m) => {
      (m.card.assets as { onnx: { path: string } }).onnx.path = '..';
    },
  });
  await assertRefusal((pub) => ['assets', pub, '--release-dir', dir], makeLayout().bin, /assets\.onnx\.path/);
});

test('refuses an ambiguous --release-dir with no --model', async () => {
  const { dir } = makeRelease({ extra: true });
  await assertRefusal((pub) => ['assets', pub, '--release-dir', dir], makeLayout().bin, /--model/);
});

test('refuses when no onnxruntime-web is reachable', async () => {
  const { dir } = makeRelease();
  const nowhere = join(tmp(), 'bin', 'komugi-assets.mjs');
  await assertRefusal((pub) => ['assets', pub, '--release-dir', dir], nowhere, /onnxruntime-web not found/);
});

test('refuses --webgpu when the stub lacks the asyncify pair', async () => {
  const { dir } = makeRelease();
  const layout = makeLayout({ dropWebgpu: true });
  await assertRefusal((pub) => ['assets', pub, '--release-dir', dir, '--webgpu'], layout.bin, /onnxruntime-web layout not recognised.*asyncify/);
});

test('refuses a models[id].card other than <id>.card.json', async () => {
  const { dir } = makeRelease({
    mutate: (m, d, index) => {
      const entry = index.models[m.id] as { card: string };
      entry.card = 'renamed.card.json';
      writeFileSync(join(d, 'renamed.card.json'), JSON.stringify(m.card));
    },
  });
  await assertRefusal((pub) => ['assets', pub, '--release-dir', dir], makeLayout().bin, /card/);
});

test('refuses a --card file not named <id>.card.json', async () => {
  const { dir, models } = makeRelease();
  const m = models[0] as Model;
  const renamed = join(dir, 'other.card.json');
  cpSync(join(dir, `${m.id}.card.json`), renamed);
  await assertRefusal((pub) => ['assets', pub, '--card', renamed], makeLayout().bin, /card file name/);
});

test('peer location: the spawned bin resolves ORT from its own location, not cwd', () => {
  const { dir, models } = makeRelease();
  const layout = makeLayout();
  const app = join(layout.root, 'app');
  placeStub(join(app, 'node_modules', 'onnxruntime-web'), '1.32.0');
  const pub = join(tmp(), 'public');
  const r = spawnBin(layout.bin, ['assets', pub, '--release-dir', dir], { cwd: app });
  assert.equal(r.status, 0, r.stderr);
  assertOutput(pub, models[0] as Model, '1.31.1', false);
  assert.equal(existsSync(join(pub, 'ort-1.32.0')), false);
});

test('stage location: TMPDIR=/nonexistent still writes the full output', () => {
  const { dir, models } = makeRelease();
  const layout = makeLayout();
  const pub = join(tmp(), 'public');
  const r = spawnBin(layout.bin, ['assets', pub, '--release-dir', dir, '--webgpu'], { env: { TMPDIR: '/nonexistent' } });
  assert.equal(r.status, 0, r.stderr);
  assertOutput(pub, models[0] as Model, STUB_VERSION, true);
});

test('atomic move: a failure on the Nth rename leaves no new ort-assets.json and no stage', async () => {
  const { dir } = makeRelease();
  const layout = makeLayout();
  // 2 ORT files + onnx + card + models.json + ort-assets.json = 6 renames.
  for (let n = 1; n <= 6; n++) {
    const pub = join(tmp(), 'public');
    let calls = 0;
    const fs = {
      ...nodeFs,
      renameSync: (from: string, to: string) => {
        calls += 1;
        if (calls === n) throw Object.assign(new Error('injected rename failure'), { code: 'EIO' });
        nodeFs.renameSync(from, to);
      },
    };
    const r = await runIn(['assets', pub, '--release-dir', dir], { resolveFrom: layout.bin, fs });
    assert.equal(r.code, 1, `rename ${n}`);
    assert.match(r.err, /not placed: .*ort-assets\.json/);
    assert.equal(existsSync(join(pub, 'ort-assets.json')), false, `rename ${n}: ort-assets.json placed`);
    assert.deepEqual(stages(pub), [], `rename ${n}: stage left behind`);
  }
});

test('atomic move: a re-run after a failed move succeeds', async () => {
  const { dir, models } = makeRelease();
  const layout = makeLayout();
  const pub = join(tmp(), 'public');
  const fs = {
    ...nodeFs,
    renameSync: (from: string, to: string) => {
      if (from.endsWith('models.json')) throw new Error('injected');
      nodeFs.renameSync(from, to);
    },
  };
  assert.equal((await runIn(['assets', pub, '--release-dir', dir], { resolveFrom: layout.bin, fs })).code, 1);
  const r = await runIn(['assets', pub, '--release-dir', dir], { resolveFrom: layout.bin });
  assert.equal(r.code, 0, r.err);
  assertOutput(pub, models[0] as Model, STUB_VERSION, false);
});

test('bin symlink: a .bin symlink with no arguments prints usage and exits 2', () => {
  const binDir = join(tmp(), '.bin');
  mkdirSync(binDir);
  const link = join(binDir, 'engine');
  symlinkSync(BIN, link);
  const r = spawnBin(link, [], { cwd: binDir });
  assert.equal(r.status, 2, r.stderr);
  assert.match(r.stderr, /^usage: npx @komugi-ai\/engine assets/m);
});

test('usage errors exit 2', () => {
  for (const args of [['frobnicate'], ['assets'], ['assets', 'pub'], ['assets', 'pub', '--release-dir', 'a', '--card', 'b'], ['assets', 'pub', '--bogus']]) {
    const r = spawnBin(BIN, args);
    assert.equal(r.status, 2, `${args.join(' ')}: ${r.stderr}`);
    assert.match(r.stderr, /usage:/);
  }
});

test('main guard: importing the bin with no argv[1] is silent and exits 0', () => {
  const env = { ...process.env };
  delete env.NODE_TEST_CONTEXT;
  const r = spawnSync(process.execPath, ['--input-type=module', '-e', "await import('./bin/komugi-assets.mjs')"], { cwd: ENGINE_DIR, env, encoding: 'utf8' });
  assert.equal(r.status, 0, r.stderr);
  assert.equal(r.stdout, '');
  assert.equal(r.stderr, '');
});

test('the shipped bin carries no internal references', () => {
  const text = readFileSync(BIN, 'utf8');
  assert.equal(text.split('\n')[0], '#!/usr/bin/env node');
  for (const banned of [/KOM-[0-9]+/, /tickets\//, /docs\/reviews/, /\/home\//]) assert.doesNotMatch(text, banned);
});

const REAL_ORT = process.env.KOM_ORT_WEB_DIR;

test(
  'opt-in real install: KOM_ORT_WEB_DIR copies both glue pairs',
  { skip: REAL_ORT === undefined || REAL_ORT === '' ? 'KOM_ORT_WEB_DIR is not set (point it at an installed onnxruntime-web package)' : false },
  () => {
    const ortDir = resolve(REAL_ORT as string);
    const pkgPath = join(ortDir, 'package.json');
    assert.ok(existsSync(pkgPath), `KOM_ORT_WEB_DIR=${ortDir} has no package.json`);
    const pkg = JSON.parse(readFileSync(pkgPath, 'utf8')) as { name?: string; version?: string; exports?: unknown };
    assert.equal(pkg.name, 'onnxruntime-web', `KOM_ORT_WEB_DIR=${ortDir} is not the onnxruntime-web package`);
    if (pkg.version === STUB_VERSION) {
      const stub = JSON.parse(readFileSync(join(STUB, 'package.json'), 'utf8')) as { exports: unknown };
      assert.deepEqual(stub.exports, pkg.exports, 'the stub exports map must copy the real install verbatim');
    }
    const { dir, models } = makeRelease();
    const layout = makeLayout({ linkTo: ortDir });
    const pub = join(tmp(), 'public');
    const r = spawnBin(layout.bin, ['assets', pub, '--release-dir', dir, '--webgpu']);
    assert.equal(r.status, 0, r.stderr);
    const version = pkg.version as string;
    assertOutput(pub, models[0] as Model, version, true);
    for (const f of [...WASM_PAIR, ...WEBGPU_PAIR]) {
      assert.deepEqual(readFileSync(join(pub, `ort-${version}`, f)), readFileSync(join(ortDir, 'dist', f)), f);
    }
  },
);
