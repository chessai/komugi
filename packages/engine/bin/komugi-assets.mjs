#!/usr/bin/env node
// komugi-assets: copy the onnxruntime-web glue pairs and one released model
// (card + .onnx + models.json) into a host's public directory, plus an
// ort-assets.json inventory of the ORT files with their sha256s. Offline;
// plain ESM so it runs from a checkout with no build step. Node-only tooling,
// not a public entry point.

import * as nodeFs from 'node:fs';
import { createHash } from 'node:crypto';
import { createRequire } from 'node:module';
import { basename, dirname, join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { parseArgs } from 'node:util';

const USAGE = [
  'usage: npx @komugi-ai/engine assets <public-dir> --release-dir <dir> [--model <id>] [--webgpu]',
  '       npx @komugi-ai/engine assets <public-dir> --card <path> [--index <path>] [--webgpu]',
].join('\n');

// The Emscripten glue pairs `ortAssets.<kind>` feeds to env.wasm.wasmPaths.
// The single source for these names; nothing is scraped from bundles.
export const ORT_GLUE = {
  wasm: { mjs: 'ort-wasm-simd-threaded.mjs', wasm: 'ort-wasm-simd-threaded.wasm' },
  webgpu: { mjs: 'ort-wasm-simd-threaded.asyncify.mjs', wasm: 'ort-wasm-simd-threaded.asyncify.wasm' },
};

const BASENAME = /^[A-Za-z0-9._-]+$/;
const VERSION = /^[0-9A-Za-z.+-]+$/;
const ORT_ASSETS = 'ort-assets.json';
const MODELS = 'models.json';

class CliError extends Error {
  constructor(message, exitCode = 1) {
    super(message);
    this.exitCode = exitCode;
  }
}

function checkBasename(field, name) {
  if (typeof name !== 'string' || !BASENAME.test(name) || name === '.' || name === '..') {
    throw new CliError(`${field}: ${JSON.stringify(name)} is not a plain file name`);
  }
}

function sha256(bytes) {
  return createHash('sha256').update(bytes).digest('hex');
}

function readJson(fs, path, what) {
  let text;
  try {
    text = fs.readFileSync(path, 'utf8');
  } catch (err) {
    throw new CliError(`${what}: cannot read ${path} (${err.code ?? err.message})`);
  }
  try {
    return JSON.parse(text);
  } catch {
    throw new CliError(`${what}: ${path} is not valid JSON`);
  }
}

function readBytes(fs, path, what) {
  try {
    return fs.readFileSync(path);
  } catch (err) {
    throw new CliError(`${what}: cannot read ${path} (${err.code ?? err.message})`);
  }
}

function isRecord(value) {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

async function loadRegistry(fs) {
  const built = new URL('../dist/model_registry.js', import.meta.url);
  const source = new URL('../src/model_registry.ts', import.meta.url);
  return import(fs.existsSync(built) ? built.href : source.href);
}

function parse(argv) {
  let parsed;
  try {
    parsed = parseArgs({
      args: argv,
      allowPositionals: true,
      strict: true,
      options: {
        'release-dir': { type: 'string' },
        model: { type: 'string' },
        card: { type: 'string' },
        index: { type: 'string' },
        webgpu: { type: 'boolean', default: false },
        help: { type: 'boolean', short: 'h', default: false },
      },
    });
  } catch (err) {
    throw new CliError(err.message, 2);
  }
  const { values, positionals } = parsed;
  if (values.help || positionals.length === 0) throw new CliError('', 2);
  if (positionals[0] !== 'assets') throw new CliError(`unknown command ${JSON.stringify(positionals[0])}`, 2);
  if (positionals.length !== 2) throw new CliError('expected exactly one <public-dir>', 2);
  const releaseDir = values['release-dir'];
  const card = values.card;
  if ((releaseDir === undefined) === (card === undefined)) throw new CliError('give exactly one of --release-dir or --card', 2);
  if (releaseDir !== undefined && values.index !== undefined) throw new CliError('--index goes with --card only', 2);
  if (card !== undefined && values.model !== undefined) throw new CliError('--model goes with --release-dir only', 2);
  return { publicDir: positionals[1], releaseDir, model: values.model, card, index: values.index, webgpu: values.webgpu };
}

// The model side: validated card bytes, .onnx bytes and the output index.
function readModel(fs, registry, args) {
  let id;
  let index;
  let cardPath;
  let outIndex;
  if (args.releaseDir !== undefined) {
    index = readJson(fs, join(args.releaseDir, MODELS), MODELS);
    if (!isRecord(index) || !isRecord(index.models)) throw new CliError(`${MODELS}: models missing or malformed`);
    const ids = Object.keys(index.models);
    if (args.model !== undefined) {
      id = args.model;
    } else if (ids.length === 1) {
      id = ids[0];
    } else {
      throw new CliError(`${MODELS} lists ${ids.length} models; choose one with --model <id> (${ids.join(', ')})`);
    }
    if (!Object.prototype.hasOwnProperty.call(index.models, id)) throw new CliError(`model ${JSON.stringify(id)} is not in ${MODELS}`);
    const entry = index.models[id];
    if (!isRecord(entry)) throw new CliError(`models[${id}]: malformed`);
    checkBasename(`models[${id}].card`, entry.card);
    if (entry.card !== `${id}.card.json`) {
      throw new CliError(`models[${id}].card: ${JSON.stringify(entry.card)} must be ${JSON.stringify(`${id}.card.json`)}`);
    }
    cardPath = join(args.releaseDir, entry.card);
    const recommended = {};
    if (isRecord(index.recommended)) {
      for (const [mode, value] of Object.entries(index.recommended)) if (value === id) recommended[mode] = value;
    }
    outIndex = { ...index, recommended, models: { [id]: entry } };
  } else {
    cardPath = args.card;
  }

  const cardBytes = readBytes(fs, cardPath, 'card');
  let card;
  try {
    card = JSON.parse(cardBytes.toString('utf8'));
  } catch {
    throw new CliError(`card: ${cardPath} is not valid JSON`);
  }
  // Names first: nothing below joins a path from an unchecked name.
  const onnxName = isRecord(card) && isRecord(card.assets) && isRecord(card.assets.onnx) ? card.assets.onnx.path : undefined;
  checkBasename('assets.onnx.path', onnxName);
  if (!registry.isModelCard(card)) {
    const failure = registry.resolveModelCard(card, (a) => a.path);
    throw new CliError(failure.status === 'failed' ? failure.failure.message : 'card: malformed');
  }

  if (args.card !== undefined) {
    id = card.id;
    const cardName = basename(cardPath);
    checkBasename('card file name', cardName);
    if (cardName !== `${id}.card.json`) {
      throw new CliError(`card file name: ${JSON.stringify(cardName)} must be ${JSON.stringify(`${id}.card.json`)}`);
    }
    outIndex = { schemaVersion: 1, recommended: {}, models: { [id]: { card: cardName, onnxSha256: card.assets.onnx.sha256 } } };
    index = args.index !== undefined ? readJson(fs, args.index, 'index') : outIndex;
  }

  const resolved = registry.resolveModelById(id, index, card, (a) => a.path);
  if (resolved.status !== 'ok') throw new CliError(resolved.failure.message);
  const incompatible = registry.checkCompatibility(card);
  if (incompatible !== null) throw new CliError(incompatible.message);
  if ([`${id}.card.json`, MODELS, ORT_ASSETS].includes(onnxName)) {
    throw new CliError(`assets.onnx.path: ${JSON.stringify(onnxName)} collides with an output file`);
  }

  const onnxPath = join(dirname(cardPath), onnxName);
  const onnxBytes = readBytes(fs, onnxPath, 'assets.onnx');
  const actual = sha256(onnxBytes);
  if (actual !== card.assets.onnx.sha256) {
    throw new CliError(`assets.onnx.sha256: ${onnxPath} hashes to ${actual}, card says ${card.assets.onnx.sha256}`);
  }
  return { id, cardBytes, onnxName, onnxBytes, outIndex };
}

// The ORT side, resolved from the engine's own location (never cwd).
function readOrt(fs, resolveFrom, webgpu) {
  const req = createRequire(resolveFrom);
  let entry;
  try {
    entry = req.resolve('onnxruntime-web/wasm');
  } catch (err) {
    if (err.code === 'MODULE_NOT_FOUND' || err.code === 'ERR_PACKAGE_PATH_NOT_EXPORTED') {
      throw new CliError(`onnxruntime-web not found (resolving from ${resolveFrom}); install the onnxruntime-web peer dependency`);
    }
    throw err;
  }
  let root;
  let pkg;
  for (let dir = dirname(entry); ; dir = dirname(dir)) {
    const candidate = join(dir, 'package.json');
    if (fs.existsSync(candidate)) {
      const parsed = readJson(fs, candidate, 'onnxruntime-web package.json');
      if (isRecord(parsed) && parsed.name === 'onnxruntime-web') {
        root = dir;
        pkg = parsed;
        break;
      }
    }
    if (dirname(dir) === dir) throw new CliError(`onnxruntime-web not found (no package root above ${entry})`);
  }
  const version = pkg.version;
  if (typeof version !== 'string' || !VERSION.test(version)) {
    throw new CliError(`onnxruntime-web layout not recognised: version ${JSON.stringify(version)} in ${join(root, 'package.json')}`);
  }
  const kinds = webgpu ? ['wasm', 'webgpu'] : ['wasm'];
  const files = [];
  for (const kind of kinds) {
    for (const role of ['mjs', 'wasm']) {
      const name = ORT_GLUE[kind][role];
      const path = join(root, 'dist', name);
      if (!fs.existsSync(path)) throw new CliError(`onnxruntime-web layout not recognised: expected ${path}`);
      files.push({ kind, role, name, bytes: readBytes(fs, path, 'onnxruntime-web') });
    }
  }
  return { version, files };
}

function stageAndMove(fs, publicDir, model, ort, out) {
  const ortDir = `ort-${ort.version}`;
  const manifest = { schemaVersion: 1, generatedAt: new Date().toISOString(), ortVersion: ort.version };
  // [stage-relative path, bytes] in the fixed move order; ort-assets.json last.
  const plan = [];
  for (const f of ort.files) {
    const rel = `${ortDir}/${f.name}`;
    plan.push([rel, f.bytes]);
    manifest[f.kind] ??= {};
    manifest[f.kind][f.role] = { path: rel, sha256: sha256(f.bytes), bytes: f.bytes.length };
  }
  plan.push([model.onnxName, model.onnxBytes]);
  plan.push([`${model.id}.card.json`, model.cardBytes]);
  plan.push([MODELS, Buffer.from(`${JSON.stringify(model.outIndex, null, 2)}\n`)]);
  plan.push([ORT_ASSETS, Buffer.from(`${JSON.stringify(manifest, null, 2)}\n`)]);

  fs.mkdirSync(publicDir, { recursive: true });
  const stage = fs.mkdtempSync(join(publicDir, '.komugi-assets-'));
  const placed = [];
  try {
    fs.mkdirSync(join(stage, ortDir));
    for (const [rel, bytes] of plan) fs.writeFileSync(join(stage, rel), bytes);
    fs.mkdirSync(join(publicDir, ortDir), { recursive: true });
    for (const [rel] of plan) {
      try {
        fs.renameSync(join(stage, rel), join(publicDir, rel));
      } catch (err) {
        const unconfirmed = plan.map(([r]) => r).filter((r) => !placed.includes(r));
        throw new CliError(
          `moving ${rel} into ${publicDir} failed (${err.code ?? err.message}); not placed: ${unconfirmed.join(', ')}; ` +
            `the previous ${ORT_ASSETS} (if any) is unchanged; re-run to recover`,
        );
      }
      placed.push(rel);
    }
  } finally {
    fs.rmSync(stage, { recursive: true, force: true });
  }
  for (const rel of placed) out(`wrote ${join(publicDir, rel)}`);
}

// main(argv, opts?) -> exit code. opts: resolveFrom (test-only peer lookup
// origin), fs (node:fs seam), stdout/stderr (line writers).
export async function main(argv, opts = {}) {
  const fs = opts.fs ?? nodeFs;
  const out = opts.stdout ?? ((line) => process.stdout.write(`${line}\n`));
  const err = opts.stderr ?? ((line) => process.stderr.write(`${line}\n`));
  try {
    const args = parse(argv);
    const registry = await loadRegistry(fs);
    const model = readModel(fs, registry, args);
    const ort = readOrt(fs, opts.resolveFrom ?? import.meta.url, args.webgpu);
    stageAndMove(fs, resolve(args.publicDir), model, ort, out);
    return 0;
  } catch (e) {
    if (!(e instanceof CliError)) {
      err(`komugi-assets: ${e instanceof Error ? e.message : String(e)}`);
      return 1;
    }
    if (e.exitCode === 2) {
      if (e.message) err(`komugi-assets: ${e.message}`);
      err(USAGE);
      return 2;
    }
    err(`komugi-assets: ${e.message}`);
    return 1;
  }
}

function isMain() {
  if (!process.argv[1]) return false;
  try {
    return import.meta.url === pathToFileURL(nodeFs.realpathSync(process.argv[1])).href;
  } catch {
    return false;
  }
}
if (isMain()) process.exitCode = await main(process.argv.slice(2));
