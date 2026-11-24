// KOM-1226 unit a (tickets/KOM-1226.md items 1, 2, 3): three host-integration
// gaps found by two sites switching onto @komugi-ai/engine 0.1.0.
//
// AC1: a worker reply's stats.model_sha256 must carry the init-verified model
// sha256, not a constructor-only value that installWorkerHost never supplies.
// AC2: PlayRequest.rules must accept a RulesConfig object with no cast --
// checked here with a real strict-tsc spawn (tsconfig.strict.json's own
// `include` is src/**/*.ts only, so test/ is never covered by the main
// typecheck run -- same reason test/pack/levels_type_discrimination.test.ts
// spawns its own compiler instance instead of relying on it).
// AC3: every InferenceSession.create must carry an explicit logSeverityLevel
// (default 3, overridable, validated).

import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { tmpdir } from 'node:os';
import { dirname, join, relative, resolve } from 'node:path';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';

import { ACTION_SIZE, ADVANCED_POSITION } from '../src/constants.ts';
import type { OrtModule, OrtSession } from '../src/ort_core.ts';
import type { WorkerInitReply, WorkerReply, WorkerReplyErr, WorkerReplyOk } from '../src/protocol.ts';
import { installWorkerHost } from '../src/worker_host.ts';
import type { WorkerHostOptions } from '../src/worker_host.ts';

const HERE = dirname(fileURLToPath(import.meta.url));

// ---------------------------------------------------------------------------
// Shared fake worker scope + fake ORT module (same shape as
// worker_host_ort.test.ts's own fakes: a real `makeInitEnv` awaits
// `globalThis.crypto.subtle.digest`, a macrotask, so flushUntil yields
// through a real timer tick rather than polling microtasks only).
// ---------------------------------------------------------------------------

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

type CreateEvent = { type: 'create'; executionProviders: string[]; logSeverityLevel: number | undefined };
type OrtEvent = CreateEvent | { type: 'run' };

// Logs every InferenceSession.create call's own options (AC3) and answers
// `run` with zero-filled rows shaped the way batch_evaluator.ts's
// runOrtBatch reads them, so a real search can complete end to end.
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
      async create(
        _pathOrBytes: string | Uint8Array,
        opts: { executionProviders: string[]; logSeverityLevel?: number },
      ): Promise<OrtSession> {
        log.push({ type: 'create', executionProviders: opts.executionProviders, logSeverityLevel: opts.logSeverityLevel });
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

function isOkInitReply(reply: Posted | undefined): reply is WorkerInitReply & { backend: string } {
  return reply !== undefined && !('err' in reply);
}

// ---------------------------------------------------------------------------
// AC1: stats.model_sha256 carries the init-verified sha.
// ---------------------------------------------------------------------------

test('kom1226 AC1: a reply after a successful init carries the init-verified model sha256, lowercased', async () => {
  const log: OrtEvent[] = [];
  const loadOrt = async (): Promise<OrtModule> => makeFakeOrtModule(log);
  const sha = await sha256Hex(MODEL_BYTES);
  const options: WorkerHostOptions = { loadOrt, fetch: fakeFetchOk(MODEL_BYTES) };
  const { scope, posted, send } = makeFakeScope();
  installWorkerHost(scope, options);

  // Uppercased on the wire -- worker_host.ts's own sha verification already
  // lowercases before comparing; this proves the engine stores/echoes the
  // SAME lowercased form, not whatever case the request happened to use.
  send({ id: 'init-1', type: 'init', modelUrl: 'model.onnx', modelSha256: sha.toUpperCase(), ortAssets: ORT_ASSETS_WASM_ONLY });
  await flushUntil(() => posted.length >= 1);
  assert.ok(isOkInitReply(posted[0]), `init should succeed, got ${JSON.stringify(posted[0])}`);

  send({ id: 'search-1', fen: ADVANCED_POSITION, sims: 4, batch: 1 });
  await flushUntil(() => posted.length >= 2);

  const [, searchReply] = posted;
  assert.ok(searchReply !== undefined && !('err' in searchReply), `search should succeed, got ${JSON.stringify(searchReply)}`);
  if (searchReply === undefined || 'err' in searchReply) return;
  assert.equal((searchReply as WorkerReplyOk).stats.model_sha256, sha.toLowerCase());
});

// ---------------------------------------------------------------------------
// AC3: ORT session options carry an explicit logSeverityLevel.
// ---------------------------------------------------------------------------

test('kom1226 AC3: InferenceSession.create defaults to logSeverityLevel 3 when ortLogSeverityLevel is absent', async () => {
  const log: OrtEvent[] = [];
  const loadOrt = async (): Promise<OrtModule> => makeFakeOrtModule(log);
  const sha = await sha256Hex(MODEL_BYTES);
  const options: WorkerHostOptions = { loadOrt, fetch: fakeFetchOk(MODEL_BYTES) };
  const { scope, posted, send } = makeFakeScope();
  installWorkerHost(scope, options);

  send({ id: 'init-1', type: 'init', modelUrl: 'model.onnx', modelSha256: sha, ortAssets: ORT_ASSETS_WASM_ONLY });
  await flushUntil(() => posted.length >= 1);
  assert.ok(isOkInitReply(posted[0]), `init should succeed, got ${JSON.stringify(posted[0])}`);

  const createEvent = log.find((event): event is CreateEvent => event.type === 'create');
  assert.ok(createEvent, 'a session must have been created');
  assert.equal(createEvent?.logSeverityLevel, 3);
});

test('kom1226 AC3: an explicit ortLogSeverityLevel on the init request is honoured', async () => {
  const log: OrtEvent[] = [];
  const loadOrt = async (): Promise<OrtModule> => makeFakeOrtModule(log);
  const sha = await sha256Hex(MODEL_BYTES);
  const options: WorkerHostOptions = { loadOrt, fetch: fakeFetchOk(MODEL_BYTES) };
  const { scope, posted, send } = makeFakeScope();
  installWorkerHost(scope, options);

  send({
    id: 'init-1',
    type: 'init',
    modelUrl: 'model.onnx',
    modelSha256: sha,
    ortAssets: ORT_ASSETS_WASM_ONLY,
    ortLogSeverityLevel: 1,
  });
  await flushUntil(() => posted.length >= 1);
  assert.ok(isOkInitReply(posted[0]), `init should succeed, got ${JSON.stringify(posted[0])}`);

  const createEvent = log.find((event): event is CreateEvent => event.type === 'create');
  assert.ok(createEvent, 'a session must have been created');
  assert.equal(createEvent?.logSeverityLevel, 1);
});

test('kom1226 AC3: an out-of-range ortLogSeverityLevel is rejected with the typed init-failed error', async () => {
  const log: OrtEvent[] = [];
  let loadOrtCalls = 0;
  const loadOrt = async (): Promise<OrtModule> => {
    loadOrtCalls++;
    return makeFakeOrtModule(log);
  };
  const sha = await sha256Hex(MODEL_BYTES);
  const options: WorkerHostOptions = { loadOrt, fetch: fakeFetchOk(MODEL_BYTES) };
  const { scope, posted, send } = makeFakeScope();
  installWorkerHost(scope, options);

  send({
    id: 'init-1',
    type: 'init',
    modelUrl: 'model.onnx',
    modelSha256: sha,
    ortAssets: ORT_ASSETS_WASM_ONLY,
    ortLogSeverityLevel: 7,
  });
  await flushUntil(() => posted.length >= 1);

  const [reply] = posted;
  assert.ok(reply !== undefined && 'err' in reply, `expected an error reply, got ${JSON.stringify(reply)}`);
  if (reply === undefined || !('err' in reply)) return;
  assert.equal((reply as WorkerReplyErr).code, 'init-failed');
  assert.equal(loadOrtCalls, 0, 'loadOrt must never be called on an invalid ortLogSeverityLevel');
  assert.equal(log.filter((event) => event.type === 'create').length, 0, 'no session must ever be created');
});

// ---------------------------------------------------------------------------
// AC2: PlayRequest.rules accepts a RulesConfig object with no cast, checked
// with a real strict-tsc spawn (tsconfig.strict.json's own `include` is
// src/**/*.ts only -- test/ is never covered by the main typecheck run).
// ---------------------------------------------------------------------------

const CLIENT_PATH = resolve(HERE, '..', 'src', 'client.ts');
const RULES_CONFIG_PATH = resolve(HERE, '..', 'src', 'rules_config.ts');

function relSpecifier(fixtureDir: string, targetPath: string): string {
  const rel = relative(fixtureDir, targetPath).replace(/\\/g, '/');
  return rel.startsWith('.') ? rel : `./${rel}`;
}

const FIXTURE_COMPILER_OPTIONS = {
  target: 'ES2022',
  module: 'ESNext',
  lib: ['ES2022', 'DOM', 'DOM.Iterable'],
  moduleResolution: 'bundler',
  rewriteRelativeImportExtensions: true,
  declaration: true,
  strict: true,
  skipLibCheck: true,
  verbatimModuleSyntax: true,
  noUncheckedSideEffectImports: true,
  noEmit: true,
  allowImportingTsExtensions: true,
  exactOptionalPropertyTypes: true,
  noUncheckedIndexedAccess: true,
  noUnusedLocals: true,
  noUnusedParameters: true,
  noFallthroughCasesInSwitch: true,
  noImplicitOverride: true,
  isolatedModules: true,
  types: [] as string[],
};

function playRequestRulesConfigFixtureSource(fixtureDir: string): string {
  return `import type { PlayRequest } from '${relSpecifier(fixtureDir, CLIENT_PATH)}';
import type { RulesConfig } from '${relSpecifier(fixtureDir, RULES_CONFIG_PATH)}';

const customRules: RulesConfig = {
  selfExposureFilter: true,
  archerElbowBlocking: false,
  tacticianBetrayalMode: 'any-subset',
  stalemateResult: 'draw',
  marshalCaptureIsEscape: true,
  plyCap: 512,
  repetitionKey: 'placement',
};

export const request: PlayRequest = { fen: 'fen', sims: 1, rules: customRules };
`;
}

function writeJson(path: string, value: unknown): void {
  writeFileSync(path, JSON.stringify(value, null, 2) + '\n');
}

// KOM_TSC names a tsc entry point; unset, typescript is resolved through
// node_modules from this package (a plain checkout after `npm install
// typescript`). Neither present is a named failure, never a skip.
function resolveTsc(): string {
  const fromEnv = process.env.KOM_TSC;
  if (fromEnv) return fromEnv;
  try {
    return createRequire(import.meta.url).resolve('typescript/bin/tsc');
  } catch {
    throw new Error('KOM_TSC is not set and typescript is not installed under packages/engine/node_modules: set KOM_TSC to a tsc entry point');
  }
}

test('kom1226 AC2: PlayRequest.rules accepts a RulesConfig literal with no cast (strict tsc)', () => {
  const tscPath = resolveTsc();
  const fixtureDir = mkdtempSync(join(tmpdir(), 'komugi-kom1226-playrequest-rules-'));
  try {
    writeFileSync(join(fixtureDir, 'fixture.ts'), playRequestRulesConfigFixtureSource(fixtureDir));
    writeJson(join(fixtureDir, 'tsconfig.fixture.json'), {
      compilerOptions: FIXTURE_COMPILER_OPTIONS,
      include: ['fixture.ts'],
    });

    const run = spawnSync(process.execPath, [tscPath, '-p', join(fixtureDir, 'tsconfig.fixture.json')], {
      cwd: fixtureDir,
      encoding: 'utf8',
    });
    assert.equal(
      run.status,
      0,
      `tsc rejected a PlayRequest literal carrying a RulesConfig object with no cast:\n${run.stdout}${run.stderr}`,
    );
  } finally {
    rmSync(fixtureDir, { recursive: true, force: true });
  }
});
