// KOM-1228 (tickets/KOM-1228.md D1-D5, AC1-AC5): client-only time-only
// thinking. `ThinkCap` -- sims, an optional time cap (today's form), or
// time only, resolved client-side to the wire message `{ sims:
// TIME_ONLY_NODE_CEILING - 1, maxMs }`. No worker/protocol/engine change:
// AC4 proves that by running a real search through `installWorkerHost`.

import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { tmpdir } from 'node:os';
import { dirname, join, relative, resolve } from 'node:path';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';

import { ADVANCED_POSITION } from '../src/constants.ts';
import { KomugiClient, TIME_ONLY_NODE_CEILING } from '../src/client.ts';
import type { KomugiClientOptions, KomugiWorkerFactory, PlayRequest, PlayResult } from '../src/client.ts';
import type { WorkerReplyOk } from '../src/protocol.ts';
import type { OrtModule, OrtSession } from '../src/ort_core.ts';
import { installWorkerHost } from '../src/worker_host.ts';
import type { WorkerHostOptions } from '../src/worker_host.ts';

const HERE = dirname(fileURLToPath(import.meta.url));

function makeClient(factory: KomugiWorkerFactory, extra: Partial<KomugiClientOptions> = {}): InstanceType<typeof KomugiClient> {
  return new KomugiClient({
    createWorker: factory,
    model: { url: 'model.onnx', sha256: 'sha-a' },
    ortAssets: { wasm: { mjs: 'ort.mjs', wasm: 'ort.wasm' } },
    ...extra,
  });
}

function expectMove(result: PlayResult): WorkerReplyOk {
  assert.equal(result.status, 'move', `expected a move, got ${JSON.stringify(result)}`);
  return (result as { status: 'move'; reply: WorkerReplyOk }).reply;
}

// ---------------------------------------------------------------------------
// A hand-driven fake worker (same shape as client.test.ts's own
// fakeDrivenWorkerFactory): the test delivers every message to the client
// itself, no real worker_threads thread needed. Used for AC1/AC2 (message
// shapes and rejections never reach a worker) and AC4's timeout maths.
// ---------------------------------------------------------------------------

function fakeDrivenWorkerFactory(): {
  factory: KomugiWorkerFactory;
  posted: () => Record<string, unknown>[];
  deliver: (data: unknown) => void;
} {
  let messageListener: ((event: { data: unknown }) => void) | null = null;
  const postedMessages: Record<string, unknown>[] = [];
  const factory: KomugiWorkerFactory = () => ({
    postMessage(message: unknown): void {
      postedMessages.push(message as Record<string, unknown>);
    },
    terminate(): void {
      // not exercised here
    },
    addEventListener(type: 'message' | 'error' | 'messageerror', listener: (event: { data: unknown } | unknown) => void): void {
      if (type === 'message') messageListener = listener as (event: { data: unknown }) => void;
    },
  });
  return {
    factory,
    posted: () => postedMessages,
    deliver: (data: unknown) => messageListener?.({ data }),
  };
}

function initReplyMessage(posted: Record<string, unknown>[]): Record<string, unknown> {
  const init = posted.find((m) => m.type === 'init');
  if (init === undefined) throw new Error('no init message posted yet');
  return { id: init.id, backend: 'wasm', threads: 1, batch: 1, probeRecord: {} };
}

function searchMessage(posted: Record<string, unknown>[]): Record<string, unknown> {
  const search = posted.find((m) => 'fen' in m);
  if (search === undefined) throw new Error('no search message posted yet');
  return search;
}

async function tick(): Promise<void> {
  await new Promise((resolve) => queueMicrotask(resolve));
  await new Promise((resolve) => queueMicrotask(resolve));
}

// ---------------------------------------------------------------------------
// AC1: message shapes -- `{fen, sims}`/`{fen, sims, maxMs}` post exactly
// today's worker message (byte-identical fields and order); `{fen, maxMs}`
// posts `sims: TIME_ONLY_NODE_CEILING - 1` plus that maxMs.
// ---------------------------------------------------------------------------

async function postedSearchFor(request: PlayRequest): Promise<Record<string, unknown>> {
  const { factory, posted, deliver } = fakeDrivenWorkerFactory();
  const client = makeClient(factory);
  try {
    void client.play(request);
    deliver(initReplyMessage(posted()));
    await tick();
    return searchMessage(posted());
  } finally {
    client.dispose();
  }
}

test('AC1: sims only posts exactly {id, fen, sims}', async () => {
  const message = await postedSearchFor({ fen: ADVANCED_POSITION, sims: 4 });
  assert.deepEqual(Object.keys(message), ['id', 'fen', 'sims']);
  assert.equal(typeof message.id, 'string');
  assert.equal(message.fen, ADVANCED_POSITION);
  assert.equal(message.sims, 4);
});

test('AC1: sims + maxMs posts exactly {id, fen, sims, maxMs}', async () => {
  const message = await postedSearchFor({ fen: ADVANCED_POSITION, sims: 4, maxMs: 1000 });
  assert.deepEqual(Object.keys(message), ['id', 'fen', 'sims', 'maxMs']);
  assert.equal(message.fen, ADVANCED_POSITION);
  assert.equal(message.sims, 4);
  assert.equal(message.maxMs, 1000);
});

test('AC1: time-only posts sims: TIME_ONLY_NODE_CEILING - 1 plus maxMs', async () => {
  const message = await postedSearchFor({ fen: ADVANCED_POSITION, maxMs: 1000 });
  assert.deepEqual(Object.keys(message), ['id', 'fen', 'sims', 'maxMs']);
  assert.equal(message.sims, TIME_ONLY_NODE_CEILING - 1);
  assert.equal(message.sims, 19_999);
  assert.equal(message.maxMs, 1000);
});

// ---------------------------------------------------------------------------
// AC2: rejections -- neither sims nor maxMs, maxMs 0/NaN/Infinity, a
// non-integer sims -- post nothing (no worker even spawned) and leave the
// client usable for a following valid play().
// ---------------------------------------------------------------------------

test('AC2: rejections post nothing; a following valid play() still works', async () => {
  let factoryCalls = 0;
  const throwingFactory: KomugiWorkerFactory = () => {
    factoryCalls += 1;
    throw new Error('must not be called');
  };
  const client = makeClient(throwingFactory);
  const badCases: Array<{ label: string; request: unknown; pattern: RegExp }> = [
    { label: 'neither sims nor maxMs', request: { fen: ADVANCED_POSITION }, pattern: /play\(\) needs sims, maxMs, or both/ },
    { label: 'maxMs 0', request: { fen: ADVANCED_POSITION, maxMs: 0 }, pattern: /play\(\) needs maxMs/ },
    { label: 'maxMs NaN', request: { fen: ADVANCED_POSITION, maxMs: NaN }, pattern: /play\(\) needs maxMs/ },
    { label: 'maxMs Infinity', request: { fen: ADVANCED_POSITION, maxMs: Infinity }, pattern: /play\(\) needs maxMs/ },
    { label: 'sims 1.5', request: { fen: ADVANCED_POSITION, sims: 1.5 }, pattern: /play\(\) needs sims to be a positive integer/ },
  ];
  try {
    for (const { label, request, pattern } of badCases) {
      await assert.rejects(client.play(request as PlayRequest), pattern, label);
    }
    assert.equal(factoryCalls, 0, 'no worker spawned for any rejected request');
    assert.equal(client.state, 'idle');
  } finally {
    client.dispose();
  }

  const { factory, posted, deliver } = fakeDrivenWorkerFactory();
  const workingClient = makeClient(factory);
  try {
    const playPromise = workingClient.play({ fen: ADVANCED_POSITION, sims: 4, batch: 1 });
    deliver(initReplyMessage(posted()));
    await tick();
    deliver({ id: searchMessage(posted()).id, san: 'P-5e', action: 0, stamp: 'match', stats: {}, descriptor: {} });
    expectMove(await playPromise);
  } finally {
    workingClient.dispose();
  }
});

// ---------------------------------------------------------------------------
// AC3: a strict-tsc consumer check -- the three ThinkCap forms compile with
// no cast; `{ fen }` alone is a type error (@ts-expect-error). Compiled the
// same way test/kom1226_host.test.ts compiles its own consumer snippet.
// ---------------------------------------------------------------------------

const CLIENT_PATH = resolve(HERE, '..', 'src', 'client.ts');

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

function thinkCapFixtureSource(fixtureDir: string): string {
  return `import type { PlayRequest } from '${relSpecifier(fixtureDir, CLIENT_PATH)}';

export const simsOnly: PlayRequest = { fen: 'fen', sims: 4 };
export const timeOnly: PlayRequest = { fen: 'fen', maxMs: 50 };
export const both: PlayRequest = { fen: 'fen', sims: 4, maxMs: 50 };
// @ts-expect-error neither sims nor maxMs is a ThinkCap
export const neither: PlayRequest = { fen: 'fen' };
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

test('AC3: the three ThinkCap forms compile with no cast; {fen} alone is a type error (strict tsc)', () => {
  const tscPath = resolveTsc();
  const fixtureDir = mkdtempSync(join(tmpdir(), 'komugi-kom1228-thinkcap-'));
  try {
    writeFileSync(join(fixtureDir, 'fixture.ts'), thinkCapFixtureSource(fixtureDir));
    writeJson(join(fixtureDir, 'tsconfig.fixture.json'), {
      compilerOptions: FIXTURE_COMPILER_OPTIONS,
      include: ['fixture.ts'],
    });

    const run = spawnSync(process.execPath, [tscPath, '-p', join(fixtureDir, 'tsconfig.fixture.json')], {
      cwd: fixtureDir,
      encoding: 'utf8',
    });
    assert.equal(run.status, 0, `tsc rejected a valid ThinkCap form or missed the {fen} type error:\n${run.stdout}${run.stderr}`);
  } finally {
    rmSync(fixtureDir, { recursive: true, force: true });
  }
});

// ---------------------------------------------------------------------------
// AC4: end to end through installWorkerHost with a fake ORT -- a time-only
// play with a small maxMs returns a move whose stats carry deadlineHit true
// and a partial playouts count. No worker/protocol/engine change: this runs
// the real search engine, just with a fake model and fake session.
// ---------------------------------------------------------------------------

async function sha256Hex(bytes: Uint8Array): Promise<string> {
  const digest = await globalThis.crypto.subtle.digest('SHA-256', bytes);
  return Array.from(new Uint8Array(digest))
    .map((byte) => byte.toString(16).padStart(2, '0'))
    .join('');
}

const MODEL_BYTES = new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
const ORT_ASSETS_WASM_ONLY = { wasm: { mjs: 'wasm.mjs', wasm: 'wasm.wasm' } };
const ACTION_SIZE_FOR_FAKE_ORT = 37341;

// Zero-filled rows shaped the way batch_evaluator.ts's runOrtBatch reads
// them (same convention as test/kom1226_host.test.ts's own fake ORT
// module) -- real tree-search/rules work, a trivial "model".
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
            const obsTensor = feeds.obs as { dims: readonly number[] };
            const bucket = obsTensor.dims[0];
            return {
              policy: { data: new Float32Array(bucket * ACTION_SIZE_FOR_FAKE_ORT) },
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
  return async () => ({ ok: true, status: 200, arrayBuffer: async () => bytes.buffer as ArrayBuffer });
}

// Bridges a KomugiClient straight to `installWorkerHost`, in-process: the
// client's own `postMessage` calls reach the host's `message` listener, and
// the host's `postMessage` replies reach the client's listener -- exactly
// what a real worker's postMessage channel does, with no worker_threads
// thread (so a real Date.now()-based deadline is still genuine wall time).
function installedWorkerFactory(options: WorkerHostOptions): KomugiWorkerFactory {
  return () => {
    const hostListeners: Array<(event: { data: unknown }) => void> = [];
    const clientListeners: Array<(event: { data: unknown }) => void> = [];
    const scope = {
      addEventListener(_type: 'message', listener: (event: { data: unknown }) => void): void {
        hostListeners.push(listener);
      },
      postMessage(reply: unknown): void {
        for (const listener of clientListeners) listener({ data: reply });
      },
    };
    installWorkerHost(scope, options);
    return {
      postMessage(message: unknown): void {
        for (const listener of hostListeners) listener({ data: message });
      },
      terminate(): void {
        // not exercised here
      },
      addEventListener(type: 'message' | 'error' | 'messageerror', listener: (event: { data: unknown } | unknown) => void): void {
        if (type === 'message') clientListeners.push(listener as (event: { data: unknown }) => void);
      },
    };
  };
}

test('AC4: time-only play through installWorkerHost (fake ORT) hits the deadline with a partial playout count', async () => {
  const sha = await sha256Hex(MODEL_BYTES);
  const options: WorkerHostOptions = { loadOrt: async () => makeFakeOrtModule(), fetch: fakeFetchOk(MODEL_BYTES) };
  const client = makeClient(installedWorkerFactory(options), {
    model: { url: 'model.onnx', sha256: sha },
    ortAssets: ORT_ASSETS_WASM_ONLY,
  });
  try {
    const init = await client.init();
    assert.equal(init.status, 'ready', `init should succeed, got ${JSON.stringify(init)}`);

    const result = await client.play({ fen: ADVANCED_POSITION, maxMs: 50, batch: 1 });
    const reply = expectMove(result);
    assert.equal(reply.stats.deadlineHit, true, `expected the deadline to fire, got ${JSON.stringify(reply.stats)}`);
    assert.ok(
      (reply.stats.playouts as number) < TIME_ONLY_NODE_CEILING - 1,
      `expected a partial playout count, got ${reply.stats.playouts}`,
    );
    assert.equal(reply.stats.sims, TIME_ONLY_NODE_CEILING - 1, 'stats.sims echoes the resolved node ceiling');
  } finally {
    client.dispose();
  }
});

test('AC4: the time-only move timeout is maxMs + moveTimeoutGraceMs', async (t) => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const { factory, posted, deliver } = fakeDrivenWorkerFactory();
  const client = makeClient(factory, { moveTimeoutGraceMs: 7 });
  try {
    const playPromise = client.play({ fen: ADVANCED_POSITION, maxMs: 50 });
    deliver(initReplyMessage(posted()));
    await tick();
    const search = searchMessage(posted());
    assert.equal(search.sims, TIME_ONLY_NODE_CEILING - 1);
    assert.equal(search.maxMs, 50);

    t.mock.timers.tick(56); // maxMs(50) + grace(7) - 1: still pending
    let settled = false;
    void playPromise.then(() => {
      settled = true;
    });
    await tick();
    assert.equal(settled, false, 'not yet timed out at maxMs + grace - 1');

    t.mock.timers.tick(1); // total 57 == maxMs + grace
    const result = await playPromise;
    assert.equal(result.status, 'failed');
    assert.equal(result.status === 'failed' && result.failure.code, 'timeout');
    assert.equal(result.status === 'failed' && result.failure.stage, 'search');
  } finally {
    client.dispose();
  }
});
