// KOM-1199 unit U3a (design docs/design/kom1199-client.md sections 2, 3, 6,
// 7): `KomugiClient` over the real node:worker_threads harness
// (test/support/node_worker_adapter.ts) with the fake evaluator -- no real
// model, no network.

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';

import { ACTION_ENCODING_VERSION, ACTION_SIZE, ADVANCED_POSITION } from '../src/constants.ts';
import { boardFromFen } from '../src/fen.ts';
import { Game } from '../src/game.ts';
import { DEFAULT_RULES_PROFILE, RepetitionRule } from '../src/repetition.ts';
import type { KomugiClientOptions, KomugiWorkerFactory, PlayRequest, PlayResult } from '../src/client.ts';
import type { BackendRecord } from '../src/backend.ts';
import type { WorkerReplyOk } from '../src/protocol.ts';
import type { WorkerProgressEvent } from '../src/protocol.ts';
import { resolveModelCard } from '../src/model_registry.ts';
import { nodeWorkerFactory } from './support/node_worker_adapter.ts';

const CLIENT_SOURCE = new URL('../src/client.ts', import.meta.url);

const { KomugiClient } = await import('../src/client.ts');

const REQUEST: PlayRequest = { fen: ADVANCED_POSITION, sims: 4, batch: 1 };

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

// Test code, not the main-thread client (section 6, review F3): the
// request's rules profile is Game's 4th constructor parameter.
function assertLegalShaped(request: PlayRequest, reply: WorkerReplyOk): void {
  const game = new Game(boardFromFen(request.fen), RepetitionRule.FULL, 1, request.rules ?? DEFAULT_RULES_PROFILE);
  assert.equal(game.validMoves()[reply.action], 1, `action ${reply.action} is not legal`);
}

test('client play: one worker, a legal-shaped move', async () => {
  const { factory, spawned } = nodeWorkerFactory(['ok']);
  const client = makeClient(factory);
  try {
    assert.equal(client.state, 'idle');
    const reply = expectMove(await client.play(REQUEST));
    assertLegalShaped(REQUEST, reply);
    assert.equal(spawned(), 1);
    assert.equal(client.state, 'ready');
    assert.equal(client.lastFailure, null);
  } finally {
    client.dispose();
  }
});

test('client init failure latch + retry (AC3)', async () => {
  const { factory, spawned } = nodeWorkerFactory(['init-model-fetch', 'ok']);
  const client = makeClient(factory);
  try {
    const first = await client.play(REQUEST);
    assert.equal(first.status, 'failed');
    assert.ok(!('reply' in first), 'a failed play carries no reply');
    assert.equal(first.status === 'failed' && first.failure.code, 'model-fetch');
    assert.equal(first.status === 'failed' && first.failure.stage, 'init');
    assert.equal(client.state, 'failed');
    assert.equal(client.lastFailure?.code, 'model-fetch');

    const second = await client.play(REQUEST);
    assert.equal(second.status, 'failed');
    assert.equal(second.status === 'failed' && second.failure.code, 'model-fetch');
    assert.equal(spawned(), 1, 'a latched failure does not call createWorker');
    const init = await client.init();
    assert.equal(init.status, 'failed');
    assert.equal(spawned(), 1);

    client.retry();
    assert.equal(client.state, 'idle');
    expectMove(await client.play(REQUEST));
    assert.equal(spawned(), 2);
  } finally {
    client.dispose();
  }
});

test('client sha mismatch', async () => {
  const { factory } = nodeWorkerFactory(['init-sha-mismatch']);
  const client = makeClient(factory);
  try {
    const result = await client.init();
    assert.equal(result.status, 'failed');
    assert.equal(result.status === 'failed' && result.failure.code, 'sha-mismatch');
    assert.equal(result.status === 'failed' && result.failure.stage, 'init');
  } finally {
    client.dispose();
  }
});

test('client uncoded search error: backend-unavailable, worker kept', async () => {
  const { factory, spawned } = nodeWorkerFactory(['uncoded-search-error']);
  const client = makeClient(factory);
  try {
    const result = await client.play(REQUEST);
    assert.equal(result.status, 'failed');
    assert.equal(result.status === 'failed' && result.failure.code, 'backend-unavailable');
    assert.equal(result.status === 'failed' && result.failure.stage, 'search');
    assert.equal(client.state, 'ready');
    assert.equal(spawned(), 1);
  } finally {
    client.dispose();
  }
});

test('client history mismatch (review F2): a move stamped mismatch, no restart', async () => {
  const { factory, spawned } = nodeWorkerFactory(['ok']);
  const client = makeClient(factory);
  try {
    const request: PlayRequest = { ...REQUEST, history: ['not-a-san'], start: ADVANCED_POSITION };
    const reply = expectMove(await client.play(request));
    assert.equal(reply.stamp, 'mismatch');
    assertLegalShaped(request, reply);
    assert.equal(client.state, 'ready');
    assert.equal(spawned(), 1);

    expectMove(await client.play(REQUEST));
    assert.equal(spawned(), 1, 'the follow-up play reuses the same worker');
  } finally {
    client.dispose();
  }
});

// Polls instead of a fixed sleep: the real worker_threads round-trip to
// reach 'searching' has no fixed duration.
async function waitFor(predicate: () => boolean, label: string): Promise<void> {
  for (let i = 0; i < 400; i += 1) {
    if (predicate()) return;
    await new Promise((resolve) => globalThis.setTimeout(resolve, 5));
  }
  throw new Error(`timed out waiting for ${label}`);
}

test('client stop: resolves the pending play stopped; next play restarts (section 5)', async () => {
  const { factory, spawned } = nodeWorkerFactory(['drop-search', 'ok']);
  const client = makeClient(factory);
  try {
    const first = client.play(REQUEST);
    await waitFor(() => client.state === 'searching', 'searching state');
    client.stop();
    assert.deepEqual(await first, { status: 'stopped' });
    assert.equal(client.state, 'idle');

    const second = expectMove(await client.play(REQUEST));
    assertLegalShaped(REQUEST, second);
    assert.equal(spawned(), 2);
  } finally {
    client.dispose();
  }
});

test('client stop: no leaked timer, and a late event from the stopped worker cannot resolve a later play', async () => {
  const { factory, spawned } = nodeWorkerFactory(['drop-search', 'ok']);
  const client = makeClient(factory);
  try {
    const first = client.play(REQUEST);
    await waitFor(() => client.state === 'searching', 'searching state');
    client.stop();
    assert.deepEqual(await first, { status: 'stopped' });

    await new Promise((resolve) => globalThis.setTimeout(resolve, 1000));
    const second = expectMove(await client.play(REQUEST));
    assertLegalShaped(REQUEST, second);
    assert.equal(spawned(), 2, 'exactly one restart; no event from the stopped worker resolved anything');
  } finally {
    client.dispose();
  }
});

test('client search timeout: terminates, restarts lazily on the next play', async () => {
  const { factory, spawned } = nodeWorkerFactory(['drop-search', 'ok']);
  const client = makeClient(factory, { moveTimeoutMs: 500 });
  try {
    const result = await client.play(REQUEST);
    assert.equal(result.status, 'failed');
    assert.equal(result.status === 'failed' && result.failure.code, 'timeout');
    assert.equal(result.status === 'failed' && result.failure.stage, 'search');
    assert.equal(client.state, 'idle');

    const second = expectMove(await client.play(REQUEST));
    assertLegalShaped(REQUEST, second);
    assert.equal(spawned(), 2);
  } finally {
    client.dispose();
  }
});

test('client crash restart (AC2): search-stage crash restarts on the next play', async () => {
  const { factory, spawned } = nodeWorkerFactory(['crash-on-search', 'ok']);
  const client = makeClient(factory);
  try {
    const result = await client.play(REQUEST);
    assert.equal(result.status, 'failed');
    assert.equal(result.status === 'failed' && result.failure.code, 'worker-crash');
    assert.equal(result.status === 'failed' && result.failure.stage, 'search');
    assert.equal(client.state, 'idle');

    const second = expectMove(await client.play(REQUEST));
    assertLegalShaped(REQUEST, second);
    assert.equal(spawned(), 2);
  } finally {
    client.dispose();
  }
});

test('client crash cap: latches failed after maxConsecutiveCrashes, no further spawn', async () => {
  const { factory, spawned } = nodeWorkerFactory(['crash-on-search']);
  const client = makeClient(factory, { maxConsecutiveCrashes: 2 });
  try {
    const first = await client.play(REQUEST);
    assert.equal(first.status, 'failed');
    assert.equal(client.state, 'idle');
    assert.equal(spawned(), 1);

    const second = await client.play(REQUEST);
    assert.equal(second.status, 'failed');
    assert.equal(client.state, 'failed');
    assert.equal(spawned(), 2);

    const third = await client.play(REQUEST);
    assert.equal(third.status, 'failed');
    assert.equal(third.status === 'failed' && third.failure.code, 'worker-crash');
    assert.equal(spawned(), 2, 'latched: play() does not call createWorker again');
  } finally {
    client.dispose();
  }
});

test('client illegal reply: terminates, restarts lazily on the next play', async () => {
  const { factory, spawned } = nodeWorkerFactory(['garbage-reply', 'ok']);
  const client = makeClient(factory);
  try {
    const result = await client.play(REQUEST);
    assert.equal(result.status, 'failed');
    assert.equal(result.status === 'failed' && result.failure.code, 'illegal-reply');
    assert.equal(client.state, 'idle');

    const second = expectMove(await client.play(REQUEST));
    assertLegalShaped(REQUEST, second);
    assert.equal(spawned(), 2);
  } finally {
    client.dispose();
  }
});

test('client threads-timeout (KOM-1214): probeRecord relayed, one restart with maxThreads 1', async () => {
  const { factory, spawned, posted } = nodeWorkerFactory(['init-threads-timeout', 'ok']);
  const records: BackendRecord[] = [];
  const client = makeClient(factory, { threads: 4, onProbeRecord: (record) => records.push(record) });
  try {
    const reply = expectMove(await client.play(REQUEST));
    assertLegalShaped(REQUEST, reply);
    assert.equal(spawned(), 2);

    // The threads-timeout reply's own record first, then the restart's
    // successful init record.
    assert.equal(records.length, 2);
    assert.equal(records[0]?.key, 'test-threads-timeout');
    assert.equal(records[0]?.probedAt, 1234);

    const firstInit = posted()[0]?.[0] as Record<string, unknown>;
    const secondInit = posted()[1]?.[0] as Record<string, unknown>;
    assert.equal(firstInit.type, 'init');
    assert.ok(!('maxThreads' in firstInit), 'the first worker is not capped');
    assert.equal(secondInit.type, 'init');
    assert.equal(secondInit.maxThreads, 1);
    assert.equal(secondInit.threads, 4);
    assert.equal(secondInit.modelUrl, 'model.onnx', 'same Komugi model, never a substitute');
    assert.deepEqual(secondInit.cachedRecord, records[0], 'the relayed record is the restart cachedRecord');
    assert.equal(client.state, 'ready');
  } finally {
    client.dispose();
  }
});

test('client threads-timeout twice: typed failure, latched, no third spawn', async () => {
  const { factory, spawned } = nodeWorkerFactory(['init-threads-timeout']);
  const records: BackendRecord[] = [];
  const client = makeClient(factory, { onProbeRecord: (record) => records.push(record) });
  try {
    const result = await client.play(REQUEST);
    assert.equal(result.status, 'failed');
    assert.equal(result.status === 'failed' && result.failure.code, 'threads-timeout');
    assert.equal(result.status === 'failed' && result.failure.stage, 'init');
    assert.equal(spawned(), 2);
    assert.equal(records.length, 2);
    assert.equal(client.state, 'failed');
  } finally {
    client.dispose();
  }
});

test('client init timeout latches timeout/init', async () => {
  let terminated = 0;
  const silent: KomugiWorkerFactory = () => ({
    postMessage: () => undefined,
    terminate: () => {
      terminated += 1;
    },
    addEventListener: () => undefined,
  });
  const client = makeClient(silent, { initTimeoutMs: 50 });
  try {
    const result = await client.init();
    assert.equal(result.status, 'failed');
    assert.equal(result.status === 'failed' && result.failure.code, 'timeout');
    assert.equal(result.status === 'failed' && result.failure.stage, 'init');
    assert.equal(terminated, 1);
    assert.equal(client.state, 'failed');
  } finally {
    client.dispose();
  }
});

test('client init is shared; dispose resolves a pending init and rejects later calls', async () => {
  let spawns = 0;
  const silent: KomugiWorkerFactory = () => {
    spawns += 1;
    return { postMessage: () => undefined, terminate: () => undefined, addEventListener: () => undefined };
  };
  const client = makeClient(silent);
  const a = client.init();
  const b = client.init();
  assert.equal(a, b);
  assert.equal(spawns, 1);
  assert.equal(client.state, 'starting');
  client.dispose();
  const result = await a;
  assert.deepEqual(result, { status: 'failed', failure: { code: 'worker-crash', stage: 'init', message: 'disposed' } });
  assert.equal(client.state, 'disposed');
  await assert.rejects(client.play(REQUEST), /^Error: KomugiClient:/);
  await assert.rejects(client.init(), /^Error: KomugiClient:/);
});

test('client play while a play is in flight rejects', async () => {
  const { factory } = nodeWorkerFactory(['ok']);
  const client = makeClient(factory);
  try {
    const first = client.play(REQUEST);
    await assert.rejects(client.play(REQUEST), /KomugiClient: play\(\) while a play is in flight/);
    expectMove(await first);
  } finally {
    client.dispose();
  }
});

test('client import starts nothing (AC4)', async () => {
  const originalFetch = globalThis.fetch;
  let fetchCalls = 0;
  let factoryCalls = 0;
  globalThis.fetch = (() => {
    fetchCalls += 1;
    return Promise.reject(new Error('no fetch in tests'));
  }) as typeof fetch;
  try {
    const mod = await import('../src/client.ts');
    const client = new mod.KomugiClient({
      createWorker: () => {
        factoryCalls += 1;
        throw new Error('must not be called');
      },
      model: { url: 'model.onnx', sha256: 'sha-a' },
      ortAssets: { wasm: { mjs: 'ort.mjs', wasm: 'ort.wasm' } },
    });
    assert.equal(client.state, 'idle');
    assert.equal(factoryCalls, 0);
    assert.equal(fetchCalls, 0);
    client.dispose();
  } finally {
    globalThis.fetch = originalFetch;
  }
});

// KOM-1188 unit 2 acceptance 8/9 (docs/design/kom1188-model-releases.md
// section 2.2.1/2.2.3, review F1/N1/N2): a client built from a resolved
// card binds its planeMapVersion; the card's rulesProfile is recorded only.
const CARD_SHA = 'ab'.repeat(32);
function releaseCard(planeMapVersion: 3 | 4, rulesProfile: string): Record<string, unknown> {
  const id = `komugi-c6-s1-${CARD_SHA.slice(0, 8)}`;
  return {
    id,
    campaign: 'c6',
    iteration: 1,
    step: 1,
    compatibilityKey: { planeMapVersion, actionEncodingVersion: ACTION_ENCODING_VERSION, rulesProfile, architectureTag: 'b2c16' },
    modesTrained: ['advanced'],
    assets: { pt: { path: `${id}.pt`, sha256: 'cd'.repeat(32) }, onnx: { path: `${id}.onnx`, sha256: CARD_SHA } },
    sourceCheckpoint: { name: 'checkpoint-000000001-0123456789abcdef.pt', sha256: 'ef'.repeat(32) },
    exportOpset: 17,
    exportParity: { n: 4, maxAbsDelta: 0, atol: 0, rtol: 0 },
    evalPointers: [],
    license: 'MIT, Copyright (c) 2026 chessai',
    createdAt: '2026-11-22T00:00:00Z',
  };
}

function resolvedModel(card: Record<string, unknown>): KomugiClientOptions['model'] {
  const result = resolveModelCard(card, (asset) => asset.path);
  assert.equal(result.status, 'ok', JSON.stringify(result));
  if (result.status !== 'ok') throw new Error('unreachable');
  return result.model;
}

function searchMessages(posted: () => unknown[][]): Record<string, unknown>[] {
  return posted()
    .flat()
    .filter((message) => (message as { type?: unknown }).type !== 'init') as Record<string, unknown>[];
}

test('client from a v4 card: posts planeMapVersion 4 by default; a v3 request fails incompatible-model, nothing posted', async () => {
  const { factory, posted } = nodeWorkerFactory(['ok']);
  const client = new KomugiClient({
    createWorker: factory,
    model: resolvedModel(releaseCard(4, 'komugi')),
    ortAssets: { wasm: { mjs: 'ort.mjs', wasm: 'ort.wasm' } },
  });
  try {
    expectMove(await client.play(REQUEST));
    const searches = searchMessages(posted);
    assert.equal(searches.length, 1);
    assert.equal(searches[0]?.planeMapVersion, 4);
    const init = posted()[0]?.[0] as Record<string, unknown>;
    assert.equal(init.type, 'init');
    assert.ok(!('compatibility' in init), 'compatibility never crosses postMessage at init');

    const refused = await client.play({ ...REQUEST, planeMapVersion: 3 });
    assert.equal(refused.status, 'failed');
    assert.equal(refused.status === 'failed' && refused.failure.code, 'incompatible-model');
    assert.equal(refused.status === 'failed' && refused.failure.stage, 'search');
    assert.equal(searchMessages(posted).length, 1, 'no search message posted for the refused request');
    assert.equal(client.lastFailure?.code, 'incompatible-model');

    // A matching explicit value is fine; the worker was kept.
    expectMove(await client.play({ ...REQUEST, planeMapVersion: 4 }));
    assert.equal(searchMessages(posted)[1]?.planeMapVersion, 4);
  } finally {
    client.dispose();
  }
});

test('client from a v4 card refuses a v3 first play before any search is posted', async () => {
  const { factory, posted } = nodeWorkerFactory(['ok']);
  const client = new KomugiClient({
    createWorker: factory,
    model: resolvedModel(releaseCard(4, 'komugi')),
    ortAssets: { wasm: { mjs: 'ort.mjs', wasm: 'ort.wasm' } },
  });
  try {
    const refused = await client.play({ ...REQUEST, planeMapVersion: 3 });
    assert.equal(refused.status === 'failed' && refused.failure.code, 'incompatible-model');
    assert.equal(searchMessages(posted).length, 0);
  } finally {
    client.dispose();
  }
});

test('client from a ligungi card: no rules field unless the request sets one (rules never defaulted from the card)', async () => {
  const cases: Array<KomugiClientOptions['model']> = [resolvedModel(releaseCard(3, 'ligungi')), { url: 'model.onnx', sha256: 'sha-a' }];
  for (const model of cases) {
    const { factory, posted } = nodeWorkerFactory(['ok']);
    const client = new KomugiClient({ createWorker: factory, model, ortAssets: { wasm: { mjs: 'ort.mjs', wasm: 'ort.wasm' } } });
    try {
      expectMove(await client.play(REQUEST));
      const gungijs: PlayRequest = { ...REQUEST, rules: 'gungijs' };
      assertLegalShaped(gungijs, expectMove(await client.play(gungijs)));
      const [plain, withRules] = searchMessages(posted);
      assert.ok(plain !== undefined && !('rules' in plain), `no rules field expected, got ${JSON.stringify(plain)}`);
      assert.equal(withRules?.rules, 'gungijs');
    } finally {
      client.dispose();
    }
  }
});

// KOM-1200 unit 3a (design docs/design/kom1200-model-loading.md section 3,
// section 8 "release ownership"): a fake worker the test drives by hand
// (no real worker_threads thread needed), used only to reach 'ready' with
// no pending play/init so a crash lands as the "idle crash" row.
function fakeIdleWorkerFactory(): { factory: KomugiWorkerFactory; replyInit: () => void; crash: () => void } {
  let messageListener: ((event: { data: unknown }) => void) | null = null;
  let errorListener: ((event: unknown) => void) | null = null;
  let initId: string | null = null;
  const factory: KomugiWorkerFactory = () => ({
    postMessage(message: unknown): void {
      const init = message as { id: string; type?: string };
      if (init.type === 'init') initId = init.id;
    },
    terminate(): void {
      // not exercised: this test disposes without a mid-scenario terminate
    },
    addEventListener(type: 'message' | 'error' | 'messageerror', listener: (event: { data: unknown } | unknown) => void): void {
      if (type === 'message') messageListener = listener as (event: { data: unknown }) => void;
      else errorListener = listener as (event: unknown) => void;
    },
  });
  return {
    factory,
    replyInit: () => {
      if (initId !== null) messageListener?.({ data: { id: initId, backend: 'wasm', threads: 1, batch: 1, probeRecord: {} } });
    },
    crash: () => errorListener?.(new Error('idle crash')),
  };
}

test('client dispose calls model.release once', async () => {
  // stop(): the pending play resolves stopped; release stays uncalled.
  {
    const { factory } = nodeWorkerFactory(['drop-search']);
    let calls = 0;
    const client = makeClient(factory, { model: { url: 'model.onnx', sha256: 'sha-a', release: () => { calls += 1; } } });
    const play = client.play(REQUEST);
    await waitFor(() => client.state === 'searching', 'searching state');
    client.stop();
    assert.deepEqual(await play, { status: 'stopped' });
    assert.equal(calls, 0, 'stop() must not call release');
    client.dispose();
    assert.equal(calls, 1);
  }

  // retry(): no pending op; release stays uncalled.
  {
    const { factory } = nodeWorkerFactory(['ok']);
    let calls = 0;
    const client = makeClient(factory, { model: { url: 'model.onnx', sha256: 'sha-a', release: () => { calls += 1; } } });
    client.retry();
    assert.equal(calls, 0, 'retry() must not call release');
    client.dispose();
    assert.equal(calls, 1);
  }

  // Search timeout (failSearch -> toIdle): release stays uncalled.
  {
    const { factory } = nodeWorkerFactory(['drop-search']);
    let calls = 0;
    const client = makeClient(factory, {
      moveTimeoutMs: 300,
      model: { url: 'model.onnx', sha256: 'sha-a', release: () => { calls += 1; } },
    });
    const result = await client.play(REQUEST);
    assert.equal(result.status, 'failed');
    assert.equal(result.status === 'failed' && result.failure.code, 'timeout');
    assert.equal(calls, 0, 'a search timeout must not call release');
    client.dispose();
    assert.equal(calls, 1);
  }

  // Idle crash (onCrash with no pending init/play -> toIdle): release
  // stays uncalled; dispose() then calls it once, and a second dispose()
  // does not call it again.
  {
    const { factory, replyInit, crash } = fakeIdleWorkerFactory();
    let calls = 0;
    const client = new KomugiClient({
      createWorker: factory,
      model: { url: 'model.onnx', sha256: 'sha-a', release: () => { calls += 1; } },
      ortAssets: { wasm: { mjs: 'ort.mjs', wasm: 'ort.wasm' } },
    });
    const initPromise = client.init();
    replyInit();
    const initResult = await initPromise;
    assert.equal(initResult.status, 'ready');
    crash();
    assert.equal(client.state, 'idle');
    assert.equal(calls, 0, 'an idle crash must not call release');
    client.dispose();
    assert.equal(calls, 1);
    client.dispose();
    assert.equal(calls, 1, 'a second dispose() does not call release again');
  }
});

test('client dispose isolates a throwing release', async () => {
  const seen: unknown[] = [];
  const originalQueueMicrotask = globalThis.queueMicrotask;
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
    const client = makeClient(
      () => {
        throw new Error('must not be called');
      },
      { model: { url: 'model.onnx', sha256: 'sha-a', release: () => { throw new Error('release boom'); } } },
    );
    client.dispose();
    assert.equal(client.state, 'disposed');
    await Promise.resolve();
    assert.equal(seen.length, 1, 'the throwing release surfaced asynchronously, not synchronously from dispose()');
  } finally {
    globalThis.queueMicrotask = originalQueueMicrotask;
  }
});

test('client.ts imports only types, except ./client_failures.ts', () => {
  const source = readFileSync(CLIENT_SOURCE, 'utf8');
  const imports = source.split('\n').filter((line) => /^import\s/.test(line));
  assert.ok(imports.length > 0);
  for (const line of imports) {
    assert.ok(line.startsWith('import type ') || line.endsWith("from './client_failures.ts';"), `value import: ${line}`);
  }
});

test('client.ts ACTION_SIZE matches constants.ts', () => {
  const source = readFileSync(CLIENT_SOURCE, 'utf8');
  assert.match(source, new RegExp(`^const ACTION_SIZE = ${ACTION_SIZE};$`, 'm'));
});

// KOM-1195 U7 (design docs/design/kom1195-analysis-api.md section 7, AC6):
// moveNow()/analyze()/onProgress -- a hand-driven fake worker the test
// delivers every message to itself, no real worker_threads thread needed
// (the search's own stats/tactical shape is never validated client-side,
// section 2 of client_failures.ts).
function fakeDrivenWorkerFactory(): {
  factory: KomugiWorkerFactory;
  posted: () => Record<string, unknown>[];
  deliver: (data: unknown) => void;
  terminated: () => number;
} {
  let messageListener: ((event: { data: unknown }) => void) | null = null;
  const postedMessages: Record<string, unknown>[] = [];
  let terminateCount = 0;
  const factory: KomugiWorkerFactory = () => ({
    postMessage(message: unknown): void {
      postedMessages.push(message as Record<string, unknown>);
    },
    terminate(): void {
      terminateCount += 1;
    },
    addEventListener(type: 'message' | 'error' | 'messageerror', listener: (event: { data: unknown } | unknown) => void): void {
      if (type === 'message') messageListener = listener as (event: { data: unknown }) => void;
    },
  });
  return {
    factory,
    posted: () => postedMessages,
    deliver: (data: unknown) => messageListener?.({ data }),
    terminated: () => terminateCount,
  };
}

function initReplyMessage(posted: Record<string, unknown>[]): Record<string, unknown> {
  const init = posted.find((m) => m.type === 'init');
  if (init === undefined) throw new Error('no init message posted yet');
  return { id: init.id, backend: 'wasm', threads: 1, batch: 1, probeRecord: {} };
}

function searchMessage(posted: Record<string, unknown>[]): { id: string } {
  const search = posted.find((m) => 'fen' in m);
  if (search === undefined) throw new Error('no search message posted yet');
  return search as { id: string };
}

function fakeMoveReply(id: unknown, stats: Record<string, unknown> = {}): Record<string, unknown> {
  return { id, san: 'P-5e', action: 0, stamp: 'match', stats, descriptor: {} };
}

// Flushes the microtask queue: settleInit's `init.resolve` runs play()'s/
// analyze()'s own `.then()` continuation (postSearch/postAnalyze) as a
// microtask, not synchronously with deliver().
async function tick(): Promise<void> {
  await new Promise((resolve) => queueMicrotask(resolve));
  await new Promise((resolve) => queueMicrotask(resolve));
}

test('moveNow(): false when no stoppable play/analysis is pending', async () => {
  assert.equal(new KomugiClient({ createWorker: () => { throw new Error('unused'); }, model: { url: 'model.onnx', sha256: 'sha-a' }, ortAssets: { wasm: { mjs: 'ort.mjs', wasm: 'ort.wasm' } } }).moveNow(), false, 'idle: no play pending');

  const { factory } = nodeWorkerFactory(['ok']);
  const client = makeClient(factory);
  try {
    const result = client.play(REQUEST); // not opted in
    assert.equal(client.moveNow(), false, 'a plain (un-opted) play is not stoppable');
    expectMove(await result);
  } finally {
    client.dispose();
  }
});

test('moveNow(): worker answers after stop, resolves move (AC6)', async () => {
  const { factory, posted, deliver } = fakeDrivenWorkerFactory();
  const client = makeClient(factory);
  try {
    const playPromise = client.play({ ...REQUEST, stoppable: true });
    deliver(initReplyMessage(posted()));
    await tick();
    const search = searchMessage(posted());

    assert.equal(client.moveNow(), true);
    await tick();
    assert.ok(posted().some((m) => m.type === 'stop'), 'stop posted');

    deliver(fakeMoveReply(search.id, { playouts: 9, stopReason: 'stop', deadlineHit: true }));
    const result = await playPromise;
    assert.equal(result.status, 'move');
    assert.equal((result as { status: 'move'; reply: WorkerReplyOk }).reply.stats.stopReason, 'stop');
  } finally {
    client.dispose();
  }
});

test('analyze(): fake timers advanced 120s stays pending; moveNow() then resolves move (AC6)', async (t) => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const { factory, posted, deliver } = fakeDrivenWorkerFactory();
  const client = makeClient(factory);
  try {
    const analyzePromise = client.analyze({ fen: REQUEST.fen, nodeCeiling: 20_001 });
    deliver(initReplyMessage(posted()));
    await tick();
    const search = searchMessage(posted());

    t.mock.timers.tick(120_000);
    let settled = false;
    void analyzePromise.then(() => {
      settled = true;
    });
    await tick();
    assert.equal(settled, false, 'no 60s move timer for analyze()');

    assert.equal(client.moveNow(), true);
    await tick();
    assert.ok(posted().some((m) => m.type === 'stop'), 'stop posted');

    deliver(fakeMoveReply(search.id, { playouts: 500, stopReason: 'stop', deadlineHit: true }));
    const result = await analyzePromise;
    assert.equal(result.status, 'move');
  } finally {
    client.dispose();
  }
});

test('moveNow(): grace timeout settles failed/timeout, no crash-latch increment (AC6)', async (t) => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const { factory, posted, deliver, terminated } = fakeDrivenWorkerFactory();
  const client = makeClient(factory, { stopGraceMs: 1000, maxConsecutiveCrashes: 1 });
  try {
    const playPromise = client.play({ ...REQUEST, stoppable: true });
    deliver(initReplyMessage(posted()));
    await tick();
    searchMessage(posted());

    assert.equal(client.moveNow(), true);
    await tick();
    assert.ok(posted().some((m) => m.type === 'stop'), 'stop posted');

    t.mock.timers.tick(1000);
    const result = await playPromise;
    assert.equal(result.status, 'failed');
    assert.equal(result.status === 'failed' && result.failure.code, 'timeout');
    assert.equal(result.status === 'failed' && result.failure.stage, 'search');
    assert.equal(terminated(), 1, 'the grace fallback terminates the worker');
    assert.equal(client.state, 'idle', 'the grace fallback never counts toward the crash latch');
  } finally {
    client.dispose();
  }
});

test('onProgress: fires only for the current search id (AC6)', async () => {
  const events: WorkerProgressEvent[] = [];
  const { factory, posted, deliver } = fakeDrivenWorkerFactory();
  const client = makeClient(factory, { onProgress: (event) => events.push(event) });
  try {
    const playPromise = client.play({ ...REQUEST, stoppable: true });
    deliver(initReplyMessage(posted()));
    await tick();
    const search = searchMessage(posted());

    deliver({ id: 'stale-id', type: 'progress', playouts: 1, bestSan: 'P-5e', bestAction: 0, rootWdl: [0, 0, 0], pv: [] });
    deliver({ id: search.id, type: 'progress', playouts: 2, bestSan: 'P-5e', bestAction: 0, rootWdl: [0, 0, 0], pv: [] });
    assert.equal(events.length, 1, 'only the current search id reaches onProgress');
    assert.equal(events[0]?.playouts, 2);

    deliver(fakeMoveReply(search.id));
    const result = await playPromise;
    assert.equal(result.status, 'move');
  } finally {
    client.dispose();
  }
});

test('moveNow(): a move reply before the grace timer fires leaves the worker alive (N9)', async (t) => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const { factory, posted, deliver, terminated } = fakeDrivenWorkerFactory();
  const client = makeClient(factory, { stopGraceMs: 1000 });
  try {
    const playPromise = client.play({ ...REQUEST, stoppable: true });
    deliver(initReplyMessage(posted()));
    await tick();
    const search = searchMessage(posted());

    assert.equal(client.moveNow(), true);
    await tick();
    assert.ok(posted().some((m) => m.type === 'stop'), 'stop posted');

    deliver(fakeMoveReply(search.id, { playouts: 3, stopReason: 'stop', deadlineHit: true }));
    const result = await playPromise;
    assert.equal(result.status, 'move');

    t.mock.timers.tick(1000);
    assert.equal(terminated(), 0, 'settlePlay cleared the grace timer; it never fires');
    assert.equal(client.state, 'ready');
  } finally {
    client.dispose();
  }
});

test('moveNow() during a slow init arms no grace timer until the stop is posted', async (t) => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const { factory, posted, deliver, terminated } = fakeDrivenWorkerFactory();
  const client = makeClient(factory, { stopGraceMs: 1000 });
  try {
    const playPromise = client.play({ ...REQUEST, stoppable: true });
    // init is still pending: no search message exists yet.
    assert.equal(client.moveNow(), true, 'records the request even before the search exists');
    assert.equal(posted().some((m) => m.type === 'stop'), false, 'no stop posted while still initialising');

    t.mock.timers.tick(1000);
    assert.equal(terminated(), 0, 'no grace timer armed during init, so stopGraceMs elapsing here does nothing');

    deliver(initReplyMessage(posted()));
    await tick();
    const search = searchMessage(posted());
    assert.ok(posted().some((m) => m.type === 'stop'), 'the deferred stop is sent right after the search message');

    deliver(fakeMoveReply(search.id, { playouts: 1, stopReason: 'stop', deadlineHit: true }));
    const result = await playPromise;
    assert.equal(result.status, 'move');
  } finally {
    client.dispose();
  }
});
