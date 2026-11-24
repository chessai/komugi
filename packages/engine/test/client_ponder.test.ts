// KOM-1226 item 4 (tickets/KOM-1226.md): pondering as a KomugiClient
// configuration option (`ponder` constructor option, `setPonder()`,
// re-applied on every worker the client spawns, default unchanged).
//
// (a)-(h) drive `KomugiClient` over an in-memory fake `KomugiWorkerLike`
// (not a real worker thread -- deterministic ordering control, in
// particular for (h), which needs a search to stay open past a
// ponderSettings reply) that records every posted message and answers
// init/search/ponderSettings itself, mirroring the engine's own reply
// shapes and k-clamping (ponder.ts's clampPonderK: round, then clamp to
// [1, 3]). (i) is a separate, in-process case through the REAL worker host
// (`installWorkerHost` + a fake ORT module), following
// test/kom1226_host.test.ts's own fake-scope pattern, wired to a
// `KomugiWorkerLike` adapter so a real `KomugiClient` drives it with no
// worker thread at all.

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { ADVANCED_POSITION } from '../src/constants.ts';
import type { KomugiClientOptions, KomugiWorkerFactory, KomugiWorkerLike, PlayRequest } from '../src/client.ts';
import { KomugiClient } from '../src/client.ts';
import type { OrtModule, OrtSession } from '../src/ort_core.ts';
import type { WorkerHostOptions } from '../src/worker_host.ts';
import { installWorkerHost } from '../src/worker_host.ts';

const REQUEST: PlayRequest = { fen: ADVANCED_POSITION, sims: 4, batch: 1 };

function makeClient(factory: KomugiWorkerFactory, extra: Partial<KomugiClientOptions> = {}): InstanceType<typeof KomugiClient> {
  return new KomugiClient({
    createWorker: factory,
    model: { url: 'model.onnx', sha256: 'sha-a' },
    ortAssets: { wasm: { mjs: 'ort.mjs', wasm: 'ort.wasm' } },
    ...extra,
  });
}

function isPonderSettingsMessage(message: unknown): message is { type: 'ponderSettings'; enabled?: boolean; k?: number } {
  return typeof message === 'object' && message !== null && (message as { type?: unknown }).type === 'ponderSettings';
}

// A plain search request has no `type` field at all (protocol.ts's
// `WorkerRequest`) -- same distinction test/support/client_test_worker.ts
// uses to tell a search apart from init/ponder messages.
function isSearchMessage(message: unknown): boolean {
  return typeof message === 'object' && message !== null && !('type' in (message as object));
}

// Polls instead of a fixed sleep: reaching 'searching' goes through at
// least one microtask round-trip (the fake worker's own init reply), with
// no fixed duration (mirrors test/client.test.ts's own waitFor).
async function waitFor(predicate: () => boolean, label: string): Promise<void> {
  for (let i = 0; i < 400; i += 1) {
    if (predicate()) return;
    await new Promise((resolve) => globalThis.setTimeout(resolve, 5));
  }
  throw new Error(`timed out waiting for ${label}`);
}

// Mirrors ponder.ts's clampPonderK (design 5.4's "up to 3 predicted
// replies"): round, then clamp to [1, 3].
function clampK(k: number): number {
  if (!Number.isFinite(k)) return 1;
  return Math.min(3, Math.max(1, Math.round(k)));
}

interface SpawnScript {
  // Emits an 'error' event instead of a reply for the search message (not
  // for init/ponderSettings) -- mirrors client_test_worker.ts's
  // 'crash-on-search' mode, in-memory.
  crashOnSearch?: boolean;
  // Defers the search's own move reply until the test calls
  // `releaseSearch(spawnIndex)`, instead of the default same-tick-ish
  // auto-reply -- lets (h) hold a play open past a ponderSettings
  // round-trip deterministically, with no real-worker-thread timing to race.
  holdSearch?: boolean;
}

// A fake KomugiWorkerFactory: records every posted message per spawn, and
// answers init/search/ponderSettings itself (no real worker thread, no
// real engine) -- each spawned worker keeps its own ponder gating state,
// defaulted the same way the worker's own default is (DEFAULT_PONDER_SETTINGS
// enabled: true, k: 1; DEFAULT_PONDER_GATING hidden/discharging: false).
function makeFakePonderWorkerFactory(scripts: SpawnScript[] = [{}]): {
  factory: KomugiWorkerFactory;
  spawned: () => number;
  posted: () => unknown[][];
  releaseSearch: (spawnIndex: number) => void;
} {
  let spawnCount = 0;
  const postedBySpawn: unknown[][] = [];
  const releasers: Array<() => void> = [];

  function factory(): KomugiWorkerLike {
    const index = spawnCount;
    const script = scripts[Math.min(index, scripts.length - 1)] ?? {};
    spawnCount += 1;
    const postedHere: unknown[] = [];
    postedBySpawn.push(postedHere);
    releasers.push(() => undefined);

    const messageListeners: Array<(event: { data: unknown }) => void> = [];
    const errorListeners: Array<(event: unknown) => void> = [];
    const gating = { enabled: true, k: 1, hidden: false, discharging: false };

    function emit(data: unknown): void {
      for (const listener of messageListeners) listener({ data });
    }

    return {
      postMessage(message: unknown): void {
        postedHere.push(message);
        const m = message as { id: string; type?: string };
        if (m.type === 'init') {
          queueMicrotask(() => emit({ id: m.id, backend: 'wasm', threads: 1, batch: 1, probeRecord: {} }));
          return;
        }
        if (m.type === 'ponderSettings') {
          const req = message as { id: string; enabled?: boolean; k?: number; hidden?: boolean; discharging?: boolean };
          if (req.enabled !== undefined) gating.enabled = req.enabled;
          if (req.k !== undefined) gating.k = clampK(req.k);
          if (req.hidden !== undefined) gating.hidden = req.hidden;
          if (req.discharging !== undefined) gating.discharging = req.discharging;
          queueMicrotask(() => emit({ id: req.id, enabled: gating.enabled, k: gating.k, hidden: gating.hidden, discharging: gating.discharging }));
          return;
        }
        if (m.type === undefined) {
          // A bare search message.
          if (script.crashOnSearch === true) {
            queueMicrotask(() => {
              for (const listener of errorListeners) listener(new Error('test crash'));
            });
            return;
          }
          const reply = (): void => emit({ id: m.id, san: 'a', action: 0, stats: {} });
          if (script.holdSearch === true) {
            releasers[index] = reply;
          } else {
            globalThis.setTimeout(reply, 0);
          }
        }
        // 'stop' and anything else: no reply needed by these tests.
      },
      terminate(): void {},
      addEventListener(type: 'message' | 'error' | 'messageerror', listener: (event: { data: unknown } | unknown) => void): void {
        if (type === 'message') messageListeners.push(listener as (event: { data: unknown }) => void);
        else errorListeners.push(listener as (event: unknown) => void);
      },
    };
  }

  return {
    factory,
    spawned: () => spawnCount,
    posted: () => postedBySpawn,
    releaseSearch: (spawnIndex: number): void => releasers[spawnIndex]?.(),
  };
}

test('client_ponder (a): a ponder option posts exactly one ponderSettings message, after init and before the first search', async () => {
  const { factory, posted } = makeFakePonderWorkerFactory();
  const client = makeClient(factory, { ponder: { enabled: false, k: 2 } });
  try {
    const result = await client.play(REQUEST);
    assert.equal(result.status, 'move', `expected a move, got ${JSON.stringify(result)}`);

    const messages = posted()[0] ?? [];
    const ponderMessages = messages.filter(isPonderSettingsMessage);
    assert.equal(ponderMessages.length, 1, `expected exactly one ponderSettings message, got ${JSON.stringify(ponderMessages)}`);
    assert.equal(ponderMessages[0]?.enabled, false);
    assert.equal(ponderMessages[0]?.k, 2);

    const ponderIndex = messages.indexOf(ponderMessages[0]);
    const searchIndex = messages.findIndex(isSearchMessage);
    assert.ok(ponderIndex >= 0 && searchIndex >= 0 && ponderIndex < searchIndex, 'ponderSettings must be posted before the search');
  } finally {
    client.dispose();
  }
});

test('client_ponder (b): no ponder option posts zero ponderSettings messages across init + two plays', async () => {
  const { factory, posted, spawned } = makeFakePonderWorkerFactory();
  const client = makeClient(factory);
  try {
    const first = await client.play(REQUEST);
    assert.equal(first.status, 'move');
    const second = await client.play(REQUEST);
    assert.equal(second.status, 'move');
    assert.equal(spawned(), 1, 'the second play reuses the same worker');

    const messages = posted()[0] ?? [];
    assert.equal(messages.filter(isPonderSettingsMessage).length, 0);
  } finally {
    client.dispose();
  }
});

test('client_ponder (c): setPonder while ready posts the merged settings and resolves with the (clamped) reply', async () => {
  const { factory, posted } = makeFakePonderWorkerFactory();
  const client = makeClient(factory);
  try {
    const played = await client.play(REQUEST);
    assert.equal(played.status, 'move');
    assert.equal(client.state, 'ready');

    const result = await client.setPonder({ k: 7 });
    assert.deepEqual(result, { enabled: true, k: 3, hidden: false, discharging: false });

    const messages = posted()[0] ?? [];
    const ponderMessages = messages.filter(isPonderSettingsMessage);
    assert.equal(ponderMessages.length, 1);
    assert.equal(ponderMessages[0]?.k, 7, 'the wire message carries the unclamped k; the worker clamps it');
    assert.equal(ponderMessages[0]?.enabled, undefined, 'enabled was never stored, so it is absent from the message');
  } finally {
    client.dispose();
  }
});

test('client_ponder (d): setPonder before init resolves null; the next init posts the stored settings before the first search', async () => {
  const { factory, posted } = makeFakePonderWorkerFactory();
  const client = makeClient(factory);
  try {
    assert.equal(client.state, 'idle');
    const stored = await client.setPonder({ enabled: false });
    assert.equal(stored, null);

    const played = await client.play(REQUEST);
    assert.equal(played.status, 'move');

    const messages = posted()[0] ?? [];
    const ponderMessages = messages.filter(isPonderSettingsMessage);
    assert.equal(ponderMessages.length, 1);
    assert.equal(ponderMessages[0]?.enabled, false);

    const ponderIndex = messages.indexOf(ponderMessages[0]);
    const searchIndex = messages.findIndex(isSearchMessage);
    assert.ok(ponderIndex >= 0 && searchIndex >= 0 && ponderIndex < searchIndex, 'ponderSettings must be posted before the search');
  } finally {
    client.dispose();
  }
});

test('client_ponder (e): after a worker crash and respawn, the new worker gets the stored settings before its first search', async () => {
  const { factory, posted, spawned } = makeFakePonderWorkerFactory([{ crashOnSearch: true }, {}]);
  const client = makeClient(factory, { ponder: { enabled: false } });
  try {
    const first = await client.play(REQUEST);
    assert.equal(first.status, 'failed', `expected the crash to fail the first play, got ${JSON.stringify(first)}`);

    const second = await client.play(REQUEST);
    assert.equal(second.status, 'move', `expected the respawned worker to serve the second play, got ${JSON.stringify(second)}`);
    assert.equal(spawned(), 2, 'the crash must have triggered a respawn');

    const messages = posted()[1] ?? [];
    const ponderMessages = messages.filter(isPonderSettingsMessage);
    assert.equal(ponderMessages.length, 1, 'the respawned worker must receive the stored settings exactly once');
    assert.equal(ponderMessages[0]?.enabled, false);

    const ponderIndex = messages.indexOf(ponderMessages[0]);
    const searchIndex = messages.findIndex(isSearchMessage);
    assert.ok(
      ponderIndex >= 0 && searchIndex >= 0 && ponderIndex < searchIndex,
      "ponderSettings must be posted before the respawned worker's first search",
    );
  } finally {
    client.dispose();
  }
});

test('client_ponder (f): setPonder rejects invalid fields, naming the field, and posts nothing', async () => {
  const { factory, spawned } = makeFakePonderWorkerFactory();
  const client = makeClient(factory);
  try {
    await assert.rejects(client.setPonder({ k: NaN }), /k/);
    await assert.rejects(client.setPonder({ enabled: 'no' as never }), /enabled/);
    assert.equal(spawned(), 0, 'an invalid setPonder() must never spawn a worker or post anything');
  } finally {
    client.dispose();
  }
});

test('client_ponder (g): setPonder after dispose rejects', async () => {
  const { factory } = makeFakePonderWorkerFactory();
  const client = makeClient(factory);
  client.dispose();
  await assert.rejects(client.setPonder({ enabled: true }), /dispose/);
});

test('client_ponder (h): a ponderSettings reply arriving while a play is in flight resolves setPonder and leaves the play pending', async () => {
  const { factory, releaseSearch } = makeFakePonderWorkerFactory([{ holdSearch: true }]);
  const client = makeClient(factory);
  try {
    const playPromise = client.play(REQUEST);
    await waitFor(() => client.state === 'searching', 'searching');

    const ponderResult = await client.setPonder({ enabled: false, k: 2 });
    assert.deepEqual(ponderResult, { enabled: false, k: 2, hidden: false, discharging: false });
    assert.equal(client.state, 'searching', 'the play must still be pending when ponderSettings resolves');

    releaseSearch(0);
    const playResult = await playPromise;
    assert.equal(playResult.status, 'move', `expected the play to still resolve normally, got ${JSON.stringify(playResult)}`);
  } finally {
    client.dispose();
  }
});

// ---------------------------------------------------------------------------
// (i): end to end through the REAL worker host (installWorkerHost + a fake
// ORT module), following test/kom1226_host.test.ts's own fake-scope
// pattern -- wired to a KomugiWorkerLike adapter so a real KomugiClient
// drives it with no worker thread at all.
// ---------------------------------------------------------------------------

function makeFakeOrtModule(): OrtModule {
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
    env: { wasm: { numThreads: 1 } },
    InferenceSession: {
      async create(): Promise<OrtSession> {
        return {
          async run(): Promise<Record<string, { data: Float32Array }>> {
            return {};
          },
        };
      },
    },
    Tensor: FakeTensor,
  };
}

function makeInProcessWorkerFactory(hostOptions: WorkerHostOptions): KomugiWorkerFactory {
  return (): KomugiWorkerLike => {
    const clientListeners: Array<(event: { data: unknown }) => void> = [];
    const hostListeners: Array<(event: { data: unknown }) => void> = [];
    const scope = {
      addEventListener(_type: 'message', listener: (event: { data: unknown }) => void): void {
        hostListeners.push(listener);
      },
      postMessage(reply: unknown): void {
        for (const listener of clientListeners) listener({ data: reply });
      },
    };
    installWorkerHost(scope, hostOptions);
    return {
      postMessage(message: unknown): void {
        for (const listener of hostListeners) listener({ data: message });
      },
      terminate(): void {},
      addEventListener(type: 'message' | 'error' | 'messageerror', listener: (event: { data: unknown } | unknown) => void): void {
        if (type === 'message') clientListeners.push(listener as (event: { data: unknown }) => void);
      },
    };
  };
}

async function sha256Hex(bytes: Uint8Array): Promise<string> {
  const digest = await globalThis.crypto.subtle.digest('SHA-256', bytes);
  return Array.from(new Uint8Array(digest))
    .map((byte) => byte.toString(16).padStart(2, '0'))
    .join('');
}

test('client_ponder (i): end to end through the real worker host -- a ponder option + setPonder({})', async () => {
  const modelBytes = new Uint8Array([9, 8, 7, 6, 5, 4, 3, 2, 1]);
  const sha = await sha256Hex(modelBytes);
  const hostOptions: WorkerHostOptions = {
    loadOrt: async () => makeFakeOrtModule(),
    fetch: async (_url: string) => ({ ok: true, status: 200, arrayBuffer: async () => modelBytes.buffer as ArrayBuffer }),
  };
  const client = new KomugiClient({
    createWorker: makeInProcessWorkerFactory(hostOptions),
    model: { url: 'model.onnx', sha256: sha },
    ortAssets: { wasm: { mjs: 'wasm.mjs', wasm: 'wasm.wasm' } },
    ponder: { enabled: false },
  });
  try {
    const initResult = await client.init();
    assert.equal(initResult.status, 'ready', `expected init to succeed, got ${JSON.stringify(initResult)}`);

    const result = await client.setPonder({});
    assert.deepEqual(result, { enabled: false, k: 1, hidden: false, discharging: false });
  } finally {
    client.dispose();
  }
});
