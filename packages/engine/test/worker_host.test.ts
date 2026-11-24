// KOM-1185 unit 4b1 (design docs/design/ts-engine-library.md section 4.2
// "The split", section 7.1's extra F2 test): `installWorkerHost`'s FIFO
// promise chain and typed-error replies, driven with a fake `WorkerScopeLike`
// and a fake `WorkerInitEnv` (via `options.makeInitEnv`) -- no real
// onnxruntime-web, no network, no browser. All three scenarios would be RED
// against a host that still read `currentBatchEvaluator ??
// defaultBatchEvaluator()` eagerly at message-arrival time.

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { ACTION_SIZE, ADVANCED_POSITION } from '../src/constants.ts';
import type { WorkerInitEnv } from '../src/engine.ts';
import type { ProbeSession, SessionFactory } from '../src/backend.ts';
import type { MlhBatchEvaluator } from '../src/batch_evaluator.ts';
import { makeFakeMlhEvaluator } from '../src/fake_evaluator.ts';
import type {
  WorkerInitReply,
  WorkerInitRequest,
  WorkerPonderNowReply,
  WorkerPonderSettingsReply,
  WorkerReply,
  WorkerRequest,
} from '../src/protocol.ts';
import { installWorkerHost } from '../src/worker_host.ts';
import type { WorkerHostOptions } from '../src/worker_host.ts';

type Posted = WorkerReply | WorkerInitReply | WorkerPonderSettingsReply | WorkerPonderNowReply;

// A fake worker global: `send` plays the role of a real `postMessage`
// arriving at the worker, dispatching to every registered listener
// synchronously, exactly like a real `message` event does for one posted
// message at a time.
function makeFakeScope(): { scope: { addEventListener: (type: 'message', listener: (event: { data: unknown }) => void) => void; postMessage: (reply: Posted) => void }; posted: Posted[]; send: (data: unknown) => void } {
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

// A no-op session: `probeBackend` only ever calls `warmup`/`run` to time a
// backend, never inspects what they actually do.
function fakeProbeSession(): ProbeSession {
  return { warmup: async () => undefined, run: async () => undefined };
}

function fakeSessionFactory(): SessionFactory {
  return {
    setNumThreads: () => undefined,
    createWebgpu: async () => fakeProbeSession(),
    createWasm: async () => fakeProbeSession(),
  };
}

// The "real injected evaluator" `init` installs -- distinguishable from the
// engine's own internal fake-leaf default purely by having its own call
// counter a test can read, and never reachable except through a search that
// actually consulted `engine.currentBatchEvaluator` after a successful init.
function spyBatchEvaluator(): { evaluator: MlhBatchEvaluator; calls: () => number } {
  const leaf = makeFakeMlhEvaluator(ACTION_SIZE);
  let calls = 0;
  const evaluator: MlhBatchEvaluator = (states) => {
    calls += 1;
    return states.map((state) => leaf(state));
  };
  return { evaluator, calls: () => calls };
}

function fakeInitEnv(evaluator: MlhBatchEvaluator): WorkerInitEnv {
  return {
    sessionFactory: fakeSessionFactory(),
    createBatchBackend: async () => evaluator,
  };
}

const INIT_REQUEST: WorkerInitRequest = { id: 'init-1', type: 'init', modelUrl: 'model.onnx', modelSha256: 'sha-a' };
const SEARCH_REQUEST: WorkerRequest = { id: 'search-1', fen: ADVANCED_POSITION, sims: 4, batch: 1 };

function isOk(reply: Posted): reply is WorkerReply & { stats: { backend?: string } } {
  return !('err' in reply);
}

// None of the fakes above ever waits on a real timer -- every `init`/search
// settles after a bounded number of microtasks. Polling `Promise.resolve()`
// (never a real timer, never a sleep) until `posted` reaches the expected
// length is the deterministic way to wait for an exact count of fire-and-
// forget async handlers without hardcoding how many microtask ticks the
// FIFO chain + `probeBackend`'s own awaits happen to take.
async function flushUntil(predicate: () => boolean, maxTicks = 10_000): Promise<void> {
  for (let i = 0; i < maxTicks; i++) {
    if (predicate()) return;
    await Promise.resolve();
  }
}

test('worker_host init-race: a search right after init waits for it, then uses the real injected evaluator', async () => {
  const { evaluator, calls } = spyBatchEvaluator();
  const options: WorkerHostOptions = { makeInitEnv: async () => fakeInitEnv(evaluator) };
  const { scope, posted, send } = makeFakeScope();
  installWorkerHost(scope, options);

  send(INIT_REQUEST);
  send(SEARCH_REQUEST);
  // Both handlers were dispatched synchronously above; the search's own
  // async work (awaiting the FIFO chain) hasn't settled yet.
  await flushUntil(() => posted.length >= 2);

  assert.equal(posted.length, 2, `expected an init reply and a search reply, got ${JSON.stringify(posted)}`);
  const [initReply, searchReply] = posted;
  assert.ok(initReply !== undefined && !('err' in initReply), `init should succeed, got ${JSON.stringify(initReply)}`);
  assert.ok(searchReply !== undefined && isOk(searchReply), `search should succeed, got ${JSON.stringify(searchReply)}`);
  if (searchReply === undefined || !isOk(searchReply)) return;
  assert.equal(searchReply.stats.backend, 'wasm', 'a successful search after init stamps the negotiated backend');
  assert.ok(calls() > 0, 'the real injected evaluator must actually have run the search');
});

test('worker_host no-init: a search with no prior init gets not-initialized and never runs a search', async () => {
  const { scope, posted, send } = makeFakeScope();
  installWorkerHost(scope);

  send(SEARCH_REQUEST);
  await flushUntil(() => posted.length >= 1);

  assert.equal(posted.length, 1, `expected exactly one reply, got ${JSON.stringify(posted)}`);
  const [reply] = posted;
  assert.ok(reply !== undefined && 'err' in reply, `expected an error reply, got ${JSON.stringify(reply)}`);
  if (reply === undefined || !('err' in reply)) return;
  assert.equal(reply.code, 'not-initialized');
  // No `stats`/`action`/`san` at all: proof the fake evaluator's
  // deterministic move was never produced, let alone returned.
  assert.ok(!('stats' in reply) && !('action' in reply), 'a not-initialized reply must never carry a search result');
});

test('worker_host failed-init: a rejected init relays its own code to a following search', async () => {
  const modelFetchError = new Error('fetch failed: 404') as Error & { code: string };
  modelFetchError.code = 'model-fetch';
  const options: WorkerHostOptions = { makeInitEnv: async () => Promise.reject(modelFetchError) };
  const { scope, posted, send } = makeFakeScope();
  installWorkerHost(scope, options);

  send(INIT_REQUEST);
  send(SEARCH_REQUEST);
  await flushUntil(() => posted.length >= 2);

  assert.equal(posted.length, 2, `expected an init-failure reply and a search reply, got ${JSON.stringify(posted)}`);
  const [initReply, searchReply] = posted;
  assert.ok(initReply !== undefined && 'err' in initReply, `init should fail, got ${JSON.stringify(initReply)}`);
  if (initReply === undefined || !('err' in initReply)) return;
  assert.equal(initReply.code, 'model-fetch');

  assert.ok(searchReply !== undefined && 'err' in searchReply, `search should fail too, got ${JSON.stringify(searchReply)}`);
  if (searchReply === undefined || !('err' in searchReply)) return;
  assert.equal(searchReply.code, 'model-fetch', "a search after a failed init relays that init's own code");
});

// Review NIT (i) (4b1 review): dispatch on `type` must be explicit --
// 'init'/'ponderSettings'/'ponderNow' go to their own handlers, and any
// OTHER `type` (isEngineMessage only checks `id`) gets its own `{ id, err }`
// reply instead of silently falling through to `handlePonderNow`.
test('worker_host dispatch: an unrecognised type replies with its own error, never falls through to ponderNow', async () => {
  const { scope, posted, send } = makeFakeScope();
  installWorkerHost(scope);

  send({ id: 'weird-1', type: 'bogus' });
  await flushUntil(() => posted.length >= 1);

  assert.equal(posted.length, 1, `expected exactly one reply, got ${JSON.stringify(posted)}`);
  const [reply] = posted;
  assert.ok(reply !== undefined && 'err' in reply, `expected an error reply, got ${JSON.stringify(reply)}`);
  if (reply === undefined || !('err' in reply)) return;
  assert.equal(reply.id, 'weird-1');
  assert.ok(reply.err.includes('bogus'), `error message should name the unknown type, got ${JSON.stringify(reply)}`);
  // Never the ponderNow shape (`started`/`reason`) -- proof it was never
  // misrouted to `handlePonderNow`.
  assert.ok(!('started' in reply), 'an unrecognised type must never be answered as a ponderNow reply');
  assert.ok(!('code' in reply) || reply.code === undefined, "worker_host's own unknown-type reply carries no code");
});

// Review NIT (ii) (4b1 review): the FIFO chain already allows a successful
// init after a failed one to make later searches succeed -- this proves it
// with a test instead of leaving it as an unverified claim.
test('worker_host init-recovery: a successful init after a failed one makes a following search succeed', async () => {
  const { evaluator, calls } = spyBatchEvaluator();
  let initCalls = 0;
  const options: WorkerHostOptions = {
    makeInitEnv: async () => {
      initCalls += 1;
      if (initCalls === 1) {
        const error = new Error('model fetch failed: 404') as Error & { code: string };
        error.code = 'model-fetch';
        throw error;
      }
      return fakeInitEnv(evaluator);
    },
  };
  const { scope, posted, send } = makeFakeScope();
  installWorkerHost(scope, options);

  send({ ...INIT_REQUEST, id: 'init-fail' });
  await flushUntil(() => posted.length >= 1);
  const [firstInitReply] = posted;
  assert.ok(firstInitReply !== undefined && 'err' in firstInitReply, `the first init should fail, got ${JSON.stringify(firstInitReply)}`);

  send({ ...INIT_REQUEST, id: 'init-ok' });
  send({ ...SEARCH_REQUEST, id: 'search-after-recovery' });
  await flushUntil(() => posted.length >= 3);

  assert.equal(posted.length, 3, `expected two more replies, got ${JSON.stringify(posted)}`);
  const [, secondInitReply, searchReply] = posted;
  assert.ok(secondInitReply !== undefined && !('err' in secondInitReply), `the second init should succeed, got ${JSON.stringify(secondInitReply)}`);
  assert.ok(searchReply !== undefined && isOk(searchReply), `the search after recovery should succeed, got ${JSON.stringify(searchReply)}`);
  if (searchReply === undefined || !isOk(searchReply)) return;
  assert.equal(searchReply.stats.backend, 'wasm');
  assert.ok(calls() > 0, 'the search after recovery must have run the real injected evaluator');
});
