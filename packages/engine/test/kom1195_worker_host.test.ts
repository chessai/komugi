// KOM-1195 unit U5 (docs/design/kom1195-analysis-api.md section 9 U5,
// section 3 AC2 cases b and c): `worker_host.ts`'s dispatch -- the listener
// opens the search's `StopTarget` SYNCHRONOUSLY, before `void
// handleSearch(...)` and so before `initChain`, so a `stop` arriving in that
// window (or right after a search is posted) is never lost; the `'stop'`
// branch replies once, immediately, with the search's own reply (if any)
// following later. Driven entirely through `installWorkerHost`'s fake
// `WorkerScopeLike`, never the engine directly -- that is what distinguishes
// these from kom1195_stop.test.ts's own (a)/(d)-(h) cases, which exercise
// `WorkerEngine` straight.

import assert from 'node:assert/strict';
import { test } from 'node:test';

import type { ProbeSession, SessionFactory } from '../src/backend.ts';
import type { MlhBatchEvaluator } from '../src/batch_evaluator.ts';
import { ACTION_SIZE, ADVANCED_POSITION } from '../src/constants.ts';
import type { WorkerInitEnv } from '../src/engine.ts';
import { makeFakeMlhEvaluator } from '../src/fake_evaluator.ts';
import type {
  WorkerAnalyzeRequest,
  WorkerInitRequest,
  WorkerOutgoingMessage,
  WorkerProgressEvent,
  WorkerRequest,
} from '../src/protocol.ts';
import { installWorkerHost } from '../src/worker_host.ts';
import type { WorkerHostOptions } from '../src/worker_host.ts';

// Same fake worker global as worker_host.test.ts's own `makeFakeScope`, plus
// `onPosted`: a list of hooks run synchronously, in `postMessage`'s own call,
// on every message posted -- lets a test react to a specific reply/progress
// event the instant it is posted, with no busy-polling loop racing the
// search's own real `setImmediate`/`MessageChannel` macrotask yields (U3/U4a).
function makeFakeScope(): {
  scope: { addEventListener: (type: 'message', listener: (event: { data: unknown }) => void) => void; postMessage: (reply: WorkerOutgoingMessage) => void };
  posted: WorkerOutgoingMessage[];
  send: (data: unknown) => void;
  onPosted: Array<(message: WorkerOutgoingMessage) => void>;
} {
  const listeners: Array<(event: { data: unknown }) => void> = [];
  const posted: WorkerOutgoingMessage[] = [];
  const onPosted: Array<(message: WorkerOutgoingMessage) => void> = [];
  const scope = {
    addEventListener(_type: 'message', listener: (event: { data: unknown }) => void): void {
      listeners.push(listener);
    },
    postMessage(reply: WorkerOutgoingMessage): void {
      posted.push(reply);
      for (const hook of onPosted) hook(reply);
    },
  };
  const send = (data: unknown): void => {
    for (const listener of listeners) listener({ data });
  };
  return { scope, posted, send, onPosted };
}

// Resolves with the first posted message matching `predicate`, including one
// that arrives synchronously before the next microtask -- the hook is
// registered BEFORE the caller's own `send`, so no message can be missed.
function waitFor(
  onPosted: Array<(message: WorkerOutgoingMessage) => void>,
  predicate: (message: WorkerOutgoingMessage) => boolean,
): Promise<WorkerOutgoingMessage> {
  return new Promise((resolve) => {
    onPosted.push((message) => {
      if (predicate(message)) resolve(message);
    });
  });
}

function isSearchReply(id: string | number): (message: WorkerOutgoingMessage) => boolean {
  return (message) => message.id === id && !('type' in message);
}

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

function fakeInitEnv(evaluator: MlhBatchEvaluator): WorkerInitEnv {
  return {
    sessionFactory: fakeSessionFactory(),
    createBatchBackend: async () => evaluator,
  };
}

// A synchronous fake evaluator -- fine for case (c), which is about the
// target being registered before `init` settles, not about the macrotask
// yield itself.
function fakeBatchEvaluator(): MlhBatchEvaluator {
  const leaf = makeFakeMlhEvaluator(ACTION_SIZE);
  return (states) => states.map((state) => leaf(state));
}

// A microtask-only evaluator (review F2/N1's own convention, kom1195_stop.ts
// F4): its promise resolves through microtasks alone, so without the
// `yieldToHost` macrotask yield a `stop` dispatched as a worker message would
// never be seen until the search already replied.
function microtaskOnlyEvaluator(): MlhBatchEvaluator {
  const leaf = makeFakeMlhEvaluator(ACTION_SIZE);
  return (states) => Promise.resolve(states.map((state) => leaf(state)));
}

const INIT_REQUEST: WorkerInitRequest = { id: 'init-1', type: 'init', modelUrl: 'model.onnx', modelSha256: 'sha-a' };

// ---- (b) stop latency, armed mid-search, through worker_host's dispatch ---

test('(b) a stop posted via worker_host mid-analysis lands within 2 chunks of the armed progress count', async () => {
  const evaluator = microtaskOnlyEvaluator();
  const options: WorkerHostOptions = { makeInitEnv: async () => fakeInitEnv(evaluator) };
  const { scope, posted, send, onPosted } = makeFakeScope();
  installWorkerHost(scope, options);

  const initReplied = waitFor(onPosted, (m) => m.id === INIT_REQUEST.id);
  send(INIT_REQUEST);
  await initReplied;

  const SEARCH_ID = 'search-b';
  const BATCH = 1;
  const request: WorkerAnalyzeRequest = {
    id: SEARCH_ID,
    fen: ADVANCED_POSITION,
    analyze: true,
    nodeCeiling: 20_001,
    batch: BATCH,
    progressEveryPlayouts: 1,
  };

  let armed = false;
  let progressCount = 0;
  let k = -1;
  const searchReplied = waitFor(onPosted, isSearchReply(SEARCH_ID));
  onPosted.push((message) => {
    if (armed || !('type' in message) || message.type !== 'progress') return;
    progressCount += 1;
    if (progressCount < 3) return;
    armed = true;
    const lastPlayouts = (message as WorkerProgressEvent).playouts;
    setTimeout(() => {
      k = lastPlayouts;
      send({ id: 'stop-b', type: 'stop', target: SEARCH_ID });
    }, 0);
  });

  send(request);
  const searchReply = await searchReplied;

  assert.ok(armed, 'expected at least 3 progress events before the search replied');
  assert.ok('stats' in searchReply, `expected a successful search reply, got ${JSON.stringify(searchReply)}`);
  if (!('stats' in searchReply)) return;
  const stats = searchReply.stats as { deadlineHit: boolean; stopReason?: string; playouts: number };
  assert.equal(stats.deadlineHit, true, JSON.stringify(stats));
  assert.equal(stats.stopReason, 'stop', JSON.stringify(stats));
  assert.ok(
    stats.playouts - k <= 2 * BATCH,
    `expected the stop to land within 2 chunks: playouts ${stats.playouts}, armed at ${k}`,
  );

  const stopReply = posted.find((m) => m.id === 'stop-b');
  assert.ok(stopReply !== undefined && 'acknowledged' in stopReply && stopReply.acknowledged === true, JSON.stringify(stopReply));
});

// ---- (c) early registration: a stop sent before init settles still lands --

test("(c) a stop posted right after the search, while init is still gated, is acknowledged immediately and the search itself stops within one chunk", async () => {
  const evaluator = fakeBatchEvaluator();
  let releaseInit: () => void = () => undefined;
  const gate = new Promise<void>((resolve) => {
    releaseInit = resolve;
  });
  const options: WorkerHostOptions = {
    makeInitEnv: async () => {
      await gate;
      return fakeInitEnv(evaluator);
    },
  };
  const { scope, posted, send, onPosted } = makeFakeScope();
  installWorkerHost(scope, options);

  const SEARCH_ID = 'search-c';
  const BATCH = 4;
  const request: WorkerRequest = { id: SEARCH_ID, fen: ADVANCED_POSITION, sims: 400, batch: BATCH, stoppable: true };
  const searchReplied = waitFor(onPosted, isSearchReply(SEARCH_ID));

  // `init` is sent first but never settles until `releaseInit()` below --
  // the search's `StopTarget` must still be open the instant `stop` is
  // dispatched, synchronously, with no await in between.
  send(INIT_REQUEST);
  send(request);
  send({ id: 'stop-c', type: 'stop', target: SEARCH_ID });

  const stopReply = posted.find((m) => m.id === 'stop-c');
  assert.ok(
    stopReply !== undefined && 'acknowledged' in stopReply && stopReply.acknowledged === true,
    `expected an immediate ack while init is still gated, got ${JSON.stringify(posted)}`,
  );
  // Proof the search's own reply has not arrived yet -- `init` is still
  // gated, so `handleSearch` is still awaiting `initChain`.
  assert.ok(!posted.some(isSearchReply(SEARCH_ID)), 'the search must not have replied yet');

  releaseInit();
  const searchReply = await searchReplied;

  assert.ok('stats' in searchReply, `expected a successful search reply, got ${JSON.stringify(searchReply)}`);
  if (!('stats' in searchReply)) return;
  const stats = searchReply.stats as { deadlineHit: boolean; stopReason?: string; playouts: number };
  assert.equal(stats.deadlineHit, true, JSON.stringify(stats));
  assert.equal(stats.stopReason, 'stop', JSON.stringify(stats));
  assert.ok(stats.playouts <= BATCH, `expected playouts <= ${BATCH}, got ${stats.playouts}`);
});
