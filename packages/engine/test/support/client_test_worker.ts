// KOM-1199 unit U2 (design docs/design/kom1199-client.md section 6): runs
// inside a real node:worker_threads thread, hosting `installWorkerHost`
// with a fake init env (no real onnxruntime-web/network) and, for every
// mode but 'ok', intercepting the search message before the host's own
// listener sees it.

import { parentPort, workerData } from 'node:worker_threads';

import { ACTION_SIZE } from '../../src/constants.ts';
import type { ProbeSession, SessionFactory } from '../../src/backend.ts';
import type { MlhBatchEvaluator } from '../../src/batch_evaluator.ts';
import type { WorkerInitEnv } from '../../src/engine.ts';
import { makeFakeMlhEvaluator } from '../../src/fake_evaluator.ts';
import { installWorkerHost } from '../../src/worker_host.ts';
import type { WorkerScopeLike } from '../../src/worker_host.ts';

export type TestWorkerMode =
  | 'ok'
  | 'init-model-fetch'
  | 'init-sha-mismatch'
  | 'drop-search'
  | 'crash-on-search'
  | 'garbage-reply'
  | 'uncoded-search-error'
  // KOM-1214: replies to init with a typed threads-timeout error carrying
  // a probeRecord (engine.ts's ThreadsTimeoutError reply shape). Put it
  // first in a modes list (`['init-threads-timeout', 'ok']`) so only the
  // first spawn times out.
  | 'init-threads-timeout';

// The probeRecord the 'init-threads-timeout' mode replies with.
export const THREADS_TIMEOUT_PROBE_RECORD = {
  backend: 'wasm',
  threads: 1,
  msPerPosition: {},
  chosenBatch: 1,
  probedAt: 1234,
  key: 'test-threads-timeout',
} as const;

if (parentPort === null) {
  throw new Error('client_test_worker.ts must run inside a worker thread');
}
const port = parentPort;
const mode = (workerData as { mode: TestWorkerMode }).mode;

// `test/worker_host.test.ts`'s `fakeSessionFactory()`/`fakeProbeSession()`
// shape, duplicated here on purpose (review F4): that file's helper is
// private, and a third copy should move to a shared `test/support/` helper
// instead of a second one.
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

function fakeBatchEvaluator(): MlhBatchEvaluator {
  const leaf = makeFakeMlhEvaluator(ACTION_SIZE);
  return (states) => states.map((state) => leaf(state));
}

function typedError(code: string, message: string): Error & { code: string } {
  const error = new Error(message) as Error & { code: string };
  error.code = code;
  return error;
}

async function makeInitEnv(): Promise<WorkerInitEnv> {
  if (mode === 'init-model-fetch') throw typedError('model-fetch', 'test model fetch failure');
  if (mode === 'init-sha-mismatch') throw typedError('sha-mismatch', 'test sha mismatch');
  return { sessionFactory: fakeSessionFactory(), createBatchBackend: async () => fakeBatchEvaluator() };
}

type MessageListener = (event: { data: unknown }) => void;
const messageListeners: MessageListener[] = [];

// A plain search request has no `type` field at all (protocol.ts's
// `WorkerRequest`); every init/ponder message does -- the same
// distinction `worker_host.ts`'s own dispatch uses.
function isSearchMessage(data: unknown): boolean {
  return typeof data === 'object' && data !== null && !('type' in data);
}

const scope: WorkerScopeLike = {
  addEventListener(_type: 'message', listener: MessageListener): void {
    messageListeners.push(listener);
  },
  postMessage(reply): void {
    port.postMessage(reply);
  },
};

port.on('message', (data: unknown) => {
  if (mode === 'init-threads-timeout' && typeof data === 'object' && data !== null && (data as { type?: unknown }).type === 'init') {
    const id = (data as { id: string | number }).id;
    port.postMessage({ id, err: 'test threads timeout', code: 'threads-timeout', probeRecord: THREADS_TIMEOUT_PROBE_RECORD });
    return;
  }
  if (isSearchMessage(data)) {
    const id = (data as { id: string | number }).id;
    switch (mode) {
      case 'drop-search':
        return; // swallowed -> the client's own search timeout fires
      case 'crash-on-search':
        // Uncaught inside the worker: Node emits 'error' then 'exit' on
        // the main-thread Worker handle.
        setImmediate(() => {
          throw new Error('test crash');
        });
        return;
      case 'garbage-reply':
        port.postMessage({ id, san: 42 });
        return;
      case 'uncoded-search-error':
        port.postMessage({ id, err: 'boom' });
        return;
      default:
        break;
    }
  }
  for (const listener of messageListeners) listener({ data });
});

installWorkerHost(scope, { makeInitEnv });
