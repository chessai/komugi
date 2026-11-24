// KOM-1199 unit U2 (design docs/design/kom1199-client.md section 6, 7):
// exercises the Node worker harness itself -- raw protocol messages
// through `nodeWorkerFactory`'s adapter, with no `KomugiClient` yet
// (that's U3).

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { ADVANCED_POSITION } from '../src/constants.ts';
import { nodeWorkerFactory } from './support/node_worker_adapter.ts';
import type { KomugiWorkerLike } from './support/node_worker_adapter.ts';

const INIT_REQUEST = { id: 'i1', type: 'init' as const, modelUrl: 'model.onnx', modelSha256: 'sha-a' };
const SEARCH_REQUEST = { id: 's1', fen: ADVANCED_POSITION, sims: 4, batch: 1 };

// Every scenario below sends one request at a time and awaits its reply
// before sending the next, so "the next message event" is always the
// reply to the request just posted -- no id matching needed.
function onceMessage(worker: KomugiWorkerLike): Promise<Record<string, unknown>> {
  return new Promise((resolve) => {
    worker.addEventListener('message', (event) => resolve(event.data as Record<string, unknown>));
  });
}

async function waitUntil(predicate: () => boolean, timeoutMs = 5000): Promise<void> {
  const start = Date.now();
  while (!predicate()) {
    if (Date.now() - start > timeoutMs) throw new Error('timed out waiting for condition');
    await new Promise((resolve) => setTimeout(resolve, 10));
  }
}

test('client_harness mode ok: init then search through the adapter succeed', async () => {
  const { factory, spawned } = nodeWorkerFactory(['ok']);
  const worker = factory();
  try {
    const initReplyP = onceMessage(worker);
    worker.postMessage(INIT_REQUEST);
    const initReply = await initReplyP;
    assert.equal(initReply.id, 'i1');
    assert.ok(!('err' in initReply), `init should succeed, got ${JSON.stringify(initReply)}`);

    const searchReplyP = onceMessage(worker);
    worker.postMessage(SEARCH_REQUEST);
    const searchReply = await searchReplyP;
    assert.equal(searchReply.id, 's1');
    assert.equal(typeof searchReply.san, 'string');
    assert.equal(typeof searchReply.action, 'number');
    assert.equal(spawned(), 1);
  } finally {
    worker.terminate();
  }
});

test('client_harness mode crash-on-search: error fires exactly once, no search reply arrives', async () => {
  const { factory } = nodeWorkerFactory(['crash-on-search']);
  const worker = factory();
  try {
    const initReplyP = onceMessage(worker);
    worker.postMessage(INIT_REQUEST);
    await initReplyP;

    let errorCount = 0;
    let messageCount = 0;
    worker.addEventListener('error', () => {
      errorCount += 1;
    });
    worker.addEventListener('message', () => {
      messageCount += 1;
    });

    worker.postMessage(SEARCH_REQUEST);
    await waitUntil(() => errorCount >= 1);
    // Give any stray search reply a chance to arrive before asserting none did.
    await new Promise((resolve) => setTimeout(resolve, 50));

    assert.equal(errorCount, 1);
    assert.equal(messageCount, 0);
  } finally {
    worker.terminate();
  }
});

test('client_harness mode init-model-fetch: init reply carries the model-fetch code', async () => {
  const { factory } = nodeWorkerFactory(['init-model-fetch']);
  const worker = factory();
  try {
    const initReplyP = onceMessage(worker);
    worker.postMessage(INIT_REQUEST);
    const initReply = await initReplyP;
    assert.equal(initReply.code, 'model-fetch');
  } finally {
    worker.terminate();
  }
});

test('client_harness terminate(): no error listener fires', async () => {
  const { factory } = nodeWorkerFactory(['ok']);
  const worker = factory();
  const initReplyP = onceMessage(worker);
  worker.postMessage(INIT_REQUEST);
  await initReplyP;

  let errorCount = 0;
  worker.addEventListener('error', () => {
    errorCount += 1;
  });
  worker.terminate();
  await new Promise((resolve) => setTimeout(resolve, 100));

  assert.equal(errorCount, 0);
});
