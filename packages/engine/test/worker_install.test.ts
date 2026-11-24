// KOM-1185 unit 4b1 (design docs/design/ts-engine-library.md section 4.2
// "The split", section 7.1 AC4): the worker entry installs nothing outside
// a real, non-pthread worker scope. Each case imports a FRESH instance of
// `src/worker.ts` (a unique `?case=` query busts Node's ESM module cache --
// no `node:vm`/child-process needed, since the module itself never caches
// anything beyond its own one-time top-level guard) against a `globalThis`
// shaped for that case, then cleans the shape back up. Cases 1-2 below must
// be RED against a `worker.ts` that installs unconditionally (today's
// pre-unit-4b `self.onmessage = ...`) and GREEN after the guarded rewrite.

import assert from 'node:assert/strict';
import { test } from 'node:test';

type Listener = (event: { data: unknown }) => void;

interface FakeSelfLike {
  name: string;
  postMessage: (message: unknown) => void;
  addEventListener: (type: string, listener: Listener) => void;
}

class FakeWorkerGlobalScope {}

function makeFakeSelf(name: string, addEventListenerCalls: Array<{ type: string }>): FakeSelfLike & FakeWorkerGlobalScope {
  const self = Object.assign(new FakeWorkerGlobalScope(), {
    name,
    postMessage: (_message: unknown) => undefined,
    addEventListener: (type: string, _listener: Listener) => {
      addEventListenerCalls.push({ type });
    },
  });
  return self;
}

// `globalThis.self`/`globalThis.WorkerGlobalScope`/`globalThis.onmessage`
// are each case's own shape -- deleted again afterwards so one case's realm
// mutation never leaks into the next (all four run in the same module
// realm, sequentially).
function resetGlobals(): void {
  delete (globalThis as { self?: unknown }).self;
  delete (globalThis as { WorkerGlobalScope?: unknown }).WorkerGlobalScope;
  delete (globalThis as { onmessage?: unknown }).onmessage;
}

let caseCounter = 0;
async function importFreshWorker(): Promise<void> {
  caseCounter += 1;
  await import(`../src/worker.ts?case=${caseCounter}`);
}

test('plain Node realm: importing worker.ts and worker_host.ts installs nothing', async () => {
  resetGlobals();
  try {
    assert.equal(typeof (globalThis as { self?: unknown }).self, 'undefined');
    caseCounter += 1;
    await import(`../src/worker.ts?case=${caseCounter}`);
    caseCounter += 1;
    await import(`../src/worker_host.ts?case=${caseCounter}`);
    assert.equal((globalThis as { onmessage?: unknown }).onmessage, undefined);
    assert.equal(typeof (globalThis as { self?: unknown }).self, 'undefined', 'importing either module must not define self');
  } finally {
    resetGlobals();
  }
});

test('main-thread-like realm (self defined, no WorkerGlobalScope): installs nothing', async () => {
  resetGlobals();
  try {
    const addEventListenerCalls: Array<{ type: string }> = [];
    (globalThis as { self?: unknown }).self = {
      postMessage: (_message: unknown) => undefined,
      addEventListener: (type: string, _listener: Listener) => {
        addEventListenerCalls.push({ type });
      },
    };
    await importFreshWorker();
    assert.deepEqual(addEventListenerCalls, []);
    assert.equal((globalThis as { onmessage?: unknown }).onmessage, undefined);
  } finally {
    resetGlobals();
  }
});

test("em-pthread worker realm: installs nothing (KOM-1208's pthread guard)", async () => {
  resetGlobals();
  try {
    const addEventListenerCalls: Array<{ type: string }> = [];
    (globalThis as { WorkerGlobalScope?: unknown }).WorkerGlobalScope = FakeWorkerGlobalScope;
    (globalThis as { self?: unknown }).self = makeFakeSelf('em-pthread', addEventListenerCalls);
    await importFreshWorker();
    assert.deepEqual(addEventListenerCalls, []);
    assert.equal((globalThis as { onmessage?: unknown }).onmessage, undefined);
  } finally {
    resetGlobals();
  }
});

test('real worker realm (any other self.name): installs exactly one message listener, leaves onmessage unset', async () => {
  resetGlobals();
  try {
    const addEventListenerCalls: Array<{ type: string }> = [];
    (globalThis as { WorkerGlobalScope?: unknown }).WorkerGlobalScope = FakeWorkerGlobalScope;
    const fakeSelf = makeFakeSelf('worker-1', addEventListenerCalls);
    (globalThis as { self?: unknown }).self = fakeSelf;
    await importFreshWorker();
    assert.equal(addEventListenerCalls.length, 1, `expected exactly one addEventListener call, got ${JSON.stringify(addEventListenerCalls)}`);
    assert.equal(addEventListenerCalls[0]?.type, 'message');
    assert.equal((globalThis as { onmessage?: unknown }).onmessage, undefined);
    assert.equal((fakeSelf as { onmessage?: unknown }).onmessage, undefined);
  } finally {
    resetGlobals();
  }
});
