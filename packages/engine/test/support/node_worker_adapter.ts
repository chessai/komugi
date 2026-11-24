// KOM-1199 unit U2 (design docs/design/kom1199-client.md section 6): a
// `KomugiWorkerFactory` backed by a real `node:worker_threads` thread
// running `client_test_worker.ts`. `posted()[k]` is every message the
// main thread posted to spawn `k`, in order (KOM-1214: lets a test read
// the restart's init request).

import { Worker } from 'node:worker_threads';

import type { KomugiWorkerFactory, KomugiWorkerLike } from '../../src/client.ts';
import type { TestWorkerMode } from './client_test_worker.ts';

export type { KomugiWorkerFactory, KomugiWorkerLike };

const WORKER_PATH = new URL('./client_test_worker.ts', import.meta.url);

export function nodeWorkerFactory(modes: TestWorkerMode[]): {
  factory: KomugiWorkerFactory;
  spawned: () => number;
  posted: () => unknown[][];
} {
  let spawnCount = 0;
  const postedBySpawn: unknown[][] = [];

  function factory(): KomugiWorkerLike {
    const mode = modes[Math.min(spawnCount, modes.length - 1)];
    spawnCount += 1;
    const postedHere: unknown[] = [];
    postedBySpawn.push(postedHere);

    const worker = new Worker(WORKER_PATH, { workerData: { mode } });
    let terminating = false;
    // Set by the first 'error'/'messageerror' so the 'exit' that always
    // follows an uncaught-exception crash does not report the SAME crash
    // to error listeners a second time.
    let crashed = false;

    const messageListeners: Array<(event: { data: unknown }) => void> = [];
    const errorListeners: Array<(event: unknown) => void> = [];

    worker.on('message', (data: unknown) => {
      for (const listener of messageListeners) listener({ data });
    });
    worker.on('error', (error: unknown) => {
      crashed = true;
      for (const listener of errorListeners) listener(error);
    });
    worker.on('messageerror', (error: unknown) => {
      crashed = true;
      for (const listener of errorListeners) listener(error);
    });
    worker.on('exit', () => {
      if (terminating || crashed) return;
      for (const listener of errorListeners) listener(new Error('worker exited unexpectedly'));
    });

    return {
      postMessage(message: unknown): void {
        postedHere.push(message);
        worker.postMessage(message);
      },
      terminate(): void {
        terminating = true;
        void worker.terminate();
      },
      addEventListener(type: 'message' | 'error' | 'messageerror', listener: (event: { data: unknown } | unknown) => void): void {
        if (type === 'message') messageListeners.push(listener as (event: { data: unknown }) => void);
        else errorListeners.push(listener);
      },
    };
  }

  return { factory, spawned: () => spawnCount, posted: () => postedBySpawn };
}
