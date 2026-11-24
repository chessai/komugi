// KOM-1221 unit 3 (design docs/design/kom1221-release-ci.md section 4,
// section 5.5): a test-only worker entry, bundled into the mock site
// alongside the real worker.js but never shipped. Same
// WorkerGlobalScope/em-pthread guard and `installWorkerHost` call
// packages/engine/src/worker.ts's real entry makes, plus one addition: a
// second `message` listener (addEventListener composes, never clobbers)
// that reacts to a message with NO `id` field by throwing a real uncaught
// error. That shape is deliberately safe for the real host:
// worker_host.ts's `isEngineMessage` guard (worker_host.ts:287) silently
// drops any id-less message, and its "unknown message type" `err` reply
// (worker_host.ts:304-305) fires only when an `id` is present -- so the
// real host neither handles nor answers this message, only this listener
// does. `page.ts`'s `crashSequence()` posts it mid-flight to force a
// worker crash (client.ts's `error` listener) without ever touching the
// real protocol.

import { installWorkerHost } from '../../../../packages/engine/dist/worker_host.js';
import type { WorkerScopeLike } from '../../../../packages/engine/dist/worker_host.js';

const WGS = (globalThis as { WorkerGlobalScope?: abstract new (...args: never[]) => unknown }).WorkerGlobalScope;

if (
  WGS !== undefined &&
  typeof self !== 'undefined' &&
  self instanceof WGS &&
  (self as { name?: string }).name !== 'em-pthread'
) {
  installWorkerHost(self as unknown as WorkerScopeLike);

  self.addEventListener('message', (event: MessageEvent) => {
    const data = event.data as unknown;
    if (typeof data !== 'object' || data === null || 'id' in data) return;
    setTimeout(() => {
      throw new Error('forced');
    });
  });
}
