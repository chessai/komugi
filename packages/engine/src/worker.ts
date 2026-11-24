// KOM-1185 unit 4b1 (design docs/design/ts-engine-library.md section 4.2
// "The split"): the real worker entry -- installs `worker_host.ts` only
// inside an actual, non-pthread worker global scope, so importing this
// module (its types, a jsdom test, a page's main thread) installs nothing,
// and an Emscripten pthread's own `self.onmessage` (KOM-1208: `self.name
// === 'em-pthread'`) is never clobbered.

import { installWorkerHost } from './worker_host.ts';
import type { WorkerScopeLike } from './worker_host.ts';

// `WorkerGlobalScope` is a DOM lib *type*, not a global value this
// tsconfig's `lib` exposes -- declared locally, just enough to check
// `self instanceof WorkerGlobalScope` without changing `tsconfig.json`.
const WGS = (globalThis as { WorkerGlobalScope?: abstract new (...args: never[]) => unknown }).WorkerGlobalScope;

if (
  WGS !== undefined &&
  typeof self !== 'undefined' &&
  self instanceof WGS &&
  (self as { name?: string }).name !== 'em-pthread'
) {
  installWorkerHost(self as unknown as WorkerScopeLike);
}
