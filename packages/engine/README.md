# @komugi-ai/engine

The Komugi Gungi engine for the browser: board/move rules, MCTS-based
search, and ONNX neural-network evaluation running inside a dedicated web
worker.

## Install

```sh
npm install @komugi-ai/engine
```

`onnxruntime-web` is an optional peer dependency, needed only if you use
the neural search (the worker entry or the ORT helpers on the core entry).
If you only need the rules entry, no peer dependency is required.

## Entry points

- **`@komugi-ai/engine`** — the core entry: the full public surface,
  including the worker-host builder (`installWorkerHost`), the
  `WorkerEngine` class, backend-selection and ORT wiring helpers, and
  everything the rules entry exports (re-exported wholesale, so importing
  the core entry alone is enough).
- **`@komugi-ai/engine/rules`** — a rules-only entry: board representation,
  FEN parsing, move generation, move application, SAN, and terminal/
  repetition checks, with no search or evaluator code reachable from it.
  Use this when you only need to validate or enumerate moves (e.g. a
  non-search UI, or a server that just wants the rules), with nothing to
  install and no peer dependency.
- **`@komugi-ai/engine/worker`** — a thin entry meant to be loaded as the
  body of a dedicated worker. Importing it installs the message handler
  automatically, but only inside a real dedicated-worker scope; it is a
  no-op anywhere else (including inside an Emscripten pthread worker).

A rules-only import, with no worker and no model:

```ts
import { boardFromFen, Game, RepetitionRule } from '@komugi-ai/engine/rules';

const game = new Game(boardFromFen(fen), RepetitionRule.FULL);
const legal = game.validMoves(); // Uint8Array mask over action ids
```

## Running in a worker

Point your bundler's worker constructor at a small host file of your own
that imports the worker entry — a side-effect import, not a re-export:

```ts
// komugi.worker.ts
import '@komugi-ai/engine/worker';
```

```ts
// page.ts
const worker = new Worker(
  new URL('./komugi.worker.ts', import.meta.url),
  { type: 'module' },
);
```

The `@komugi-ai/engine/worker` entry installs itself as the sole side
effect of being imported — point your bundler's worker constructor at the
host file above and nothing else is required on the worker side. A host
composing its own worker module instead (for example, to add its own
message types alongside the engine's) can call `installWorkerHost(self,
options)` (from the core entry) directly; it has no side effects of its
own, so it can be combined with the host's own message types in the same
file. `options` lets a host inject its own model fetcher, its own
`onnxruntime-web` loader, or its own GPU adapter in place of the defaults
(used by this package's own tests to run without a real network or a real
browser).

Once the worker is running, initialize it with an `init` message carrying:

- `modelUrl` / `modelSha256` — the URL of the ONNX model to fetch, and the
  expected SHA-256 digest of its bytes. The worker fetches the model
  itself and verifies the digest before building a session; a mismatch
  never reaches inference.
- `ortAssets` — the `onnxruntime-web` build files the host serves:
  `{ wasm: { mjs, wasm }, webgpu?: { mjs, wasm } }`. The `webgpu` build is
  loaded lazily, only if a WebGPU backend is actually selected.

After a successful `init`, send search requests as ordinary worker
messages; replies come back on the same channel.

## Using KomugiClient

`KomugiClient` (exported from the core entry, and from `@komugi-ai/engine/client`)
is the main-thread class most hosts should use instead of writing their own
worker glue: it owns one dedicated worker at a time, speaks the init/search
protocol above on the host's behalf, and gives back a promise API whose every
outcome is one of a move, a typed failure, or `{ status: 'stopped' }` (from
`stop()`).

```ts
import { KomugiClient } from '@komugi-ai/engine';

const client = new KomugiClient({
  createWorker: () => new Worker(new URL('./komugi.worker.ts', import.meta.url), { type: 'module' }),
  model: { url: modelUrl, sha256: modelSha256 },
  ortAssets: { wasm: { mjs: wasmMjsUrl, wasm: wasmUrl } },
});

const result = await client.play({ fen, sims: 400 });
if (result.status === 'move') {
  // result.reply.san / result.reply.action
} else if (result.status === 'failed') {
  // show result.failure.message and a "Try again" action: client.retry() then client.play(...)
}
```

Just like the worker protocol itself, `KomugiClient` never substitutes
another engine's move for Komugi's own: there is no option, callback, or
return value — on the client or anywhere else — through which a host asks,
or the client lets a host ask, a different engine to move in Komugi's place.
When Komugi cannot produce a real search result, `play()` resolves with a
typed failure instead of a silent fallback, and the host's only recourse is
to show that failure's `message` and offer Try again (`retry()` then
`play()` again); `retry()` is always safe, and no failure code is treated as
permanent by the client.

To load a released model, resolve its card (and optionally `models.json`)
first; both functions are synchronous and fetch nothing — the host fetches
and parses the JSON and maps asset file names to URLs:

```ts
import { resolveModelById } from '@komugi-ai/engine';

const resolved = resolveModelById(id, modelsJson, cardJson, (asset) => new URL(asset.path, releaseBase).href);
if (resolved.status === 'ok') {
  const client = new KomugiClient({ createWorker, model: resolved.model, ortAssets });
}
```

The resolved `model` carries the card's `planeMapVersion`: every search is
sent with it, and a `play()` asking for a different one fails
`incompatible-model`. The card's `rulesProfile` is recorded only; `play()`
uses the request's `rules` (or the engine default) exactly as before.

Two behaviours worth knowing before wiring up a host UI:

- **Threads-timeout recovery.** If a worker's threaded wasm startup does not
  finish within `threadsTimeoutMs` (see "Hosting requirements" above), the
  client relays that reply's `probeRecord` to `onProbeRecord` exactly as it
  would on a normal init, then transparently starts one fresh worker capped
  at `maxThreads: 1` and retries the same `init` — a host sees this as a
  slightly slower `init()`/`play()`, not a failure, unless that retry itself
  fails.
- **Crash cap.** A search-stage timeout, crash, or malformed reply restarts
  the worker automatically and keeps the client usable for the next `play()`;
  the client only gives up after `maxConsecutiveCrashes` (default 3) such
  outcomes in a row, at which point the failure is latched into
  `client.state === 'failed'` / `client.lastFailure` so a deterministic
  crash cannot loop forever. Any successful move resets the count, and
  `retry()` clears the latch.

### Thinking: sims, time, or both

`play()`'s thinking budget is a `ThinkCap`: `{ sims }`, `{ sims, maxMs }`, or
`{ maxMs }` alone — whichever of `sims` and `maxMs` is hit first stops the
search. `{ fen }` with neither is a type error, and also a runtime rejection
(`play() needs sims, maxMs, or both`) if it somehow reaches `play()` unchecked.

A time-only request is resolved on the client to a node ceiling,
`TIME_ONLY_NODE_CEILING` (20,000): the real stop is still the `maxMs`
deadline, but a reply that the deadline cut off reports `stats.sims === 19999`
and `stats.deadlineHit === true`. The built-in `LEVELS` (see Strength levels,
below) always use both — `sims` for a strength that's consistent across
devices, `maxMs` as a time cap for slow ones.

## Analysis, stop, and progress

For an opted-in play (`stoppable: true`, a `topN`, or a `progressEvery*`
field) or an `analyze()` call, `KomugiClient` can ask the search for its
result early and can watch it while it runs.

- **`moveNow()`** asks the pending stoppable search for its best move so
  far instead of waiting out its full budget. It returns `false` (and does
  nothing) when no stoppable play or analysis is pending. The resolved
  reply carries `stats.stopReason === 'stop'`, unless the stop raced the
  last chunk, in which case it is a normal reply. Calling it before the
  search has reached the worker is fine — the stop is sent right after.
- **`analyze(request)`** runs a search with no `sims`/`maxMs`: it keeps
  going until `moveNow()`, its node ceiling, or `stop()`. `request` is an
  `AnalyzeRequest` — a `PlayRequest` without `sims`/`maxMs`/`stoppable`,
  plus `nodeCeiling`. `nodeCeiling` defaults to
  `ANALYSIS_DEFAULT_NODE_CEILING` (20,000 nodes) and cannot exceed
  `ANALYSIS_MAX_NODE_CEILING` (80,000); reaching it ends the search with
  `stats.stopReason === 'ceiling'`.
- **`stats.stopReason`** is `'stop'`, `'ceiling'`, or `'deadline'` — set
  only on an opted-in search, and only when it ended early or an analysis
  hit its ceiling, never on a search that ran to its full budget.
- **`onProgress`** (a `KomugiClient` constructor option) is called with
  each progress event while an opted-in search runs: `playouts`
  (cumulative, strictly increasing), `bestSan`/`bestAction` (the search's
  current pick), `rootWdl`, `pv`, and, when the request's `topN > 0`,
  `topMoves`.
- **`topN`** (on a play or an `analyze()` request) ranks the root's moves
  the way the engine itself chooses between them, without the post-search
  tactical override. The reply's (and any progress event's) `topMoves`
  holds up to `topN` entries, each with `action`, `san`, `descriptor`,
  `visits`, `wdlMean`, `prior`, `lcbEligible`, and `lcbScore`.
- **`stop()` is unchanged** — it still means *abandon*: it settles
  `{ status: 'stopped' }` immediately and terminates the worker, and never
  hands back a move. Use `moveNow()` when you want the search's current
  best move instead.

For time-only play (no `sims`), see "Thinking: sims, time, or both" above.

```ts
const client = new KomugiClient({
  createWorker,
  model,
  ortAssets,
  onProgress: (e) => console.log(e.playouts, e.bestSan),
});

const resultPromise = client.analyze({ fen, topN: 3, nodeCeiling: 5_000 });
// later, to take whatever move is best so far:
client.moveNow();
const result = await resultPromise;
if (result.status === 'move') {
  // result.reply.stats.stopReason, result.reply.topMoves
}
```

## Pondering

The engine ponders by default: while the host is idle between moves, the
worker keeps thinking on the opponent's time, using up to `k` predicted
replies. Turn it off (or back on), or change `k`, with a `ponder` option on
`KomugiClient` or the `setPonder()` method:

```ts
const client = new KomugiClient({
  createWorker,
  model,
  ortAssets,
  ponder: { enabled: false }, // off from the start; omit this to keep the default (on, k 1)
});

await client.setPonder({ enabled: true, k: 2 }); // turn it back on, with up to 2 predicted replies
```

`k` is the number of predicted replies pondered per move, from 1 to 3 — a
higher `k` covers more of the opponent's likely replies at the cost of
spreading the same thinking time across them. `setPonder()` resolves with
the worker's own `{ enabled, k, hidden, discharging }` once a worker is
ready to apply it, or `null` if none is (the setting is still stored, and
applied to the next worker the client starts).

Pondering also pauses itself while the page is hidden or the battery is
discharging; relay that from the page with `setPonder()`'s `hidden`/
`discharging` fields:

```ts
document.addEventListener('visibilitychange', () => client.setPonder({ hidden: document.hidden }));
```

## Loading a model

`loadModel(locator, options?)` fetches a released model once on the main
thread, verifies its sha256, and hands back a `KomugiModel` ready for
`KomugiClient`. It never rejects: it resolves `{ status: 'ok', model, card }`
or `{ status: 'failed', failure }`.

```ts
import { KomugiClient, loadModel } from '@komugi-ai/engine';

const result = await loadModel({ manifestUrl: 'models/komugi.card.json' }, {
  onProgress: (loaded, total) => showProgress(loaded, total), // total is null when unknown
});
if (result.status === 'ok') {
  const client = new KomugiClient({ createWorker, model: result.model, ortAssets });
  await client.init();
  // ... play as usual ...
  client.dispose(); // also releases the loaded model
} else {
  // show result.failure.message and offer Try again (a fresh loadModel call)
}
```

The `ModelLocator` takes one of three forms:

- `{ manifestUrl, resolveUrl? }` — fetch the model card JSON from
  `manifestUrl`, then the model. Asset paths resolve against the card's own
  URL unless `resolveUrl` is given. A relative `manifestUrl` resolves
  against `options.baseUrl`, which defaults to the page's `location.href`;
  with no base at all the load fails with a typed failure.
- `{ card, resolveUrl }` — a card object the host already has.
- `{ id, index, card, resolveUrl }` — a card checked against `models.json`
  by ID, as with `resolveModelById`.

`resolveUrl` maps a card asset to the URL to fetch it from. The card's
compatibility check runs before anything is fetched, so an incompatible
card fails `incompatible-model` without downloading the model.

**Ownership.** The returned `model.url` is a `blob:` URL holding the
verified bytes, so the worker reads the model from memory instead of
fetching it a second time (it still re-checks the sha256). The client
revokes that URL in `dispose()`; worker restarts after `stop()`, `retry()`
or a crash keep reusing it. One loaded model serves one client: to build
another client, call `loadModel` again. A host that never builds a client
from an `'ok'` result calls `result.model.release?.()` itself.

**Cache Storage.** Unless `cache: false` is passed, verified model bytes
are stored in Cache Storage keyed by their sha256, and only bytes that
match the card's `sha256` are ever written or served. A repeat load is a
cache hit with no network fetch; a corrupted entry is evicted and fetched
again. The cache keeps at most two models: the one just loaded and the most
recently used other one. If Cache Storage is unavailable or full, the load
still succeeds from the network.

**Secure context.** Verification needs `crypto.subtle`. In a non-secure
context `loadModel` fails `init-failed` and never loads unverified bytes.

**Content Security Policy.** The worker fetches the model from a `blob:`
URL, so the `connect-src` of the policy that governs the worker must allow
`blob:`. A worker loaded from a non-blob URL gets its policy from its own
script response, not from the page. This rule is not yet verified in a
browser; please confirm it in yours before relying on it.

## Strength levels

Instead of computing `sims`/`batch`/`maxMs` yourself, a fast-search request
can carry a named level:

```ts
{ id, fen, history, level: 'focused' }
```

`LEVEL_NAMES` lists the four levels, weakest to strongest: `'casual'`,
`'focused'`, `'sharp'`, `'awakened'`. `DEFAULT_LEVEL` (`'focused'`) is a
product-level default for a level picker's starting position, not an
engine-level fallback — a request must give exactly one of `sims` or
`level`; giving neither (or both) is a request error.

`LEVELS[name]` holds each level's fixed playout count and time cap
(`capMs`), plus the per-backend batch size the engine resolves against
(`webgpu`/`wasmMt`/`wasm1t`). The table is frozen, including each entry's
`batch` object. `resolveLevel(name, { backend, threads })` is the pure
function behind level resolution, turning a level name and the already
backend/thread-selected session into the exact `{ sims, batch, maxMs }` a
raw request would otherwise have to spell out; `backendCaseOf(backend,
threads)` is the three-way backend classification (`'webgpu' | 'wasm-mt' |
'wasm-1t'`) it resolves against.

A `level` field is only accepted by the fast-search path; sending it
alongside `sims`, `batch`, or `maxMs` is a request error, and an unknown
level name fails with a message matching `/unknown level/` rather than a
generic type error.

**What `capMs` bounds.** `capMs` is a soft cap on the playout loop only,
checked between batches of evaluator calls — not a bound on the full
round-trip time of a reply. A reply's wall time can exceed `capMs` by:

- *before the cap starts:* time spent queued behind other work (including
  an in-flight ponder search), replaying history, and — on a worker's first
  request — the one-time backend probe during `init`.
- *inside the capped window but never checked against the cap:* the root
  evaluation, and the batch of evaluator calls already in flight when the
  cap is reached (at most one evaluator call's worth of extra time).
- *after the last check:* the post-search refutation survey, then move
  notation and predicted-reply bookkeeping.

A host picking a client-side request timeout should budget `capMs` plus
those terms, not `capMs` alone.

## Failure codes

The worker never substitutes another engine's move for its own — if it
cannot produce a real search result, it replies with a typed error instead
of a silent fallback. A host should show the user the failure and offer a
retry, not paper over it with a different engine or a built-in heuristic.

- `not-initialized` — a search arrived before any `init` was sent.
- `model-fetch` — fetching `modelUrl` failed or returned a non-2xx status.
  `loadModel` emits it too, for a failed manifest or model fetch, a body
  shorter than the card's declared size, or a relative `manifestUrl` with
  no base to resolve against. A body *longer* than the declared size is
  instead cut at that size and then fails `sha-mismatch`, below.
- `sha-mismatch` — the fetched model's bytes did not match `modelSha256`.
  `loadModel` emits it too, when verified bytes — including a body cut at
  the card's declared size — do not match the card's `sha256`. A cache
  hit that fails this re-check is evicted and refetched by `loadModel`
  itself, so a cached entry never surfaces this failure.
- `init-failed` — `init` itself failed for any other reason. `loadModel`
  emits it too, for example in a non-secure context with no `crypto.subtle`.

A search sent after a failed `init` gets that `init`'s own code back
(`model-fetch`, `sha-mismatch`, or `init-failed`), not a fresh
`not-initialized`.

`KomugiClient` adds its own codes on top of the worker's (`KomugiFailure.code`):

- `timeout` — `init` or a search did not answer within its time budget.
- `worker-crash` — the worker errored, could not be created, or was stopped
  mid-`init`.
- `illegal-reply` — the worker's reply was malformed.
- `backend-unavailable` — a search failed without a typed worker code.
- `incompatible-model` — a model card does not fit this engine build (wrong
  `actionEncodingVersion`, a `planeMapVersion` other than 3 or 4, a malformed
  card, or a card that disagrees with `models.json`), or a `play()` request's
  `planeMapVersion` contradicts the one its model was resolved with.
- `model-not-found` — `resolveModelById` was given an ID that is not in
  `models.json`, or a `models.json` that is malformed.

A failure reply carries both a human-readable message and its code:

```ts
{ id, err: 'no init was ever sent', code: 'not-initialized' }
```

A request to the worker never throws and never crashes the worker — every
failure, typed or not, comes back as a reply on the same channel.

## Hosting requirements

- The host serves the ONNX model file and the `onnxruntime-web` `.wasm`/
  `.mjs` build files referenced by `ortAssets`; this package never bundles
  or fetches them from anywhere else.
- Model verification uses `crypto.subtle`, which is only available in a
  secure context (`https:`, or `localhost` during development).
- Multi-threaded wasm execution needs cross-origin isolation on the page
  that creates the worker:

  ```
  Cross-Origin-Opener-Policy: same-origin
  Cross-Origin-Embedder-Policy: require-corp
  ```

  Without these headers, `onnxruntime-web` still works, but falls back to
  a single-threaded backend.
- Multi-threaded wasm without the `onnxruntime-web-use-extern-wasm`
  condition below (see Bundler configuration) can stall instead of
  failing outright — see that section for why.
- If a page later turns on these headers, a returning visitor's HTTP
  cache may still hold worker/asset files fetched before isolation was
  enabled, which the now-isolated page will refuse to use. Serving those
  files under a new URL (for example, a cache-busting filename suffix)
  forces a one-time refetch.
- A threaded wasm start that does not finish within `threadsTimeoutMs`
  (default 10 s) fails `init` with the typed error code `threads-timeout`
  instead of hanging forever. The host should persist the reply's
  `probeRecord` and start a *new* worker with `maxThreads: 1` — the failed
  worker cannot usefully retry itself, since the wasm runtime only reads
  its thread count once per worker.

## Developing

- Requires Node >= 22.18 (for native TypeScript type stripping).
- `npm test` runs the suite.
- `npm run build` emits the `dist/` directory published to npm.
- Some suites are fixture-backed and need fixture files exported by the
  Komugi repository's own Python tooling; they are not required to build
  or use the published package.
- `npm run typecheck:strict` runs the stricter, consumer-parity typecheck.

### Tests

Most of the fixture-backed suites read a small golden sample committed
under `test/fixtures/`, so `npm test` runs to green on a clean checkout
with no external fixture generation step and no Python interpreter
required. The handful of suites backed by a real ONNX model skip under
that default, with the reason `requires KOM_FULL_REGRESSION=1`, since no
model is committed to this package.

- `npm test` — the default suite against the committed fixtures; the
  ONNX-backed suites skip.
- `npm run test:full` — the same suite with `KOM_FULL_REGRESSION=1` and
  every fixture environment variable pointed at a freshly generated,
  full-size fixture (produced by the Komugi repository's own Python
  tooling); nothing skips.
- `npm run test:pack` — the packaging-only suite under `test/pack/`,
  checking what actually ships in the published tarball.

## Bundler configuration

`onnxruntime-web`'s default browser build inlines the Emscripten
wasm-threads loader into the chunk that imports it. When that chunk is
the worker's own chunk, the inlined loader's pthreads re-run the
worker's own chunk on each thread instead of a dedicated loader file —
and because nothing throws, a threaded session can simply stall instead
of erroring. `onnxruntime-web`'s exports map offers a build that loads
that loader from a separate file behind the
`onnxruntime-web-use-extern-wasm` condition; a host enables it like this
(verified to move the threads loader out of the worker chunk in a
production build; not verified against a real threaded run):

```ts
// vite.config.ts (host-side, not this package)
import { defaultClientConditions, defineConfig } from 'vite';

export default defineConfig({
  resolve: {
    conditions: ['onnxruntime-web-use-extern-wasm', ...defaultClientConditions],
  },
  optimizeDeps: {
    exclude: ['onnxruntime-web', '@komugi-ai/engine'],
  },
  worker: {
    format: 'es',
  },
});
```

- `worker: { format: 'es' }` is required too: this package's worker entry
  lazily `import()`s `onnxruntime-web` per backend, and only a module
  (`'es'`) worker bundle can split that out — Vite's default IIFE worker
  format inlines it, and the production build fails.
- `resolve.conditions` is spread with `...defaultClientConditions` rather
  than set to a bare array, because a plain `resolve.conditions` *replaces*
  Vite's default client conditions instead of extending them — omitting
  the spread silently drops Vite's own conditions, not just the ORT one.
- `optimizeDeps.exclude` matters independently: without it, Vite's
  dependency pre-bundler can pull `onnxruntime-web` (and this package)
  through a path that never sees the `resolve.conditions` override,
  reintroducing the inlined-loader problem through the back door.

For webpack 5 hosts: a run-time, non-literal `import()` of an ORT asset
URL needs `/* webpackIgnore: true */` (webpack hard-fails without it;
Vite only warns). This package's own code has no such import — the
worker loads `onnxruntime-web` through a bare specifier, not a URL
`import()` — so this only matters for a host wiring its own model URL.

## Integration guide

Hosting a release — the HTTP headers to serve, the optional CSP and
COOP/COEP headers and what they trade off, the backend ladder, the
failure contract, bundle size, the bundler requirement, and a
step-by-step checklist — is covered end to end in the integration guide:
[`packages/engine/docs/integration.md`](https://github.com/chessai/komugi/blob/main/packages/engine/docs/integration.md).
Installing this package also adds a local `komugi-assets` command (for
example in an npm script) that runs the same CLI the guide's `npx` form does.

## License

MIT — see [LICENSE](./LICENSE). See [NOTICE](./NOTICE) for third-party
credit, including gungi.js.
