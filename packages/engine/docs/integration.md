# Integration guide

This page covers hosting a release of `@komugi-ai/engine`: the HTTP headers
to serve, the optional cross-origin-isolation headers and what they trade
off, the backend ladder, the failure contract, bundle size, the bundler
requirement, how to wire the asset paths the CLI produces, and a checklist.

One command copies everything this guide describes into a host's public
directory:

```
npx @komugi-ai/engine assets <public-dir> --release-dir <dir> [--model <id>] [--webgpu]
```

It writes the `onnxruntime-web` build files, one released model (a card,
its `.onnx`, and `models.json`), and an `ort-assets.json` inventory of the
`onnxruntime-web` files with their sha256s and sizes.

## 1. Headers

| Extension | Content-Type | Cache-Control |
| --- | --- | --- |
| `.mjs` | `text/javascript` | `public, max-age=31536000, immutable` |
| `.wasm` | `application/wasm` | `public, max-age=31536000, immutable` |
| `.onnx` | `application/octet-stream` | `no-cache` |
| `.json` | `application/json` | `no-cache` |

- The `.mjs`/`.wasm` rows are long-lived and `immutable` only because the
  CLI copies them into a **versioned** directory, `ort-<version>/`. The
  glue basenames themselves are **not** content-addressed — they are
  identical across `onnxruntime-web` versions — so serving them from an
  unversioned path with `immutable` caching is unsafe.
- There is no registered MIME type for `.onnx`; verify your server's
  default before relying on `application/octet-stream`.
- Serve `<id>.card.json`, `models.json` and `ort-assets.json` with
  `no-cache`. The `.onnx` keeps its fixed release basename, so it also
  gets `no-cache`, unless your host versions its own path for it.

| Header | Value |
| --- | --- |
| `Cross-Origin-Opener-Policy` | `same-origin` |
| `Cross-Origin-Embedder-Policy` | `require-corp` |
| `Content-Security-Policy` | `script-src 'self' 'wasm-unsafe-eval'; connect-src 'self' blob:` |

**If you have a CSP**, add `'wasm-unsafe-eval'` to `script-src` and
`blob:` to `connect-src`, as in the example row above — the worker loads a
verified model through a `blob:` URL. CSP is a **recommended example, not
a required header**: a host with no CSP needs nothing here, and pasting
the whole example policy over an existing one can break the host's own
inline or third-party scripts.

## 2. Optional COOP/COEP

`Cross-Origin-Opener-Policy: same-origin` and
`Cross-Origin-Embedder-Policy: require-corp` (the same two lines as the
engine README's
[Hosting requirements](https://github.com/chessai/komugi/blob/main/packages/engine/README.md)
section; see the table in Headers, above, for the exact values).

**What it buys.** Multi-threaded wasm. Without these headers,
`onnxruntime-web` still works, falling back to a single-threaded backend.

**What it breaks**, named explicitly:

- **Cross-origin subresources without CORP or CORS.** Plain cross-origin
  images, fonts, scripts and iframes stop loading under `require-corp`
  unless they send `Cross-Origin-Resource-Policy` or are fetched with CORS
  (`crossorigin`).
- **Popups.** Under `same-origin` COOP, a `window.open()` to a
  non-isolated page loses `window.opener`.

## 3. The backend ladder

`backend.ts` probes backends in a fixed order at `init`: WebGPU first,
then multi-threaded wasm, then single-threaded wasm, falling back down the
ladder on each step's failure. The wasm pair
(`ort-wasm-simd-threaded.{mjs,wasm}`) feeds every wasm step; the
`--webgpu` pair (`ort-wasm-simd-threaded.asyncify.{mjs,wasm}`) feeds the
WebGPU step only, and is only copied when the CLI is run with `--webgpu`.
Multi-threaded wasm needs both the page that creates the worker to be
cross-origin isolated (Optional COOP/COEP, above) and the bundler's
extern-wasm condition (Bundler configuration, below): without cross-origin
isolation it falls back to single-threaded wasm, and without the
extern-wasm condition the inlined threads loader can stall a threaded
session instead of failing outright.

## 4. The failure contract

The engine never substitutes another engine's move for its own: on a
search or load failure it replies with a typed failure code, and the
host's only recourse is to show the message and offer a retry — never a
silent fallback to a different engine or a built-in heuristic. The full
list of codes, and what each one means, lives in one place, the engine
README's
[Failure codes](https://github.com/chessai/komugi/blob/main/packages/engine/README.md#failure-codes)
section; this guide does not copy that list, so the two cannot drift.

## 5. Bundle size

The engine's own JavaScript is small; the `onnxruntime-web` files it needs
are not. For `onnxruntime-web` 1.31.1, `ort-wasm-simd-threaded.wasm` is
about 14.8 MB (14,848,852 bytes), and, with `--webgpu`,
`ort-wasm-simd-threaded.asyncify.wasm` is about 28.7 MB (28,747,738
bytes). Those sizes are for 1.31.1; `ort-assets.json`'s `bytes` field on
each entry is the exact size for whatever version the CLI actually
copied. The `.onnx` model's size is the card's `assets.onnx.bytes` when
the card carries it, otherwise the size of the file the CLI copied.

**Extern-wasm caveat.** None of the files above lands in your own
JavaScript bundle — *only* when your bundler resolves the
`onnxruntime-web-use-extern-wasm` condition described in Bundler
configuration, below. Without it, the default `ort.wasm.bundle.min.mjs`
module contains a `new URL("ort-wasm-simd-threaded.wasm",
import.meta.url)`, which a bundler such as Vite treats as a bundled
asset.

## 6. Bundler configuration

Both target sites use a bundler, and a bundler is required: a host
without one cannot resolve `onnxruntime-web`'s bare import from inside the
engine's worker, because import maps do not apply inside module workers.

Set up the `onnxruntime-web-use-extern-wasm` resolve condition exactly as
shown in the engine README's
[Bundler configuration](https://github.com/chessai/komugi/blob/main/packages/engine/README.md)
section — this guide does not duplicate that snippet. The same snippet
also sets `worker: { format: 'es' }`, which a Vite production build needs
because this package's worker entry lazily imports `onnxruntime-web` per
backend, and only a module worker bundle can be split that way.
`packages/engine/examples/vite` is a minimal, type-checked source example
of a host wired up this way.

## 7. Wiring the paths

`ort-assets.json`'s `path` values are relative to `ort-assets.json`
**itself**, and must never be passed straight into `ortAssets` or into
`loadModel({ manifestUrl })`. Fetch `ort-assets.json` from an absolute
URL, then resolve every `path` it names — and the model card's URL —
against that same absolute URL, never against a bare specifier or a
`./`-relative string:

```ts
const ortAssetsUrl = new URL('/ort-assets.json', location.href).href;
const ort = await (await fetch(ortAssetsUrl)).json();
const abs = (p: string) => new URL(p, ortAssetsUrl).href;
const ortAssets = {
  wasm: { mjs: abs(ort.wasm.mjs.path), wasm: abs(ort.wasm.wasm.path) },
  ...(ort.webgpu ? { webgpu: { mjs: abs(ort.webgpu.mjs.path), wasm: abs(ort.webgpu.wasm.path) } } : {}),
};
const cardUrl = abs('<id>.card.json');
```

`cardUrl` is what you pass as `loadModel({ manifestUrl: cardUrl })`.
Skipping this step and passing a path such as
`ort-1.31.1/ort-wasm-simd-threaded.mjs` straight into `ortAssets` throws
inside the worker (it is a bare specifier, not a URL); a `./`-relative
path instead resolves against the bundled ORT chunk or the worker's own
URL, not your public root.

## 8. Thinking budget (ThinkCap)

`KomugiClient.play()` takes `sims` (a playout count), `maxMs` (a time cap),
or both — whichever stops the search first. A time-only request (`{ fen,
maxMs }`) is capped client-side at `TIME_ONLY_NODE_CEILING` (20,000 nodes),
so a reply the deadline cut off reports `stats.sims === 19999` and
`stats.deadlineHit === true`. The built-in `LEVELS` always use both: `sims`
for a consistent strength, `maxMs` as a time cap for slow devices.

## 9. Pondering

The engine ponders by default (on, `k: 1`): the worker keeps thinking on
the opponent's time between moves. Pass `ponder: { enabled: false }` to
`KomugiClient` to turn it off from the start, or call
`client.setPonder({ enabled, k })` later — `k` (1-3) is how many predicted
replies it ponders. Relay tab visibility and battery state so pondering
pauses itself when neither is in the host's favor:

```ts
document.addEventListener('visibilitychange', () => client.setPonder({ hidden: document.hidden }));
```

## 10. Checklist

1. Run `npx @komugi-ai/engine assets <public-dir> --release-dir <dir> [--model <id>] [--webgpu]`.
2. Serve `<public-dir>` with the headers in Headers, above.
3. If you have a CSP, extend it as shown in Headers, above.
4. Decide whether you want COOP/COEP (Optional COOP/COEP, above).
5. Fetch `ort-assets.json` from an absolute URL and resolve every `path`
   and the card URL against it, as in Wiring the paths, above, then pass
   the resolved `ortAssets` and `loadModel({ manifestUrl: cardUrl })`.
6. Before shipping, check that every file under `<public-dir>` returns
   `200` with the `Content-Type` documented above.
