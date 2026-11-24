// KOM-1185 unit 4b (design docs/design/ts-engine-library.md section 4.2
// "The split", section 7.1 AC4, section 8 failure contract): the real
// `postMessage` dispatch a worker entry needs -- `init`/`ponderSettings`/
// `ponderNow`/search routing, a FIFO so a search never races an in-flight
// `init`, and typed-error replies instead of the old eager
// `currentBatchEvaluator ?? defaultBatchEvaluator()` read that could hand a
// real search a hash-based fake move (review F2). Side-effect free on
// import -- only `installWorkerHost(scope, ...)` does anything.
//
// Unit 4b2 (section 5.2): the default `makeInitEnv` fetches `modelUrl`,
// sha256-verifies it against `modelSha256`, and wires `ort_core.ts`'s
// `makeOrtBackend` from the request's `ortAssets` -- no storage access in
// the worker at all (the page relays `cachedRecord`/`probeRecord` instead).

import type { BackendKind, GpuLike } from './backend.ts';
import { WorkerEngine } from './engine.ts';
import type { WorkerInitEnv } from './engine.ts';
import type { OrtModule } from './ort_core.ts';
import { makeOrtBackend } from './ort_core.ts';
import { observationSize } from './observation.ts';
import type { ScoreTerminalHook } from './rules_config.ts';
import type {
  WorkerAnalyzeRequest,
  WorkerInitRequest,
  WorkerLevelRequest,
  WorkerOutgoingMessage,
  WorkerPonderNowRequest,
  WorkerPonderSettingsRequest,
  WorkerReplyErr,
  WorkerRequest,
  WorkerStopRequest,
} from './protocol.ts';

// What a real Worker global (or a test's fake one) needs to offer --
// registering with `addEventListener`, never `onmessage =` (installing
// twice must compose, not clobber).
export interface WorkerScopeLike {
  addEventListener(type: 'message', listener: (event: { data: unknown }) => void): void;
  postMessage(reply: WorkerOutgoingMessage): void;
}

// What `installWorkerHost`'s default `makeInitEnv` needs to fetch the model
// -- narrower than the real `fetch`/`Response` DOM types (only the fields
// this module reads), so a test can inject one with no real network.
export interface FetchLike {
  (url: string): Promise<{ ok: boolean; status: number; arrayBuffer(): Promise<ArrayBuffer> }>;
}

// `options.makeInitEnv` overrides the default `makeInitEnv` wholesale (a
// test, or a host with its own asset-loading story); `loadOrt`/`fetch`/
// `gpu` are the injectables the DEFAULT below is actually built from --
// unused when `makeInitEnv` is supplied.
export interface WorkerHostOptions {
  makeInitEnv?: (request: WorkerInitRequest) => Promise<WorkerInitEnv>;
  loadOrt?: (kind: BackendKind) => Promise<OrtModule>;
  fetch?: FetchLike;
  gpu?: GpuLike;
  // KOM-1203 unit 6 (section 4): engine-scoped terminal-scoring hook for
  // behaviour no public RulesProfile covers -- never per-request (it cannot
  // cross postMessage), installed once on the WorkerEngine this host builds.
  scoreTerminal?: ScoreTerminalHook | undefined;
}

type IncomingMessage =
  | WorkerRequest
  | WorkerLevelRequest
  | WorkerAnalyzeRequest
  | WorkerInitRequest
  | WorkerPonderSettingsRequest
  | WorkerPonderNowRequest
  | WorkerStopRequest;

function isEngineMessage(data: unknown): data is IncomingMessage {
  if (typeof data !== 'object' || data === null) return false;
  const id = (data as Record<string, unknown>).id;
  return typeof id === 'string' || typeof id === 'number';
}

// A thrown `makeInitEnv` error may carry its own `code` (section 8's
// 'model-fetch'/'sha-mismatch', unit 4b2); anything else -- including the
// default's own rejection -- is 'init-failed'.
function initErrorCode(error: unknown): string {
  if (typeof error === 'object' && error !== null) {
    const code = (error as { code?: unknown }).code;
    if (typeof code === 'string') return code;
  }
  return 'init-failed';
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function typedError(code: string, message: string): Error & { code: string } {
  const error = new Error(message) as Error & { code: string };
  error.code = code;
  return error;
}

// `OrtModule`'s structural check (section 5.2's loadOrt contract): the
// three members every call site here actually touches. The real
// onnxruntime-web ESM build's namespace object carries these directly; a
// module whose namespace lacks them but carries a `default` (some bundler
// interop shapes) is checked there instead.
//
// KOM-1225: real onnxruntime-web (1.31.1, verified in-browser) exports
// `InferenceSession` as a CLASS -- `typeof InferenceSession === 'function'`,
// not `'object'`. Only the fake test modules were object-shaped, so this
// accepts either, as long as `create` is callable.
export function isOrtModule(candidate: unknown): candidate is OrtModule {
  if (typeof candidate !== 'object' || candidate === null) return false;
  const mod = candidate as Record<string, unknown>;
  const env = mod.env as { wasm?: unknown } | undefined;
  const inferenceSession = mod.InferenceSession as { create?: unknown } | undefined;
  const inferenceSessionShape = typeof inferenceSession === 'function' || (typeof inferenceSession === 'object' && inferenceSession !== null);
  return (
    typeof env === 'object' &&
    env !== null &&
    typeof env.wasm === 'object' &&
    env.wasm !== null &&
    inferenceSessionShape &&
    typeof inferenceSession?.create === 'function' &&
    typeof mod.Tensor === 'function'
  );
}

// The default `loadOrt` (section 5.2): a literal bare-specifier dynamic
// `import()` per kind -- WebGPU ships as a separate onnxruntime-web build,
// loaded only when this is actually called for 'webgpu'. `onnxruntime-web`
// is an optional peer dependency, not installed for this package's own
// typecheck/tests (`test/syntax.test.ts` loads every `src` module with no
// `node_modules` present), so each import() is lazy (inside this function,
// never at module top level) and `@ts-ignore`d, not `@ts-expect-error`d --
// the latter would start failing the moment the package IS installed.
async function defaultLoadOrt(kind: BackendKind): Promise<OrtModule> {
  let imported: unknown;
  if (kind === 'webgpu') {
    // @ts-ignore -- optional peer dependency, absent at typecheck; checked structurally below
    imported = await import('onnxruntime-web/webgpu');
  } else {
    // @ts-ignore -- optional peer dependency, absent at typecheck; checked structurally below
    imported = await import('onnxruntime-web/wasm');
  }
  const candidate = isOrtModule(imported) ? imported : (imported as { default?: unknown }).default;
  if (!isOrtModule(candidate)) {
    throw typedError('init-failed', `onnxruntime-web/${kind} did not export the expected ORT module shape`);
  }
  return candidate;
}

function hexDigest(bytes: ArrayBuffer): string {
  return Array.from(new Uint8Array(bytes))
    .map((byte) => byte.toString(16).padStart(2, '0'))
    .join('');
}

// Section 5.2's model-bytes pipeline: fetch `modelUrl` (never through
// `WorkerInitEnv.storage` -- there is no storage access in the worker at
// all, section 4.2/8), sha256 the bytes, compare against `modelSha256`
// BEFORE anything is loaded or a session created, then hand the verified
// bytes to `makeOrtBackend`. `options.gpu`/navigator's own `gpu`/
// `hardwareConcurrency` and `crossOriginIsolated` are read here, once, as
// plain values -- never a storage read.
function buildDefaultMakeInitEnv(options: WorkerHostOptions): (request: WorkerInitRequest) => Promise<WorkerInitEnv> {
  const loadOrt = options.loadOrt ?? defaultLoadOrt;
  const doFetch = options.fetch ?? (globalThis.fetch as unknown as FetchLike);

  return async (request: WorkerInitRequest): Promise<WorkerInitEnv> => {
    if (!request.ortAssets) {
      throw typedError('init-failed', 'init request is missing ortAssets');
    }
    const ortAssets = request.ortAssets;

    // KOM-1226: validated before anything is fetched/created -- an
    // out-of-range level is a request error, never silently clamped.
    const ortLogSeverityLevel = request.ortLogSeverityLevel;
    if (
      ortLogSeverityLevel !== undefined &&
      (!Number.isInteger(ortLogSeverityLevel) || ortLogSeverityLevel < 0 || ortLogSeverityLevel > 4)
    ) {
      throw typedError('init-failed', `ortLogSeverityLevel must be an integer between 0 and 4, got ${ortLogSeverityLevel}`);
    }

    let response: { ok: boolean; status: number; arrayBuffer(): Promise<ArrayBuffer> };
    try {
      response = await doFetch(request.modelUrl);
    } catch (error) {
      throw typedError('model-fetch', `model fetch failed for ${request.modelUrl}: ${errorMessage(error)}`);
    }
    if (!response.ok) {
      throw typedError('model-fetch', `model fetch for ${request.modelUrl} returned status ${response.status}`);
    }

    const modelBytes = new Uint8Array(await response.arrayBuffer());
    const digest = hexDigest(await globalThis.crypto.subtle.digest('SHA-256', modelBytes));
    const expected = request.modelSha256.toLowerCase();
    if (digest !== expected) {
      throw typedError('sha-mismatch', `model sha256 mismatch: expected ${expected}, got ${digest}`);
    }

    const { sessionFactory, createBatchBackend } = makeOrtBackend({
      loadOrt,
      ortAssets,
      modelBytes,
      obsShape: observationSize(),
      ...(ortLogSeverityLevel !== undefined ? { ortLogSeverityLevel } : {}),
    });

    const nav = (globalThis as { navigator?: { gpu?: GpuLike; hardwareConcurrency?: number } }).navigator;
    const isolated = (globalThis as { crossOriginIsolated?: boolean }).crossOriginIsolated === true;

    return {
      sessionFactory,
      createBatchBackend,
      gpu: options.gpu ?? nav?.gpu,
      isolated,
      hardwareConcurrency: nav?.hardwareConcurrency ?? 1,
    };
  };
}

// What every search consults instead of racing `init`: `ok: true` once an
// `init` has actually installed a real evaluator, `ok: false` with the
// code/message a following search should relay otherwise.
type InitOutcome = { ok: true } | { ok: false; err: string; code: string };

const NOT_INITIALIZED = { ok: false as const, err: 'no init was ever sent', code: 'not-initialized' };

// Registers a single `message` listener and owns the FIFO promise chain
// (design 4.2's F2 fix) that makes "every search waits for the latest init
// to settle" true regardless of arrival order. No top-level side effects --
// nothing here runs before this function is actually called.
export function installWorkerHost(scope: WorkerScopeLike, options: WorkerHostOptions = {}): void {
  // KOM-1195 design section 4 (U4b/U5): `engine.ts` has no `self` reference,
  // so it calls this option instead of posting directly.
  const engine = new WorkerEngine({
    scoreTerminal: options.scoreTerminal,
    onProgress: (e) => scope.postMessage(e),
  });
  const makeInitEnv = options.makeInitEnv ?? buildDefaultMakeInitEnv(options);

  // Reassigned synchronously by `handleInit` before any `await`, so a
  // search dispatched right after an `init` (both handled synchronously,
  // one `message` event at a time) always captures the NEW chain, never
  // the stale one.
  let initChain: Promise<InitOutcome> = Promise.resolve(NOT_INITIALIZED);

  async function handleInit(request: WorkerInitRequest): Promise<void> {
    const previous = initChain;
    initChain = previous.then(async (): Promise<InitOutcome> => {
      let env: WorkerInitEnv;
      try {
        env = await makeInitEnv(request);
      } catch (error) {
        const code = initErrorCode(error);
        const message = errorMessage(error);
        const reply: WorkerReplyErr = { id: request.id, err: message, code };
        scope.postMessage(reply);
        return { ok: false, err: message, code };
      }
      const reply = await engine.queueInit(request, env);
      scope.postMessage(reply);
      if ('err' in reply) return { ok: false, err: reply.err, code: reply.code ?? 'init-failed' };
      // KOM-1226: every later reply's stats.model_sha256 must carry the
      // init-verified sha, not the null this host's engine started with.
      engine.setVerifiedModelSha256(request.modelSha256);
      return { ok: true };
    });
    await initChain;
  }

  // KOM-1195 design section 3 (U5): `target` was already opened
  // SYNCHRONOUSLY by the listener below, before this async function's first
  // await -- a `stop` arriving while the request is still queued/
  // initialising still finds its target.
  async function handleSearch(
    request: WorkerRequest | WorkerLevelRequest | WorkerAnalyzeRequest,
    target: Exclude<ReturnType<typeof engine.openStopTarget>, WorkerReplyErr>,
  ): Promise<void> {
    const outcome = await initChain;
    if (!outcome.ok) {
      const reply: WorkerReplyErr = { id: request.id, err: outcome.err, code: outcome.code };
      scope.postMessage(reply);
      return;
    }
    // `init` settled ok -- `currentBatchEvaluator` is the real evaluator it
    // installed. Never `engine.defaultBatchEvaluator()`: that is the
    // hash-based fake move the product rule (section 8) bans here.
    const evaluator = engine.currentBatchEvaluator;
    if (evaluator === null) {
      const reply: WorkerReplyErr = { id: request.id, err: NOT_INITIALIZED.err, code: NOT_INITIALIZED.code };
      scope.postMessage(reply);
      return;
    }
    scope.postMessage(await engine.handleFast(request, evaluator, target));
  }

  scope.addEventListener('message', (event) => {
    const data = event.data;
    if (!isEngineMessage(data)) return;
    if ('type' in data) {
      if (data.type === 'init') {
        void handleInit(data);
        return;
      }
      if (data.type === 'ponderSettings') {
        scope.postMessage(engine.handlePonderSettings(data));
        return;
      }
      if (data.type === 'ponderNow') {
        scope.postMessage(engine.handlePonderNow(data));
        return;
      }
      // KOM-1195 design section 3 (U5): `handleStop` never posts the target
      // search's reply, only one `WorkerStopReply` for THIS message, right
      // away -- the search's own reply (if any) follows later, from
      // `handleSearch` below.
      if (data.type === 'stop') {
        scope.postMessage(engine.handleStop(data));
        return;
      }
      // review NIT (i): explicit dispatch, not a ponderNow fallthrough -- an
      // unrecognised `type` (isEngineMessage only checks `id`) gets its own
      // reply instead of being misrouted as a ponder-now request.
      const unrecognised = data as { id: string | number; type: unknown };
      scope.postMessage({ id: unrecognised.id, err: `unknown message type: ${String(unrecognised.type)}` });
      return;
    }
    // KOM-1195 design section 3 (U5): opened SYNCHRONOUSLY, before
    // `void handleSearch(...)` and so before `initChain`/`this.queue`/
    // `this._ponderTask` -- a `stop` arriving in any of those windows finds
    // its target and is honoured at the first chunk boundary. The listener's
    // own `.finally` releases it on paths that never reach `handleFast`
    // (an `initChain` rejection, or no evaluator); `handleFast`'s own
    // `finally` releases it on every path that does.
    const opened = engine.openStopTarget(data);
    if ('err' in opened) {
      scope.postMessage(opened);
      return;
    }
    void handleSearch(data, opened).finally(() => engine.closeStopTarget(opened));
  });
}
