// The browser-safe ORT-web leaf evaluator seam (design doc section 2.7,
// section 6 U8; split out per docs/design/ts-engine-library.md section
// 4.1a, review F1/F4). Pure ORT structural types and math every entry
// needs -- no Node reference, no vendor filesystem path.
//
// KOM-1078 resolves what U8's evaluator.ts could not: search/tree.ts's
// MatchTree now takes a MaybeAsyncWdlEvaluator (tree.ts's own seam) and
// suspends with a JS generator exactly at the one evaluator call per
// expansion, so a genuinely async per-leaf evaluator -- onnxruntime-web's
// `InferenceSession.run`, which has no synchronous API -- can be driven
// straight through the search via `searchAsync`/`playoutAsync` (no
// synchronous bridge, no replay). `makeOrtMlhLeafEvaluator` below is that
// evaluator, in the search/mlh.ts `MlhLeafEvaluatorAsync` shape worker.ts's
// async path wraps with the MLH seam, same as the sync fake evaluator does
// for the sync path.

import type { BackendKind, ProbeSession, SessionFactory } from './backend.ts';
import { BackendError, runOrtBatch } from './batch_evaluator.ts';
import type { MlhBatchEvaluator } from './batch_evaluator.ts';
import { encodeStatesBatch } from './host_encode.ts';
import { observation } from './observation.ts';
import type { MlhLeafEval, MlhLeafEvaluatorAsync } from './search/mlh.ts';
import type { MatchState } from './search/state.ts';

// Structural types for the pieces of onnxruntime-web's API this module
// touches; the library ships no .ts declarations of its own and vendoring
// one is out of scope (design section 1 Q1).
export interface OrtTensorLike {
  data: Float32Array | ArrayLike<number>;
}
export interface OrtSession {
  run(feeds: Record<string, unknown>): Promise<Record<string, OrtTensorLike>>;
}
export interface OrtModule {
  env: {
    wasm: {
      numThreads: number;
      // design 5.2: the host-supplied exact build-file URLs (`ortAssets`),
      // wired by `makeOrtBackend` before the first `InferenceSession.create`.
      wasmPaths?: string | { mjs?: string; wasm?: string };
      proxy?: boolean;
    };
  };
  InferenceSession: {
    create(pathOrBytes: string | Uint8Array, opts: { executionProviders: string[]; logSeverityLevel?: number }): Promise<OrtSession>;
  };
  Tensor: new (kind: string, data: Float32Array, dims: readonly number[]) => unknown;
}

export interface OrtRawEval {
  policy: Float32Array;
  wdl: Float32Array;
  movesLeft: number;
}

// One forward pass, RAW ORT outputs -- pre-softmax wdl, exactly as the ONNX
// model and the Python ONNX exporter's fixture both carry it (it encodes
// `out.wdl_logits[0].numpy()`, not a softmaxed array). What the ORT test
// suite compares against the fixture, and what the ORT benchmark script
// already compares the same way.
export async function runOrt(
  session: OrtSession,
  ort: OrtModule,
  obs: Float32Array,
  obsShape: readonly number[],
): Promise<OrtRawEval> {
  const tensor = new ort.Tensor('float32', obs, [1, ...obsShape]);
  const result = await session.run({ obs: tensor });
  return {
    policy: Float32Array.from(result.policy!.data),
    wdl: Float32Array.from(result.wdl!.data),
    movesLeft: Float32Array.from(result.moves_left!.data)[0]!,
  };
}

// softmax3f32 (design 2.7, N4): the serving seam's torch float32 3-way
// softmax over wdl logits (src/komugi/selfplay/inference.py:1782,
// `torch.softmax(out.wdl_logits.float(), dim=-1)`), ported with
// Math.fround after every operation and widened to float64 only at the
// end -- state.ts's WdlEval.wdl is float64. Any residual ulp difference
// from torch's own float32 kernel is net noise (design 2.7), measured by
// U9, not bit-matched here.
export function softmax3f32(logits: ArrayLike<number>): Float64Array {
  let max = Math.fround(logits[0]!);
  for (let i = 1; i < 3; i++) max = Math.fround(Math.max(max, Math.fround(logits[i]!)));
  const exps = new Float32Array(3);
  for (let i = 0; i < 3; i++) exps[i] = Math.fround(Math.exp(Math.fround(Math.fround(logits[i]!) - max)));
  let sum = Math.fround(0);
  for (let i = 0; i < 3; i++) sum = Math.fround(sum + exps[i]!);
  const out = new Float64Array(3);
  for (let i = 0; i < 3; i++) out[i] = Math.fround(exps[i]! / sum);
  return out;
}

export interface Evaluation {
  logits: Float32Array;
  wdl: Float64Array;
  movesLeft: number;
}

// The design section 2.7 `Evaluator` seam: one position per run, wdl turned
// into a probability triple. Async, per the reasoning above -- used
// standalone (ort.test.ts), not threaded through MatchTree.
export function makeOrtEvaluator(
  session: OrtSession,
  ort: OrtModule,
  obsShape: readonly number[],
): (obs: Float32Array) => Promise<Evaluation> {
  return async (obs: Float32Array): Promise<Evaluation> => {
    const raw = await runOrt(session, ort, obs, obsShape);
    return { logits: raw.policy, wdl: softmax3f32(raw.wdl), movesLeft: raw.movesLeft };
  };
}

// KOM-1078: the real ORT evaluator in the search/mlh.ts `MlhLeafEvaluatorAsync`
// shape -- the one worker.ts's async path installs. One obs per leaf, built
// the same way the sync fake evaluator and e2e_driver.mjs's own leaf
// evaluator already do (`observation(state.game)`); `movesLeft` passes
// straight through raw, exactly as the fake evaluator's `movesLeft` head
// does (search/mlh.ts never reads anything else out of it before `applyMlh`
// gates on `|v|`).
export function makeOrtMlhLeafEvaluator(
  session: OrtSession,
  ort: OrtModule,
  obsShape: readonly number[],
): MlhLeafEvaluatorAsync {
  return async (state: MatchState): Promise<MlhLeafEval> => {
    const obs = observation(state.game, { planeMapVersion: state.planeMapVersion });
    const raw = await runOrt(session, ort, obs, obsShape);
    return { policyLogits: raw.policy, wdl: softmax3f32(raw.wdl), movesLeft: raw.movesLeft };
  };
}

// Browser delivery of the same seam (design doc section 1 "Browser delivery
// without a bundler", section 6 U8/KOM-1078) -- same runOrt/softmax3f32
// math above, for a session loaded from a caller-supplied browser ESM
// build of onnxruntime-web rather than the Node vendor loader.
export function makeOrtEvaluatorBrowser(
  session: OrtSession,
  ort: OrtModule,
  obsShape: readonly number[],
): (obs: Float32Array) => Promise<Evaluation> {
  return async (obs: Float32Array): Promise<Evaluation> => {
    const raw = await runOrt(session, ort, obs, obsShape);
    return { logits: raw.policy, wdl: softmax3f32(raw.wdl), movesLeft: raw.movesLeft };
  };
}

// The real ORT evaluator in worker.ts's `MlhLeafEvaluatorAsync` shape, for
// a session loaded through the browser path instead of a vendor filesystem
// path.
export function makeOrtMlhLeafEvaluatorBrowser(
  session: OrtSession,
  ort: OrtModule,
  obsShape: readonly number[],
): MlhLeafEvaluatorAsync {
  return async (state: MatchState): Promise<MlhLeafEval> => {
    const obs = observation(state.game, { planeMapVersion: state.planeMapVersion });
    const raw = await runOrt(session, ort, obs, obsShape);
    return { policyLogits: raw.policy, wdl: softmax3f32(raw.wdl), movesLeft: raw.movesLeft };
  };
}

function rowSizeOf(obsShape: readonly number[]): number {
  return obsShape.reduce((a, b) => a * b, 1);
}

// Wraps a raw session so every `run` failure -- the probe's warm-up/timing
// calls and the batch evaluator's search-time calls alike -- surfaces as a
// `BackendError` (design 5.2/section 8): the one typed exception
// `engine.ts`'s fallback ladder catches to try the next backend.
function wrapSession(raw: OrtSession): OrtSession {
  return {
    async run(feeds: Record<string, unknown>): Promise<Record<string, OrtTensorLike>> {
      try {
        return await raw.run(feeds);
      } catch (error) {
        if (error instanceof BackendError) throw error;
        throw new BackendError(error instanceof Error ? error.message : String(error));
      }
    },
  };
}

function probeSessionOf(session: OrtSession, ort: OrtModule, obsShape: readonly number[]): ProbeSession {
  const rowSize = rowSizeOf(obsShape);
  async function runAt(rows: number): Promise<void> {
    const data = new Float32Array(rowSize * rows);
    const tensor = new ort.Tensor('float32', data, [rows, ...obsShape]);
    await session.run({ obs: tensor });
  }
  return { warmup: runAt, run: runAt };
}

// design doc section 5.2 "One ORT delivery model": the one builder a host or
// `worker_host.ts` calls -- it owns the injectable `loadOrt` loader and the
// host-supplied `ortAssets` URLs, loading each kind's module (and building
// its one InferenceSession, from `modelBytes`) at most once, lazily, only
// when that kind is actually requested. WebGPU is never loaded unless
// `createWebgpu`/`createBatchBackend('webgpu')` is called, and asking for it
// without `ortAssets.webgpu` rejects with a `BackendError` instead of
// calling `loadOrt` at all -- the engine's fallback ladder (`engine.ts`)
// walks to wasm on exactly that error.
export function makeOrtBackend(opts: {
  loadOrt: (kind: BackendKind) => Promise<OrtModule>;
  ortAssets: { wasm: { mjs: string; wasm: string }; webgpu?: { mjs: string; wasm: string } };
  modelBytes: Uint8Array;
  obsShape: readonly number[];
  // KOM-1226: forwarded to every InferenceSession.create below; absent
  // defaults to 3 (errors only), same as before this field existed.
  ortLogSeverityLevel?: number;
}): { sessionFactory: SessionFactory; createBatchBackend: (backend: BackendKind, threads: number) => Promise<MlhBatchEvaluator> } {
  let currentThreads = 1;
  const built = new Map<BackendKind, Promise<{ mod: OrtModule; session: OrtSession }>>();

  function getBuilt(kind: BackendKind): Promise<{ mod: OrtModule; session: OrtSession }> {
    let promise = built.get(kind);
    if (promise === undefined) {
      promise = (async (): Promise<{ mod: OrtModule; session: OrtSession }> => {
        const assets = kind === 'webgpu' ? opts.ortAssets.webgpu : opts.ortAssets.wasm;
        if (assets === undefined) {
          throw new BackendError('makeOrtBackend: webgpu backend requested but ortAssets.webgpu was not supplied');
        }
        const mod = await opts.loadOrt(kind);
        // S-2's ordering (preserved from backend.ts's own `makeOrtSessionFactory`):
        // wasmPaths/proxy/numThreads set before this kind's first `create`.
        mod.env.wasm.wasmPaths = { mjs: assets.mjs, wasm: assets.wasm };
        mod.env.wasm.proxy = false;
        mod.env.wasm.numThreads = currentThreads;
        const rawSession = await mod.InferenceSession.create(opts.modelBytes, {
          executionProviders: [kind],
          logSeverityLevel: opts.ortLogSeverityLevel ?? 3,
        });
        return { mod, session: wrapSession(rawSession) };
      })();
      built.set(kind, promise);
    }
    return promise;
  }

  const sessionFactory: SessionFactory = {
    setNumThreads(threads: number): void {
      currentThreads = threads;
      for (const promise of built.values()) {
        // A kind whose `getBuilt` rejected (e.g. webgpu with no
        // `ortAssets.webgpu`) stays in `built` as a memoized rejection;
        // `.catch` here only silences THIS derived promise -- the original
        // await site (`createWebgpu`/`createBatchBackend`) still sees and
        // surfaces the rejection itself.
        void promise.then(({ mod }) => {
          mod.env.wasm.numThreads = threads;
        }).catch(() => undefined);
      }
    },
    async createWebgpu(): Promise<ProbeSession> {
      const { mod, session } = await getBuilt('webgpu');
      return probeSessionOf(session, mod, opts.obsShape);
    },
    async createWasm(): Promise<ProbeSession> {
      const { mod, session } = await getBuilt('wasm');
      return probeSessionOf(session, mod, opts.obsShape);
    },
  };

  async function createBatchBackend(backend: BackendKind, threads: number): Promise<MlhBatchEvaluator> {
    sessionFactory.setNumThreads(threads);
    const { mod, session } = await getBuilt(backend);
    const rowSize = rowSizeOf(opts.obsShape);
    return (states): Promise<MlhLeafEval[]> =>
      runOrtBatch(session, mod, encodeStatesBatch(states, rowSize), states.length, opts.obsShape);
  }

  return { sessionFactory, createBatchBackend };
}
