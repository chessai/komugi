// @komugi-ai/engine -- core public entry (design doc section 3.1).
// Re-exports only, no logic of its own. AC6 (test/import_graph.test.ts)
// asserts this file's import graph has no node builtins, no non-literal
// dynamic import(), no absolute specifiers, and no disallowed text.

export * from './rules.ts';

export type {
  WorkerRequest,
  WorkerReply,
  WorkerReplyOk,
  WorkerReplyErr,
  WorkerStats,
  WorkerPhases,
  RequestConfig,
  MlhRequestConfig,
  WorkerInitRequest,
  WorkerInitReply,
  WorkerInitReplyOk,
  WorkerPonderSettingsRequest,
  WorkerPonderSettingsReply,
  WorkerPonderNowRequest,
  WorkerPonderNowReply,
  PredictedReply,
  WorkerLevelRequest,
  WorkerStopRequest,
  WorkerStopReply,
  WorkerProgressEvent,
  RootMoveStat,
  WorkerAnalyzeRequest,
  WorkerOutgoingMessage,
} from './protocol.ts';
export { PRODUCTION_IDENTITY } from './protocol.ts';

export { WorkerEngine, snapshotBias, restoreBias } from './engine.ts';
export type { WorkerInitEnv } from './engine.ts';

export { installWorkerHost } from './worker_host.ts';
export type { WorkerHostOptions, WorkerScopeLike } from './worker_host.ts';

// KOM-1203 unit 6 (section 4, N5): the engine-scoped scoreTerminal hook's
// public types -- not re-exported from rules.ts (that closure stays
// JSON-only-RulesConfig), so the barrel exports them directly.
export type { AbsoluteWdl, ScoreTerminalHook, TerminalReason, TerminalScoreContext } from './rules_config.ts';

export type {
  BackendKind,
  BackendRecord,
  BackendTiming,
  ProbeResult,
  ProbeOptions,
  SessionFactory,
  ProbeSession,
  GpuLike,
  GpuAdapterLike,
  StorageLike,
} from './backend.ts';
export {
  probeBackend,
  backendCacheKey,
  WEBGPU_BATCH,
  WASM_MT_BATCH,
  defaultThreads,
  makeOrtSessionFactory,
} from './backend.ts';

export type { MlhBatchEvaluator } from './batch_evaluator.ts';
export { BackendError, wrapMlhBatch, runOrtBatch } from './batch_evaluator.ts';

export { makeFakeMlhEvaluator, makeFakeMlhEvaluatorAsync } from './fake_evaluator.ts';

export {
  runOrt,
  softmax3f32,
  makeOrtEvaluator,
  makeOrtMlhLeafEvaluator,
  makeOrtBackend,
} from './ort_core.ts';
export type { OrtModule, OrtSession, OrtTensorLike, OrtRawEval, Evaluation } from './ort_core.ts';

export { encodeStatesBatch } from './host_encode.ts';

export type { PlaneMapVersion } from './rule_features.ts';
export { parsePlaneMapVersion } from './rule_features.ts';

export { LEVEL_NAMES, DEFAULT_LEVEL, LEVELS, parseLevel, backendCaseOf, resolveLevel } from './levels.ts';
export type { LevelName, LevelBatch, LevelDef, BackendCase, ResolvedLevel } from './levels.ts';

export { KomugiClient, TIME_ONLY_NODE_CEILING } from './client.ts';
export type {
  KomugiWorkerLike,
  KomugiWorkerFactory,
  KomugiModel,
  KomugiOrtAssets,
  KomugiClientOptions,
  KomugiPonderOptions,
  KomugiPonderState,
  PlayRequest,
  ThinkCap,
  KomugiReady,
  InitResult,
  PlayResult,
  KomugiClientState,
} from './client.ts';
export type { ClientErrorCode, KomugiFailureCode, KomugiFailure } from './client_failures.ts';

export * from './model_registry.ts';
export * from './model_loader.ts';
