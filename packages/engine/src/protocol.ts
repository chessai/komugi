// Worker protocol (design doc section 2.7, section 6 U8): the JSON-safe
// postMessage request/reply shapes, and PRODUCTION_IDENTITY_ENV as the
// worker's own config defaults -- the production identity is the Python
// MLH search wrapper's tuple, so this port's defaults must match that, not
// the bare server's.

import type { BackendKind, BackendRecord } from './backend.ts';
import type { LevelName } from './levels.ts';
// N1 (design section 9): MoveHistoryEntry is defined in move_io.ts, not here
// -- root.ts needs it and sits in the rules-only closure import_graph.test.ts
// pins; protocol.ts's own heavy imports (backend.ts, rule_features.ts,
// search/search.ts) must never join that closure. Re-exported here for
// host-facing consumers.
import type { MoveDescriptor, MoveHistoryEntry } from './move_io.ts';
export type { MoveHistoryEntry } from './move_io.ts';
import type { RulesProfile } from './repetition.ts';
import type { RulesConfig } from './rules_config.ts';
import type { HistoryStamp } from './root.ts';
import type { PlaneMapVersion } from './rule_features.ts';
import type { MatchTactical, SearchPhases } from './search/search.ts';

export interface MlhRequestConfig {
  lam?: number;
  tau0?: number;
  tau1?: number;
  slope?: number;
}

export interface RequestConfig {
  draftmin?: number;
  discovery?: boolean;
  mlh?: MlhRequestConfig;
}

export interface WorkerRequest {
  id: string | number;
  fen: string;
  // Widened from string[] to MoveHistoryEntry[] (design section 5): an
  // entry may be SAN (unchanged) or a MoveDescriptor. Every existing caller
  // sending only SAN strings keeps compiling and behaving identically.
  history?: MoveHistoryEntry[] | null;
  // KOM-1104: the FEN `history` replays from (root.ts's `gameFromHistory`/
  // `rootGame`), instead of always the advanced start. Absent keeps every
  // existing caller byte-identical; only consulted when `history` is an
  // array -- a history-less request's FEN path never reads it. An invalid
  // `start` falls back exactly like a bad history (stamp `mismatch`).
  start?: string;
  // KOM-1105 (plumbing only, no rule change): the rules profile the search
  // is played under. Absent defaults to 'komugi' (repetition.ts's
  // DEFAULT_RULES_PROFILE), validated before the ponder cache lookup and
  // `buildRoot` (an invalid value is a request error), and threaded into
  // `Game` via `rootGame`/`gameFromHistory`.
  // KOM-1203 (design docs/design/kom1203-rules-config.md): widened to also
  // accept a host-supplied `RulesConfig` object (`toRulesConfig` resolves
  // either), never `WorkerInitRequest` -- this field, not a new one.
  rules?: RulesProfile | RulesConfig;
  // KOM-1162: the observation plane-map version every encoder call of this
  // search uses (rule_features.ts), passed from the model's sidecar. Absent
  // defaults to 3; validated like `rules` (parsePlaneMapVersion), before the
  // ponder cache lookup and `buildRoot` -- an invalid value is a request
  // error. Part of the ponder key; independent of `rules`.
  planeMapVersion?: PlaneMapVersion;
  sims: number;
  threads?: number;
  config?: RequestConfig;
  // KOM-1081: true -> the reply's stats carry `phases`. Off by default, so a
  // reply stays a pure function of the request apart from `ms`.
  timing?: boolean;
  // Fast-mode U6a (design section 8 U6): `handleFast`'s batch size. Absent
  // or 1 with no `maxMs` degenerates to `handle()`'s own search (design
  // 1.7's dispatch/degeneracy guarantees).
  batch?: number;
  // Fast-mode U6a (design 1.7 C3, 4.3): a "maximum think time" deadline in
  // milliseconds. When set, `handleFast` installs a `shouldStop` closure
  // that `searchFastAsync`/`searchBatchedGen` check once per completed
  // batch (one playout at a time when a deadline is present, design 1.7's
  // own rule) -- the deadline has exactly one implementation regardless of
  // `batch`. The reply's stats carry `deadlineHit`; `playouts` is already
  // the real (possibly partial) count.
  maxMs?: number;
  // KOM-1202 (design section 3, review V1): a raw request never carries a
  // level -- `WorkerLevelRequest` below is the level form, and this field
  // (together with `WorkerLevelRequest`'s own `sims?`/`batch?`/`maxMs?:
  // never`) discriminates the two on BOTH sides of the union, so a mixed
  // literal fails the excess-property check instead of structurally
  // matching whichever member declares each property.
  level?: never;
  // KOM-1195 design section 2: opt in to `stop` with no other new field
  // (`isOptedIn`, section 2). Un-opted requests keep today's exact path
  // (AC1).
  stoppable?: true;
  // KOM-1195 design section 4: progress cadence, positive integers.
  progressEveryPlayouts?: number;
  progressEveryMs?: number;
  // KOM-1195 design section 5: non-negative integer; 0 is treated as absent.
  topN?: number;
  // KOM-1195 design section 6: the analysis form is `WorkerAnalyzeRequest`
  // below, discriminated on both sides like `WorkerLevelRequest.level`.
  analyze?: never;
  nodeCeiling?: never;
}

// KOM-1195 design section 2 (review F4/F6): the analysis form,
// `handleFast`-only -- discriminated on BOTH sides exactly like
// `WorkerLevelRequest`, so only here may `sims` be omitted.
export type WorkerAnalyzeRequest =
  Omit<WorkerRequest, 'sims' | 'level' | 'analyze' | 'nodeCeiling'> & {
    analyze: true;
    // KOM-1195 design section 6: positive integer <= ANALYSIS_MAX_NODE_CEILING.
    nodeCeiling?: number;
    sims?: never;
    level?: never;
  };

// KOM-1202 (design section 3): the level form of a search request --
// `handleFast`-only (rule 5 below). `sims`/`batch`/`maxMs` are all resolved
// from `level` plus the engine's selected backend (levels.ts's
// `resolveLevel`), so this type forbids supplying them directly.
export type WorkerLevelRequest = Omit<WorkerRequest, 'sims' | 'batch' | 'maxMs' | 'level'> & {
  level: LevelName;
  sims?: never;
  batch?: never;
  maxMs?: never;
};

// KOM-1081: the search's own phases (search.ts's SearchPhases) plus the
// worker's request handling around it, all in performance.now()
// milliseconds: prepareMs (validation, root reconstruction, tree setup) +
// searchMs + finishMs (ceiling check, SAN) == totalMs.
export interface WorkerPhases extends SearchPhases {
  prepareMs: number;
  finishMs: number;
  totalMs: number;
}

// KOM-1195 design section 2/5: a single root move's read off the existing
// root (no second search). `lcbScore` is `null` exactly when
// `!lcbEligible` (never -Infinity: JSON-safe).
export interface RootMoveStat {
  action: number;
  san: string;
  descriptor: MoveDescriptor;
  visits: number;
  wdlMean: [number, number, number];
  prior: number;
  lcbEligible: boolean;
  lcbScore: number | null;
}

export interface WorkerStats {
  playouts: number;
  evaluations: number;
  nodes: number;
  rootWdl: number[];
  q: number;
  tactical: MatchTactical | null;
  ms: number;
  model_sha256: string | null;
  sims: number;
  draftmin: number;
  discovery: 'on' | 'off';
  mlh: string;
  // Present only when the request set `timing: true`.
  phases?: WorkerPhases;
  // Fast-mode U6a (design 1.7 C3): true only when a `maxMs` deadline fired
  // before `sims` playouts completed; always false through `handle()`/
  // `handleAsync()`, which never see one.
  deadlineHit: boolean;
  // Fast-mode U6b (design 3.3): the backend this reply's search actually
  // ran on, present once `init()` has chosen one (absent through
  // `handle()`/`handleAsync()`/`handleFast()` on an engine that never
  // called `init()`).
  backend?: BackendKind;
  // Fast-mode U6b (design 3.3): present only when this request hit a
  // mid-search backend failure and retried from scratch on the next
  // backend of the ladder -- "<failed backend>: <error message>".
  fallback?: string;
  // KOM-1101: present alongside `fallback` -- the record the page should
  // persist now that a mid-search failure has moved `backend` off the one
  // `init()` originally settled on.
  probeRecord?: BackendRecord;
  // Fast-mode U7 (design 5.1 N-6): present and true only on a ponder HIT --
  // this reply is a cached speculative search's result, not a fresh one.
  pondered?: boolean;
  // KOM-1105: the request's effective rules profile, echoed ONLY when it is
  // not the default ('komugi') -- a default-path reply stays byte-identical
  // to before this field existed. Not folded into `configIdentityString`.
  // KOM-1203 F2(b): a request that sent a `RulesConfig` object echoes the
  // matching preset's name (gungijs/ligungi) or the canonical `RulesConfig`
  // object itself (a genuinely custom config) -- see `engine.ts`'s
  // `echoRules`.
  rules?: RulesProfile | RulesConfig;
  // KOM-1195 design section 2 (review F12): which trigger ended an
  // opted-in search early. Absent by default -- never set on a request
  // that did not opt in (AC1), and never set when the search ran to its
  // playout cap other than an analysis reaching its ceiling.
  stopReason?: 'stop' | 'ceiling' | 'deadline';
  // KOM-1219 design section 2.2: present only when the root had exactly one
  // legal action and the reply never touched the evaluator -- absent (never
  // false) on every other reply.
  forced?: true;
}

// KOM-1102 item 1: the top predicted human reply to this engine move, read
// off the standing tree the same way pondering does (ponder.ts's
// `predictReplies`, worker.ts's own F9/S-5 call) -- zero evaluator calls,
// no change to this reply's own move/stats/bias table. `action` is the
// predicted child's action id; `san` its SAN under the position this
// reply's `action` reaches (the same board a pondered search for it would
// start from).
export interface PredictedReply {
  action: number;
  san: string;
  // KOM-1198 unit 4 (design section 5): describes `action` at the
  // post-chosen-move board, last key.
  descriptor: MoveDescriptor;
}

export interface WorkerReplyOk {
  id: string | number;
  san: string;
  action: number;
  stamp: HistoryStamp;
  stats: WorkerStats;
  // Fast-mode U7 follow-up (KOM-1102 item 1): present whenever
  // `predictReplies` finds a prediction under the chosen child -- regardless
  // of whether pondering is enabled or gated off -- absent when it does not
  // (a terminal `action`, or a chosen child with no visits yet).
  predictedReply?: PredictedReply;
  // KOM-1195 design section 2/5: present only when `request.topN > 0`;
  // assigned BEFORE `descriptor`, which stays the last key (KOM-1198).
  topMoves?: RootMoveStat[];
  // KOM-1198 unit 4 (design section 5): always present, always the LAST key
  // -- describes `action` at the root board this reply resolved.
  descriptor: MoveDescriptor;
}

export interface WorkerReplyErr {
  id: string | number;
  err: string;
  // KOM-1185 unit 4b (design section 4.2/8): set only by `worker_host.ts`'s
  // own typed-error replies (`WORKER_ERROR_CODES`); unset for every
  // `engine.ts`-internal error (a malformed request, a search-time
  // `BackendError`) -- a bare `string` so callers that don't recognise a
  // particular code still see the `err` message.
  code?: string;
}

// KOM-1185 unit 4b (design section 4.2/8): the codes `worker_host.ts`
// generates on a `WorkerReplyErr` -- the one place that knows which of
// these happened before the reply crosses the `postMessage` boundary as a
// bare string.
export const WORKER_ERROR_CODES = [
  'not-initialized',
  'model-fetch',
  'sha-mismatch',
  'init-failed',
  'threads-timeout',
] as const;
export type WorkerErrorCode = (typeof WORKER_ERROR_CODES)[number];

export type WorkerReply = WorkerReplyOk | WorkerReplyErr;

// KOM-1195 design section 2/3: `stop` is a new message shape, dispatched
// like `ponderSettings`/`ponderNow` in worker_host.ts -- it targets an
// in-flight search by id, never a `WorkerRequest` field.
export interface WorkerStopRequest {
  id: string | number;
  type: 'stop';
  target: string | number;
}
export type WorkerStopReply =
  | { id: string | number; type: 'stop'; acknowledged: true }
  | { id: string | number; type: 'stop'; acknowledged: false; reason: 'not-stoppable' | 'unknown-target' };

// KOM-1195 design section 2/4: progress is sent unprompted mid-search, not
// a reply to anything the host posted.
export interface WorkerProgressEvent {
  id: string | number;
  type: 'progress';
  playouts: number;
  bestSan: string;
  bestAction: number;
  rootWdl: [number, number, number];
  pv: string[];
  topMoves?: RootMoveStat[];
}

// KOM-1195 design section 2 (review F6): `WorkerReply` is left alone --
// widening it breaks engine.ts's existing `'stats' in`-style narrowing.
// This is the separate outgoing union for `postMessage`'s own signature
// (worker_host.ts, U5).
export type WorkerOutgoingMessage =
  | WorkerReply
  | WorkerInitReply
  | WorkerPonderSettingsReply
  | WorkerPonderNowReply
  | WorkerStopReply
  | WorkerProgressEvent;

// Fast-mode U6b (design section 3, section 8 U6; ticket KOM-1095): the
// worker's own model-load/backend-probe message, independent of a search
// request. JSON-safe like WorkerRequest -- a loaded ORT module or a probe
// session is never part of this message; whoever embeds this worker
// resolves `modelUrl` and supplies the loaded pieces out of band
// (ort_browser.ts's own loaders), `init()` itself only ever receives
// already-resolved objects, never a URL to fetch.
export interface WorkerInitRequest {
  id: string | number;
  // KOM-1185 unit 4b (design section 4.2): required, not optional -- a
  // plain search request has no `type` field at all and keeps not having
  // one, so an init message sent without `type` would otherwise dispatch
  // as a search instead of an init.
  type: 'init';
  modelUrl: string;
  modelSha256: string;
  // The user's thread-count override (design 3.2 step 1); backend.ts's own
  // probe clamps/ignores it exactly as a search request's absence of one
  // would (defaultThreads).
  threads?: number;
  // KOM-1214: a hard ceiling on `threads` above (and on its default when
  // absent) -- the host's way of forcing a fresh worker to 1 thread (or any
  // lower cap) after a prior worker's `threads-timeout`. backend.ts's probe
  // applies it after the default/override is computed.
  maxThreads?: number;
  // KOM-1214: how long a threads > 1 wasm session creation/warm-up may run
  // before `init()` fails with the typed `threads-timeout` error. Default
  // 10000 (backend.ts's own default); never applied at threads === 1.
  threadsTimeoutMs?: number;
  // The user's backend preference: 'wasm' skips WebGPU probing entirely
  // (never calls requestAdapter()); 'webgpu' changes nothing, since it is
  // the probe's own default order.
  backend?: BackendKind;
  // KOM-1185 unit 4b2 (design section 5.2 "One ORT delivery model"):
  // host-supplied URLs to the exact onnxruntime-web build files it serves --
  // replaces `ortUrl` (dropped, nothing implemented it). `worker_host.ts`'s
  // `loadOrt` loads the bare `onnxruntime-web/wasm`(`/webgpu`) specifier;
  // these URLs are wired onto the loaded module's `env.wasm.wasmPaths`
  // before the first `InferenceSession.create` (`ort_core.ts`'s
  // `makeOrtBackend`). `webgpu` absent means the worker never probes
  // WebGPU. `WorkerEngine.init()` never reads this field itself -- only
  // `worker_host.ts`/`makeOrtBackend` touch it.
  ortAssets?: { wasm: { mjs: string; wasm: string }; webgpu?: { mjs: string; wasm: string } };
  // KOM-1226: ORT's own `logSeverityLevel` (0 verbose .. 4 fatal) for every
  // `InferenceSession.create` this init builds. Absent defaults to 3
  // (errors only) -- unlike the default, warnings such as "Some nodes were
  // not assigned to the preferred execution providers" print. Validated as
  // an integer in [0, 4]; an out-of-range value is an 'init-failed' error.
  ortLogSeverityLevel?: number;
  // KOM-1101: a Web Worker has no localStorage -- the page relays the
  // BackendRecord it persisted from the previous `probeRecord` (below) back
  // here, taking the place of `WorkerInitEnv.storage`'s own read. `null`
  // means the page has no record to relay (still distinct from omitting the
  // field, which leaves the `WorkerInitEnv.storage` path -- if any --
  // untouched); passed straight through to `probeBackend`'s own
  // `cachedRecord` option (backend.ts, KOM-1098).
  cachedRecord?: BackendRecord | null;
}

export interface WorkerInitReplyOk {
  id: string | number;
  backend: BackendKind;
  threads: number;
  batch: number;
  // KOM-1096 item 1 (design 3.3): present whenever `init()` itself fell back
  // off a failed backend before settling -- either the probe's own choice
  // failing to build a real session, or (future rungs) a build failure on
  // an intermediate rung of the ladder. `from` is the backend that failed,
  // `reason` its error message; `backend`/`threads`/`batch` above are
  // already the settled fallback rung's. Mid-search fallbacks are instead
  // stamped on a search reply's WorkerStats.fallback (a plain string: those
  // belong to a search, not init, and already have a reply shape of their
  // own).
  fallback?: { from: BackendKind; reason: string };
  // KOM-1101: the record the page should persist (and relay back as
  // `cachedRecord` next start) -- `probeBackend`'s own `ProbeResult.record`
  // on the happy path, or the fallback ladder's own record (the backend this
  // init settled on after a session-build failure) when `fallback` above is
  // set.
  probeRecord: BackendRecord;
}

// KOM-1214: the init failure reply additionally carries the probe's own
// failure mark whenever one was built and persisted before the error (so
// far only a ThreadsTimeoutError's own `record`) -- least-intrusive add-on
// to the existing bare `WorkerReplyErr` shape, rather than a new field on it.
export type WorkerInitReplyErr = WorkerReplyErr & { probeRecord?: BackendRecord };

export type WorkerInitReply = WorkerInitReplyOk | WorkerInitReplyErr;

// Fast-mode U7 (design section 5, 5.4; section 8 U7 M-6/F7 fix): the
// pondering settings/gating message -- independent of a search or init
// request, JSON-safe like both. Every field is a partial update (absent ->
// unchanged); the page is the only source of `hidden`/`discharging` (the
// `visibilitychange`/`getBattery()` relay is page-side, design 6.3 M-6d/out
// of scope here) -- the worker only ever reads whatever it was last told.
export interface WorkerPonderSettingsRequest {
  id: string | number;
  type: 'ponderSettings';
  // Pondering is on by default (ticket KOM-1100 F7, overriding design 5.4's
  // "default off" text).
  enabled?: boolean;
  // Top-K predicted replies pondered per move, default 1, clamped to [1, 3]
  // (design 5.4).
  k?: number;
  // Page-provided gating flags (design 5.4): a hidden tab or a battery
  // reporting `charging === false`. Absent leaves the worker's last-known
  // value in place -- a missing battery API is "respected as set", never
  // inferred as discharging.
  hidden?: boolean;
  discharging?: boolean;
}

export interface WorkerPonderSettingsReplyOk {
  id: string | number;
  enabled: boolean;
  k: number;
  hidden: boolean;
  discharging: boolean;
}

export type WorkerPonderSettingsReply = WorkerPonderSettingsReplyOk | WorkerReplyErr;

// KOM-1102 item 2 (design 5.1's reply-triggered ponder start, extended): the
// page's own request to resume pondering from the standing tree its last
// reply left behind, instead of waiting for the next reply once a gating
// flag (tab visible again, battery charging) clears. `fen`/`history` mirror
// a search request's own identity fields -- the page's current view of the
// game, checked against what the engine's last reply actually produced; a
// mismatch is one of several reasons this can be a no-op (see
// `WorkerPonderNowReply.reason`).
export interface WorkerPonderNowRequest {
  id: string | number;
  type: 'ponderNow';
  fen: string;
  // Widened from string[] to MoveHistoryEntry[] (design section 5), same as
  // WorkerRequest.history above.
  history?: MoveHistoryEntry[] | null;
}

export interface WorkerPonderNowReply {
  id: string | number;
  started: boolean;
  // Present only when `started` is false.
  reason?: 'no_prior_reply' | 'history_changed' | 'real_search_since' | 'gated' | 'already_pondering' | 'no_prediction';
}

// PRODUCTION_IDENTITY: discovery on, DRAFTMIN 10, MLH lam 0.05 / tau0 0.8 /
// tau1 0.9 / slope 32. The bare server's own defaults (discovery off,
// draftmin 0, lam 0, tau0 0.20, tau1 0.40, slope 32 -- the Python match
// server's and MLH search wrapper's own defaults) stay reachable through a request's
// `config`, which overrides field by field.
export const PRODUCTION_IDENTITY: {
  discovery: boolean;
  draftmin: number;
  mlh: { lam: number; tau0: number; tau1: number; slope: number };
} = {
  discovery: true,
  draftmin: 10,
  mlh: { lam: 0.05, tau0: 0.8, tau1: 0.9, slope: 32 },
};
