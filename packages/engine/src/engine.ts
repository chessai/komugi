// The Web Worker entry point (design doc section 2.7, section 6 U8): wires
// root reconstruction (root.ts), the match search (search/*.ts) and the
// worker protocol (protocol.ts) into one request-in/reply-out function,
// loads the model once and answers one search at a time. Like EngineServer,
// the NODES are dropped before every request (the Python match server's own
// request handler -> player.py:239-244 -> search.py:548-550, which only clears the tree) but
// the ONE MatchTree and its bias table live as long as the engine:
// tree.py:253-261 keeps the table across clear() on purpose, so a request's
// answer depends on the requests before it, exactly as the Python server's
// does.
//
// KOM-1078: `prepare()`/`finish()` hold every statement `handle()`'s request
// processing used to run; `handleAsync()` is the resumable twin that awaits
// the search (searchAsync) instead of calling it, for a real
// per-leaf-async evaluator (ORT-web) via `wrapMlhEvaluatorAsync`.
// `wrapMlhEvaluatorAsync` accepts a synchronous leaf evaluator too (the
// default fake one included), so `handleAsync()` works out of the box with
// no evaluator installed.

import { decodeAction } from './actions.ts';
import { BackendError, wrapMlhBatch } from './batch_evaluator.ts';
import type { MlhBatchEvaluator } from './batch_evaluator.ts';
import { ThreadsTimeoutError, backendCacheKey, probeBackend } from './backend.ts';
import type { BackendKind, BackendRecord, GpuAdapterLike, GpuLike, SessionFactory, StorageLike } from './backend.ts';
import { ACTION_SIZE, ADVANCED_POSITION } from './constants.ts';
import { boardToFen } from './fen.ts';
import { makeFakeMlhEvaluator } from './fake_evaluator.ts';
import type { Game } from './game.ts';
import { parseLevel, resolveLevel } from './levels.ts';
import { describeRecord } from './move_io.ts';
import {
  DEFAULT_PONDER_GATING,
  DEFAULT_PONDER_SETTINGS,
  PonderCache,
  clampPonderK,
  ponderKey,
  predictReplies,
  runPonderSearch,
  shouldPonder,
} from './ponder.ts';
import type { PonderGating, PonderSettings, PonderSignal } from './ponder.ts';
import { DEFAULT_RULES_PROFILE, RULES_PRESETS, RULES_PROFILES, rulesConfigKey, toRulesConfig } from './rules_config.ts';
import type { RulesConfig, RulesProfile, ScoreTerminalHook } from './rules_config.ts';
import { DEFAULT_PLANE_MAP_VERSION, parsePlaneMapVersion } from './rule_features.ts';
import type { PlaneMapVersion } from './rule_features.ts';
import { HISTORY_PRESENT, historyKey, rootGame } from './root.ts';
import { buildSan } from './san.ts';
import type { BiasEntry } from './search/bias.ts';
import { MlhContext, wrapMlhEvaluator, wrapMlhEvaluatorAsync } from './search/mlh.ts';
import type { MlhConfig, MlhLeafEvaluator, MlhLeafEvaluatorAsync } from './search/mlh.ts';
import type { MatchNode } from './search/node.ts';
import { childMeanWdl, lcbEligible, lcbScores, rankRootMoves, selectAction } from './search/select.ts';
import type { BiasReader } from './search/select.ts';
import { DEFAULT_TRANSITION_CAP, search, searchAsync, searchFastAsync, soleLegalAction } from './search/search.ts';
import type { MatchResult, SearchConfig, SearchOptions, SearchPhases } from './search/search.ts';
import { MatchGameState } from './search/state.ts';
import type { MatchState, WdlEval } from './search/state.ts';
import { MatchTree } from './search/tree.ts';
import type { MaybeAsyncWdlEvaluator, MatchConfig as TreeConfig } from './search/tree.ts';
import { PRODUCTION_IDENTITY } from './protocol.ts';
import type { HistoryStamp } from './root.ts';
import type {
  PredictedReply,
  RootMoveStat,
  WorkerAnalyzeRequest,
  WorkerInitReply,
  WorkerInitRequest,
  WorkerPonderNowReply,
  WorkerPonderNowRequest,
  WorkerPonderSettingsReply,
  WorkerPonderSettingsRequest,
  WorkerProgressEvent,
  WorkerReply,
  WorkerLevelRequest,
  WorkerReplyErr,
  WorkerReplyOk,
  WorkerRequest,
  WorkerStats,
  WorkerStopRequest,
  WorkerStopReply,
} from './protocol.ts';

// Every MatchConfig field the match package reads, at its config.py default
// (src/komugi/search/match/config.py:120-149) -- the match defaults, which
// are a property of the search itself and distinct from the product
// identity (discovery/draftmin/mlh) layered on top per request.
const BASE_MATCH_CONFIG: Omit<TreeConfig, 'playouts'> = {
  num_actions: ACTION_SIZE,
  contempt: 0.0,
  dynamic_cpuct: true,
  c_puct: 1.0,
  c_explore: 1.0,
  c_log: 0.45,
  c_base: 500.0,
  sigma_prior: 0.4,
  sigma_prior_weight: 2.0,
  sigma_scale: 0.85,
  sigma_floor: 0.02,
  fpu_reduction: 0.2,
  fpu_reduction_root: 0.1,
  subtree_bias_factor: 0.45,
  bias_weight_exponent: 0.85,
  bias_free_prop: 0.8,
  bias_weight_floor: 1e-6,
  k_lcb: 5.0,
  min_visit_prop_for_lcb: 0.15,
  var_floor: 0.05,
  variance_mode: 'moment',
};

// CEILING_SLACK (the Python match server's own constant): the per-move evaluation ceiling is
// sized from the request's own sims rather than left at the match stack's
// generic default (design section 2.6's "Evaluation ceiling" bullet).
const CEILING_SLACK = 2;

// KOM-1219 design section 2.2: every `MatchResult.phases` key at 0 -- a
// forced reply's `timing: true` phases, since no playout/select/survey/
// evaluator work ever ran.
const ZERO_PHASES: SearchPhases = {
  setupMs: 0,
  rootEvalMs: 0,
  rootWinMs: 0,
  playoutsMs: 0,
  selectMs: 0,
  surveyMs: 0,
  resultMs: 0,
  searchMs: 0,
  evaluateMs: 0,
  evaluatorCalls: 0,
};

// KOM-1195 design section 6 (review F5, re-review N7): an analysis's node
// ceiling IS its playout cap (`sims = nodeCeiling - 1`; each playout adds at
// most one node to a tree that starts from the root alone). Derived from a
// stated memory budget of 64 MiB default / 256 MiB hard maximum at
// ANALYSIS_NODE_BYTES_ESTIMATE bytes per node (the measured ~2.5 KB/node,
// bias entries included, plus 20% headroom -- an upper bound).
export const ANALYSIS_NODE_BYTES_ESTIMATE = 3000;
export const ANALYSIS_DEFAULT_NODE_CEILING = 20_000;
export const ANALYSIS_MAX_NODE_CEILING = 80_000;

function formatMlhStamp(config: MlhConfig): string {
  return `${config.lam}/${config.tau0}/${config.tau1}/${config.slope}`;
}

function discoveryLabel(enabled: boolean): 'on' | 'off' {
  return enabled ? 'on' : 'off';
}

// KOM-1195 design section 2 (review F1): true iff `request` opts into the
// new fast-mode behaviour -- the analysis form, or any of `stoppable`/a
// `progressEvery*` field/a positive `topN`. `maxMs`/`level` alone do NOT opt
// in (section 1's byte-identical path). `topN: 0` is treated as absent.
type OptInRequest = WorkerRequest | WorkerLevelRequest | WorkerAnalyzeRequest;
function isOptedIn(request: OptInRequest): boolean {
  if (request.analyze === true) return true;
  if (request.stoppable === true) return true;
  if (request.progressEveryPlayouts !== undefined) return true;
  if (request.progressEveryMs !== undefined) return true;
  if (request.topN !== undefined && request.topN > 0) return true;
  return false;
}

// KOM-1195 design section 6: the analysis form is `handleFast`-only --
// exact mode (`handle()`/`handleAsync()`) has no chunked loop, no stop and no
// ceiling, so it rejects `analyze` explicitly. Read untyped: `WorkerRequest`
// forbids `analyze`, but an untyped host can still send it.
const ANALYZE_EXACT_MODE_ERR = 'analyze is supported only in fast mode (handleFast)';
function isAnalyzeRequest(request: object): boolean {
  return (request as { analyze?: unknown }).analyze === true;
}

// KOM-1195 design section 2 (review F12; retry hardening). `isOptedIn`
// above treats `progressEveryPlayouts: 0`/`topN: 1.5` as opted in (it only
// checks definedness/positivity for the chunked-path decision, never
// validity), and the ponder key folds in none of these three fields -- so an
// invalid value placed only behind `prepareFast`'s own checks could reach
// `lookupPonder` and come back a truncated success instead of an error. This
// is the ONE shared check, called at the top of every entry point that can
// reach `finish()`: `handle()`, `handleAsync()`, and `handleFast`'s queued
// callback (before `lookupPonder`, right next to `resolveRequest`). Returns
// the error message, or `null` when every present field is valid.
function validateProgressOptions(request: {
  topN?: number;
  progressEveryPlayouts?: number;
  progressEveryMs?: number;
}): string | null {
  if (request.topN !== undefined && (!Number.isInteger(request.topN) || request.topN < 0)) {
    return `topN must be a non-negative integer, got ${request.topN}`;
  }
  if (
    request.progressEveryPlayouts !== undefined &&
    (!Number.isInteger(request.progressEveryPlayouts) || request.progressEveryPlayouts <= 0)
  ) {
    return `progressEveryPlayouts must be a positive integer, got ${request.progressEveryPlayouts}`;
  }
  if (
    request.progressEveryMs !== undefined &&
    (!Number.isInteger(request.progressEveryMs) || request.progressEveryMs <= 0)
  ) {
    return `progressEveryMs must be a positive integer, got ${request.progressEveryMs}`;
  }
  return null;
}

// KOM-1195 design section 3 (U4a; re-review N3/N8). One `StopTarget` per
// fast-mode search for its whole life, registered before any await so a
// `stop` arriving during init/queueing is never lost (`handleFast` opens it
// synchronously at entry). `analysis` is read once, at open time, since a
// resolved `WorkerRequest` (what `prepareFast` sees) forbids `analyze`.
export type StopReason = 'stop' | 'ceiling' | 'deadline';
interface StopTarget {
  readonly id: string | number;
  readonly stoppable: boolean;
  readonly analysis: boolean;
  stopped: boolean;
}

// KOM-1195 design section 3 (review F2, re-review N1): one macrotask per
// yield, so a `stop` (itself a worker message, i.e. a macrotask) dispatched
// during the yield is seen by the very next `shouldStop` poll. `setImmediate`
// where it exists (Node); otherwise a FRESH `MessageChannel` per call -- never
// one shared for the whole search, since Node drains up to 1000 messages per
// port per event-loop turn and a shared channel would starve a late `stop`.
function nextMacrotask(): Promise<WdlEval[]> {
  return new Promise((resolve) => {
    const immediate = (globalThis as { setImmediate?: (cb: () => void) => void }).setImmediate;
    if (immediate !== undefined) {
      immediate(() => resolve([]));
      return;
    }
    const channel = new MessageChannel();
    channel.port1.onmessage = () => {
      channel.port1.close();
      channel.port2.close();
      resolve([]);
    };
    channel.port2.postMessage(null);
  });
}

// Fast-mode U7: named once, since pondering (below) needs the same board
// type to clone the game forward and decode a predicted reply's SAN.
type RootBoard = ReturnType<typeof rootGame>['game']['board'];

// What `prepare()` hands back to build the reply once a result is in hand --
// every statement `answer()` used to run between validating the request and
// calling `search()`, unchanged (KOM-1078: shared by `answer`/`answerAsync`
// so the two differ only in how the result is obtained).
interface PreparedSearch {
  tree: MatchTree;
  state: MatchGameState;
  searchConfig: SearchConfig;
  searchOpts: {
    discovery: boolean;
    ply: number;
    tacticalTransitionCap: number;
    clock?: () => number;
    // Fast-mode U6a (design 1.7 C3): present only when the request set
    // `maxMs`, checked once per completed batch by searchFastAsync.
    shouldStop?: () => boolean;
    // KOM-1195 design section 3 (U4a): present only for an opted-in fast
    // request's async search -- see `prepareFast`.
    yieldToHost?: () => Promise<WdlEval[]>;
    // KOM-1195 design section 4 (U4b): set only by `attemptFast`'s cadence
    // wrapper, and only when this engine has a constructor `onProgress`
    // callback and the request carries a `progressEvery*` field.
    onProgress?: (playoutsUsed: number) => void;
  };
  rootBoard: RootBoard;
  stamp: HistoryStamp;
  sims: number;
  draftmin: number;
  discovery: boolean;
  mlhConfig: MlhConfig;
  started: number;
  // KOM-1081: the phase clock and the prepare-phase reading, null unless the
  // request asked for `timing`.
  timing: { clock: () => number; t0: number; prepareMs: number } | null;
  // KOM-1105/KOM-1203: the effective, resolved `RulesConfig` this search ran
  // under (`toRulesConfig(request.rules)`), already validated by `buildRoot`.
  config: RulesConfig;
}

// Fast-mode U6a: everything `prepare()` hands back, plus the batch size
// `handleFast` passes through to `searchFastAsync` (design section 8 U6).
interface PreparedFastSearch extends PreparedSearch {
  batch: number;
  // KOM-1195 design section 3 (U4a): `target.stoppable` -- whether this
  // request opted in, read by `finish()` to decide whether `stats.stopReason`
  // may be set at all (never, for an un-opted request).
  opted: boolean;
  // KOM-1195 design section 3: a box, not a plain value, because the
  // composed `shouldStop` closure (built here, polled later by the search
  // itself) writes into it; `finish()` reads `.reason` only after the search
  // has returned.
  stopState: { reason: StopReason | null };
}

// Fast-mode U6b (design section 3, section 8 U6; ticket KOM-1095): builds
// (or rebuilds, on fallback) the real batch leaf evaluator for one backend.
// Owned by whoever calls `init()` -- real browser wiring loads an ORT
// module through ort_browser.ts's own loaders and wraps it with
// batch_evaluator.ts's `runOrtBatch`; tests hand this a fake directly, the
// same way `env.sessionFactory` below is a fake `SessionFactory`.
type BatchBackendBuilder = (backend: BackendKind, threads: number) => MlhBatchEvaluator | Promise<MlhBatchEvaluator>;

// `init()`'s dependencies (design 3.2's probe inputs, plus the real-session
// builder design 3.1/2.2 calls "the batched session"). Everything here is
// an already-resolved object or function -- `init()` never dynamic-imports
// a URL itself (that loading is ort_browser.ts's job, design section 6
// M-6c/section 3.1).
export interface WorkerInitEnv {
  sessionFactory: SessionFactory;
  createBatchBackend: BatchBackendBuilder;
  gpu?: GpuLike | undefined;
  isolated?: boolean;
  hardwareConcurrency?: number;
  storage?: StorageLike;
  now?: () => number;
}

// KOM-1096 item 2 (design 3.3): the fallback ladder itself, independent of
// WHERE a backend failed (init-time session build, or a mid-search
// `BackendError`) -- webgpu drops straight to wasm at the SAME thread
// count; wasm only drops to 1 thread (exact mode, design 3.1 item 3) when it
// was running at more than one. `null` means the ladder is exhausted: wasm
// at 1 thread never fails over to anything, since it has no capability
// requirement left to lose.
function nextLadderRung(backend: BackendKind, threads: number): { backend: BackendKind; threads: number } | null {
  if (backend === 'webgpu') return { backend: 'wasm', threads };
  if (backend === 'wasm' && threads > 1) return { backend: 'wasm', threads: 1 };
  return null;
}

// Mirrors backend.ts's own (unexported) `requestRealAdapter` closely enough
// to recompute the SAME `backendCacheKey` a later fallback write needs --
// two attempts, a fallback/software adapter rejected outright (design 3.2
// step 2). Only ever called off the ladder's failure path, never on the
// happy path, so it costs nothing when every backend keeps working.
async function resolveAdapterForCache(gpu: GpuLike | undefined): Promise<GpuAdapterLike | null> {
  if (!gpu) return null;
  for (let attempt = 0; attempt < 2; attempt++) {
    const adapter = await gpu.requestAdapter();
    if (adapter) return adapter.isFallbackAdapter ? null : adapter;
  }
  return null;
}

// What a successful `fallbackLadder` call hands back: the new backend/
// threads and the evaluator already built for them, plus enough of the
// ORIGINAL failure to format either a search reply's `fallback: "<from>:
// <message>"` string stamp or an init reply's `fallback: { from, reason }`
// object.
interface FallbackAdvance {
  backend: BackendKind;
  threads: number;
  evaluator: MlhBatchEvaluator;
  failedFrom: BackendKind;
  message: string;
  // KOM-1101: the record `writeFallbackCache` built for this rung -- `null`
  // only when `init()` was never called (`_cacheCtx` unset), which cannot
  // happen on either real call site (both run after `init()` has set it).
  record: BackendRecord | null;
}

// Everything `prepare()`/`prepareFast()` compute before installing an
// evaluator -- the request validation and root/tree-config build neither
// seam (single-leaf vs batch) changes. Shared so the two diverge only in how
// they wrap `leafEvaluator`/`batchEvaluator` and in what they install on
// `this._tree`.
interface RootBuild {
  started: number;
  clock: (() => number) | null;
  t0: number;
  sims: number;
  discovery: boolean;
  draftmin: number;
  mlhConfig: MlhConfig;
  rootBoard: RootBoard;
  stamp: HistoryStamp;
  state: MatchGameState;
  treeConfig: TreeConfig;
  turns: number;
  config: RulesConfig;
}

// The stats clock (design section 1: performance.now() only in benches and
// the worker's reported stats). Wrapped, not passed bare: a browser's
// performance.now throws when called detached from `performance`.
const WALL_CLOCK = (): number => performance.now();

// Fast-mode U7 (design 5.1): the request-level identity (everything C2's key
// calls "config identity") a cached ponder reply must also match -- the
// product-identity fields a request's own `config` can override, not the
// engine-wide MatchConfig (BASE_MATCH_CONFIG), which never varies per
// request.
function configIdentityString(discovery: boolean, draftmin: number, mlhConfig: MlhConfig): string {
  return JSON.stringify({ discovery, draftmin, mlh: mlhConfig });
}

// KOM-1203 unit 6 (advisor correction on section 5's F2(c)/section 4's "no
// bias/ponder key component needed" claim): two WorkerEngines differing only
// in their installed scoreTerminal hook must never share bias-table or
// ponder-cache entries, since the hook can change winState's terminal score
// for the exact same RulesConfig. Each distinct hook function gets a stable
// per-process id (assigned on first use); `undefined` (no hook installed)
// maps to the empty string, so every existing key is byte-identical to today
// whenever no hook is installed (AC1).
let nextHookId = 1;
const HOOK_IDS = new WeakMap<ScoreTerminalHook, number>();
function hookKeyComponent(hook: ScoreTerminalHook | undefined): string {
  if (hook === undefined) return '';
  let id = HOOK_IDS.get(hook);
  if (id === undefined) {
    id = nextHookId++;
    HOOK_IDS.set(hook, id);
  }
  return `|hook:${id}`;
}

// KOM-1203 F2(b): the echo rule for `stats.rules`. `requestRules` is the
// ORIGINAL request value (string, RulesConfig object, or undefined);
// `config` is its resolution (`toRulesConfig(requestRules)`). A string
// request value echoes back unchanged whenever it isn't the default --
// byte-identical to the pre-RulesConfig behaviour (`rules_profile.test.ts`'s
// goldens). An object/absent request value echoes nothing when `config`
// resolved onto the komugi preset (default-path byte-identical), the
// matching preset's own name when it resolved onto gungijs/ligungi (AC2's
// "indistinguishable from the preset" claim extended to the echo), or the
// canonical `RulesConfig` object itself for a config that matched no preset
// (a genuinely custom config; JSON-serialisable, no `hooks` field to strip).
function echoRules(
  requestRules: RulesProfile | RulesConfig | undefined,
  config: RulesConfig,
): RulesProfile | RulesConfig | undefined {
  if (typeof requestRules === 'string') {
    return requestRules !== DEFAULT_RULES_PROFILE ? requestRules : undefined;
  }
  // KOM-1203 unit 5b fix: compare by VALUE (`rulesConfigKey`), never object
  // identity -- a RulesConfig that reaches here via `postMessage` (every
  // worker deployment) is a structured clone, a different object with the
  // same fields, and `toRulesConfig`'s WeakSet fast path (`rules_config.ts`)
  // never fires for a clone, so `config === RULES_PRESETS[profile]` is false
  // for the normal case even though the two are value-equal.
  const configKey = rulesConfigKey(config);
  for (const profile of RULES_PROFILES) {
    if (configKey === rulesConfigKey(RULES_PRESETS[profile])) {
      return profile === DEFAULT_RULES_PROFILE ? undefined : profile;
    }
  }
  return config;
}

// What a ponder cache hit needs to rebuild a reply (via `finish()`) and to
// start the *next* ponder session (design F8: "the next turn's prediction
// must read the pondered tree") without re-running `buildRoot`/`rootGame`.
// `game` is the position BEFORE this entry's own `result.action` -- the
// exact root `prepareFast` would have reconstructed for the request this
// entry answers.
interface PonderContext {
  game: Game;
  historyBase: string[];
  // KOM-1104: the EFFECTIVE start (request.start ?? ADVANCED_POSITION) the
  // ponder key was computed against -- carried so the NEXT prediction, after
  // a hit continues pondering (F8), keys with the same start the human's
  // actual request will normalize to.
  start: string;
  // KOM-1105/KOM-1203: the EFFECTIVE, resolved `RulesConfig` (C2 key, folded
  // in via `rulesConfigKey`) -- see `PonderStartCtx.rules`.
  rules: RulesConfig;
  // KOM-1162: the EFFECTIVE plane-map version (C2 key) -- see
  // `PonderStartCtx.planeMapVersion`.
  planeMapVersion: PlaneMapVersion;
  sims: number;
  batch: number;
  draftmin: number;
  discovery: boolean;
  mlhConfig: MlhConfig;
  maxMs: number | null;
}

// Everything `maybeStartPonder` needs to kick off a ponder session, from
// either a fresh search (`attemptFast`) or a ponder hit continuing onto the
// pondered tree (F8) -- the two call sites differ only in where these come
// from.
interface PonderStartCtx {
  tree: MatchTree;
  batchEvaluator: MlhBatchEvaluator;
  preMoveGame: Game;
  chosenAction: number;
  historyBase: string[];
  // KOM-1104: the EFFECTIVE start (C2 key) -- see `PonderContext.start`.
  start: string;
  // KOM-1105/KOM-1203: the EFFECTIVE, resolved `RulesConfig`, included (via
  // `rulesConfigKey`) in both `ponderKey` sites -- see `PonderContext.rules`.
  rules: RulesConfig;
  // KOM-1162: the EFFECTIVE plane-map version (request `planeMapVersion` ??
  // 3), in both `ponderKey` sites and on every ponder root state, so a
  // pondered search encodes exactly as the real request would.
  planeMapVersion: PlaneMapVersion;
  sims: number;
  batch: number;
  draftmin: number;
  discovery: boolean;
  mlhConfig: MlhConfig;
  maxMs: number | null;
  // KOM-1195 design section 6 (re-review N2): `= target.stoppable` at the
  // search that started this ponder session. True makes `runPonderSession`
  // ponder on the chunked loop (the no-deadline form of the composed
  // `shouldStop`), so an opted-in history play never ponders on a continuous
  // path whose reply can differ from its own chunked one.
  chunked: boolean;
}

// KOM-1102 item 2: what `handlePonderNow` needs to validate and resume the
// ponder-start path `maybeStartPonder` would have taken right after the
// last reply -- `ctx` is that EXACT call's own argument (reused verbatim,
// never rebuilt), `tree`/`evaluationsAtReply` are the identity check for
// "no real search since" (the SAME `MatchTree` instance, with the SAME
// cumulative evaluator-call count `tree.evaluations` had at snapshot time --
// any real search, on this tree or a ponder hit adopting a different one,
// changes one or the other), and `fen`/`history` are the resulting position
// the page must still agree on.
interface StandingPonder {
  ctx: PonderStartCtx;
  tree: MatchTree;
  evaluationsAtReply: number;
  fen: string;
  history: string[];
}

// N1 fix (design 3.3): `SubtreeValueBias.update` mutates a stored entry's
// `errSum` Float64Array in place, and inserts fresh entries for unseen keys
// -- a shallow `Map` copy would share the mutated arrays, and a
// merge-restore would keep post-snapshot keys. `snapshotBias` deep-copies
// every entry; `restoreBias` replaces the whole map (never merges), itself
// deep-copying again so neither the live table nor a kept snapshot ever
// alias the same `Float64Array`. The fallback that calls these (a retry
// after a backend failure) is U6b; this unit only provides them, tested
// directly.
export function snapshotBias(tree: MatchTree): Map<number, BiasEntry> {
  const snapshot = new Map<number, BiasEntry>();
  for (const [key, entry] of tree.bias.entries) {
    snapshot.set(key, { errSum: Float64Array.from(entry.errSum), weightSum: entry.weightSum, gen: entry.gen });
  }
  return snapshot;
}

export function restoreBias(tree: MatchTree, snapshot: Map<number, BiasEntry>): void {
  const restored = new Map<number, BiasEntry>();
  for (const [key, entry] of snapshot) {
    restored.set(key, { errSum: Float64Array.from(entry.errSum), weightSum: entry.weightSum, gen: entry.gen });
  }
  tree.bias.entries = restored;
}

// KOM-1195 design section 6 (re-review N4): the WHOLE bias state -- the
// table plus the rules key/plane-map version `resetBiasOnRulesChange` last
// recorded. Restoring the table alone would hand an A-era table back under
// key B (and the next B request would keep it), breaking KOM-1105's rule.
interface BiasState {
  table: Map<number, BiasEntry> | null;
  rulesKey: string | null;
  planeMapVersion: PlaneMapVersion;
}

export class WorkerEngine {
  private leafEvaluator: MlhLeafEvaluator | MlhLeafEvaluatorAsync;
  private modelSha256: string | null;
  private clock: () => number;
  private _tree: MatchTree | null = null;
  // Fast-mode U6a (design 6.3 M-4): a promise-chain queue so a second
  // `handleFast` call never starts a search on the one `MatchTree` while an
  // earlier one is still running -- each call's work is chained onto the
  // previous call's completion, never onto its rejection (handleFast/runFast
  // never throw, so this promise itself never rejects).
  private queue: Promise<void> = Promise.resolve();
  // Fast-mode U6b: the engine's own idea of "the current backend", set by
  // `init()` and updated in place by a mid-search fallback (design 3.3).
  // `null` means `init()` was never called -- `handle()`/`handleAsync()`/
  // `handleFast()` all still work (the fake default evaluator, or whatever
  // evaluator a caller passes explicitly to `handleFast`), just with no
  // `backend` stamped on the reply.
  private _backend: BackendKind | null = null;
  private _threads = 1;
  private _fastEvaluator: MlhBatchEvaluator | null = null;
  private _fallbackBuilder: BatchBackendBuilder | null = null;
  // KOM-1096 item 2: everything `writeFallbackCache` needs to recompute the
  // SAME `backendCacheKey` `init()`'s own probe used, so a later ladder
  // fallback (init-time session build, or mid-search) marks the one cache
  // row that call's `probeBackend` read/wrote. `null` until `init()` runs.
  private _cacheCtx: {
    gpu?: GpuLike | undefined;
    isolated: boolean;
    hardwareConcurrency: number;
    modelSha256: string;
    storage?: StorageLike | undefined;
    now: () => number;
  } | null = null;
  // Fast-mode U7 (design section 5; ticket KOM-1100): pondering settings/
  // gating (M-6/F7, default on), the cache C2 keys into, and the current
  // in-flight session's abort signal (`null` when nothing is pondering).
  // `_ponderTask` is exposed (via `ponderingPromise()`) purely for tests --
  // nothing here ever awaits its own ponder work.
  private _ponderSettings: PonderSettings = { ...DEFAULT_PONDER_SETTINGS };
  private _ponderGating: PonderGating = { ...DEFAULT_PONDER_GATING };
  private _ponderCache = new PonderCache<PonderContext>();
  private _ponderSignal: PonderSignal | null = null;
  private _ponderTask: Promise<void> | null = null;
  // KOM-1102 item 2: the standing-tree snapshot `handlePonderNow` validates
  // against -- set by `maybeStartPonder` on EVERY call (regardless of
  // whether pondering itself actually started), so it always reflects "the
  // tree the last reply left behind," gated or not. `null` before the first
  // reply that carried a history array.
  private _lastStanding: StandingPonder | null = null;
  // KOM-1105/KOM-1203: the previous request's effective `rulesConfigKey`,
  // `null` before the first search -- lets `resetBiasOnRulesChange` tell "no
  // prior request" apart from "prior request used the default profile".
  private _lastRulesKey: string | null = null;
  // KOM-1162: the previous request's effective plane-map version (only read
  // once `_lastRulesKey` is non-null).
  private _lastPlaneMapVersion: PlaneMapVersion = DEFAULT_PLANE_MAP_VERSION;
  // KOM-1203 unit 6 (section 4, N3): engine-scoped for this engine's whole
  // lifetime, set once at construction and installed on the root `Game`
  // every `buildRoot` call builds (never per-request).
  private _scoreTerminal?: ScoreTerminalHook | undefined;
  // KOM-1195 design section 3 (U4a): every fast-mode search's `StopTarget`,
  // keyed by request id, for its whole life (registered in `openStopTarget`,
  // released on every path in `closeStopTarget`).
  private readonly _stopTargets = new Map<string | number, StopTarget>();
  // KOM-1195 design section 4 (U4b): `engine.ts` has no `self` reference and
  // must keep it that way -- `attemptFast`'s cadence wrapper calls this
  // option instead of posting directly. `undefined` when the host never
  // wired one (no progress event is ever built in that case).
  private readonly _onProgress: ((event: WorkerProgressEvent) => void) | undefined;

  constructor(
    opts: {
      leafEvaluator?: MlhLeafEvaluator | MlhLeafEvaluatorAsync;
      modelSha256?: string | null;
      clock?: () => number;
      scoreTerminal?: ScoreTerminalHook | undefined;
      onProgress?: (event: WorkerProgressEvent) => void;
    } = {},
  ) {
    this.leafEvaluator = opts.leafEvaluator ?? makeFakeMlhEvaluator(ACTION_SIZE);
    this.modelSha256 = opts.modelSha256 ?? null;
    this.clock = opts.clock ?? WALL_CLOCK;
    this._scoreTerminal = opts.scoreTerminal;
    this._onProgress = opts.onProgress;
  }

  // KOM-1226: lets a host that built this engine before the model sha was
  // known (installWorkerHost's single long-lived instance) record the
  // init-verified sha once init succeeds, without `init()` itself stamping
  // every caller's engine (test/kom1195_record_goldens.ts's `initWasmMt`
  // among them) with a sha its own golden fixture never carried.
  setVerifiedModelSha256(sha: string): void {
    this.modelSha256 = sha.toLowerCase();
  }

  // The engine's one retained tree (null before the first search); exposed
  // for diagnostics and tests, as EngineServer exposes its player.
  get tree(): MatchTree | null {
    return this._tree;
  }

  // Fast-mode U6b: the backend/evaluator `init()` picked, or whatever a
  // fallback most recently switched to -- `null` before the first `init()`.
  get currentBackend(): BackendKind | null {
    return this._backend;
  }

  get currentBatchEvaluator(): MlhBatchEvaluator | null {
    return this._fastEvaluator;
  }

  // The fake-leaf default (`this.leafEvaluator`, same one `handle()` uses)
  // mapped over a batch's rows -- what `handleFast` runs against when no
  // `init()` has installed a real one, so the real worker wiring below
  // always has *some* evaluator to pass it.
  defaultBatchEvaluator(): MlhBatchEvaluator {
    const leaf = this.leafEvaluator;
    return (states: MatchState[]) => Promise.all(states.map((state) => Promise.resolve(leaf(state))));
  }

  // Never throws: every failure becomes `{ id, err }` (the Python match
  // server's own handle_request contract -- a request never kills the server).
  handle(request: WorkerRequest): WorkerReply {
    try {
      if (isAnalyzeRequest(request)) return { id: request.id, err: ANALYZE_EXACT_MODE_ERR };
      const invalid = validateProgressOptions(request);
      if (invalid !== null) return { id: request.id, err: invalid };
      const prepared = this.prepare(request, (inner, config, ctx) =>
        wrapMlhEvaluator(inner as MlhLeafEvaluator, config, ctx),
      );
      if (!('tree' in prepared)) return prepared;
      const result = search(prepared.tree, prepared.state, prepared.searchConfig, prepared.searchOpts);
      return this.finish(request, prepared, result);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      return { id: request.id, err: message };
    }
  }

  // KOM-1078: the resumable entry point. Identical request handling to
  // `handle()`, apart from awaiting the search (searchAsync) instead of
  // calling it -- which lets `this.leafEvaluator` answer asynchronously,
  // suspending the search exactly at each evaluation and resuming it with
  // no replay, unlike the e2e_driver.mjs cache-miss workaround this
  // supersedes. Never throws, for the same reason `handle()` doesn't.
  async handleAsync(request: WorkerRequest): Promise<WorkerReply> {
    try {
      if (isAnalyzeRequest(request)) return { id: request.id, err: ANALYZE_EXACT_MODE_ERR };
      const invalid = validateProgressOptions(request);
      if (invalid !== null) return { id: request.id, err: invalid };
      const prepared = this.prepare(request, (inner, config, ctx) => wrapMlhEvaluatorAsync(inner, config, ctx));
      if (!('tree' in prepared)) return prepared;
      const result = await searchAsync(prepared.tree, prepared.state, prepared.searchConfig, prepared.searchOpts);
      return this.finish(request, prepared, result);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      return { id: request.id, err: message };
    }
  }

  // Fast-mode U6b (design section 3, 3.3; section 8 U6): loads the model
  // (`env.createBatchBackend`, real ORT wiring built out of band -- this
  // method never dynamic-imports a URL itself), picks a backend with U5's
  // probe/cache (`env.sessionFactory`, backend.ts), and installs the
  // result as the engine's own current backend/evaluator -- every later
  // `handleFast`/`defaultBatchEvaluator` call sees it via `currentBackend`/
  // `currentBatchEvaluator`. Never throws, same contract as `handle()`.
  async init(request: WorkerInitRequest, env: WorkerInitEnv): Promise<WorkerInitReply> {
    try {
      if (typeof request.modelSha256 !== 'string') throw new Error('modelSha256 must be a string');
      const isolated = env.isolated ?? false;
      const hardwareConcurrency = env.hardwareConcurrency ?? 1;
      // A 'wasm' preference skips WebGPU probing outright -- requestAdapter()
      // is never even called, not just "loses the ladder".
      const gpu = request.backend === 'wasm' ? undefined : env.gpu;
      const probe = await probeBackend({
        factory: env.sessionFactory,
        gpu,
        isolated,
        hardwareConcurrency,
        threadsOverride: request.threads,
        maxThreads: request.maxThreads,
        modelSha256: request.modelSha256,
        storage: env.storage,
        cachedRecord: request.cachedRecord,
        now: env.now,
        threadsTimeoutMs: request.threadsTimeoutMs,
      });
      this._fallbackBuilder = env.createBatchBackend;
      this._cacheCtx = {
        gpu,
        isolated,
        hardwareConcurrency,
        modelSha256: request.modelSha256,
        storage: env.storage,
        now: env.now ?? Date.now,
      };

      try {
        const evaluator = await env.createBatchBackend(probe.backend, probe.threads);
        this._backend = probe.backend;
        this._threads = probe.threads;
        this._fastEvaluator = evaluator;
        return { id: request.id, backend: probe.backend, threads: probe.threads, batch: probe.chosenBatch, probeRecord: probe.record };
      } catch (buildError) {
        // KOM-1096 item 1/2: building the probe's OWN chosen backend can
        // still fail (a session-creation error the probe's own warm-up
        // never hit) -- walk the same ladder a mid-search failure uses
        // (`fallbackLadder`) instead of surfacing a bare `{id, err}`.
        const message = buildError instanceof Error ? buildError.message : String(buildError);
        const advanced = await this.fallbackLadder(env.createBatchBackend, probe.backend, probe.threads, message);
        if (advanced === null) throw buildError;
        this._backend = advanced.backend;
        this._threads = advanced.threads;
        this._fastEvaluator = advanced.evaluator;
        // Exact mode (design 3.1 item 3): wasm at 1 thread never batches.
        const batch = advanced.backend === 'wasm' && advanced.threads === 1 ? 1 : probe.chosenBatch;
        return {
          id: request.id,
          backend: advanced.backend,
          threads: advanced.threads,
          batch,
          fallback: { from: advanced.failedFrom, reason: advanced.message },
          probeRecord: advanced.record ?? probe.record,
        };
      }
    } catch (error) {
      // KOM-1214: a bounded wasm threads start/warm-up that never settled --
      // typed so the host knows to start a FRESH worker with maxThreads: 1
      // rather than retry this one (ORT reads numThreads once per worker).
      if (error instanceof ThreadsTimeoutError) {
        return { id: request.id, err: error.message, code: 'threads-timeout', probeRecord: error.record };
      }
      const message = error instanceof Error ? error.message : String(error);
      return { id: request.id, err: message };
    }
  }

  // KOM-1096 item 4 (design 6.3 M-4/M-6c): `init`'s own counterpart to
  // `handleFast` -- queued onto the SAME `this.queue`, so an `init` posted
  // while a search is still running never swaps `_backend`/`_fastEvaluator`
  // out from under it; it runs only once the running search's queued work
  // has settled.
  queueInit(request: WorkerInitRequest, env: WorkerInitEnv): Promise<WorkerInitReply> {
    const run = this.queue.then(() => this.init(request, env));
    this.queue = run.then(() => undefined);
    return run;
  }

  // The product-identity defaulting `buildRoot` and (Fast-mode U7) a ponder
  // cache lookup both need -- factored out so the two can never read
  // different defaults for the same request field.
  private effectiveIdentity(request: WorkerRequest): { discovery: boolean; draftmin: number; mlhConfig: MlhConfig } {
    return {
      discovery: request.config?.discovery ?? PRODUCTION_IDENTITY.discovery,
      draftmin: request.config?.draftmin ?? PRODUCTION_IDENTITY.draftmin,
      mlhConfig: {
        lam: request.config?.mlh?.lam ?? PRODUCTION_IDENTITY.mlh.lam,
        tau0: request.config?.mlh?.tau0 ?? PRODUCTION_IDENTITY.mlh.tau0,
        tau1: request.config?.mlh?.tau1 ?? PRODUCTION_IDENTITY.mlh.tau1,
        slope: request.config?.mlh?.slope ?? PRODUCTION_IDENTITY.mlh.slope,
      },
    };
  }

  // Validates the request and builds the root state -- everything
  // `answer()` used to do before installing an evaluator, shared by
  // `prepare()` (single-leaf) and `prepareFast()` (batch). Returns a
  // terminal `WorkerReply` directly (the M5 turn-cap path) when there is
  // nothing to search.
  private buildRoot(request: WorkerRequest): RootBuild | WorkerReplyErr {
    const started = Date.now();
    const clock = request.timing === true ? this.clock : null;
    const t0 = clock === null ? 0 : clock();
    if (typeof request.fen !== 'string') throw new Error('fen must be a string');
    // KOM-1202 rule 5 (design section 3): `level` is `handleFast`-only --
    // `handle()`/`handleAsync()` go straight to `buildRoot`, which reads
    // only `sims`, so a `{ sims, level }` request there would otherwise
    // silently ignore `level`. A resolved request never carries `level`, so
    // this never fires on the `handleFast` path.
    if (request.level !== undefined) {
      throw new Error('level is only supported by handleFast');
    }
    const sims = request.sims;
    if (!Number.isInteger(sims) || sims < 0) {
      throw new Error(`sims must be a non-negative integer, got ${sims}`);
    }
    // KOM-1105/KOM-1203: validated before the root/tree is built at all --
    // an invalid `rules` (string or RulesConfig object) never reaches
    // `rootGame`.
    const config = toRulesConfig(request.rules);
    // KOM-1162: likewise validated before any root is built; carried on the
    // root state, which every leaf evaluator encodes under.
    const planeMapVersion = parsePlaneMapVersion(request.planeMapVersion);

    const { discovery, draftmin, mlhConfig } = this.effectiveIdentity(request);

    // N1: the ORIGINAL request value (string, RulesConfig object, or
    // undefined), never the resolved `config` -- `rootGame`/`gameFromHistory`
    // resolve and default it themselves, and pass the original through into
    // `new Game` (Game.rules round-trips exactly what the caller passed).
    const { game, stamp } = rootGame(request.fen, request.history, request.start, request.rules);
    // KOM-1203 unit 6 (section 4, N3): installed on the root Game only --
    // every Game reachable from the search tree below carries the same
    // reference via `Game.clone()`, so `winState()` stays zero-argument.
    game.scoreTerminal = this._scoreTerminal;
    const rootBoard = game.board;

    const state = new MatchGameState(game, { minPlacements: draftmin, isRoot: true, planeMapVersion });

    if (state.isTerminal()) {
      // M5 (design 2.7 "Turn cap"): under komugi/gungijs, a request whose
      // history reaches the engine's own 512-ply cap has a terminal root,
      // which the search refuses (search/search.ts throws on it). A
      // terminal root is not only the cap -- a `fen`-only request (no
      // history) at turns = 0 can already be terminal by checkmate,
      // stalemate, repetition or insufficient material -- so turns >=
      // plyCap is checked explicitly below rather than assumed.
      // KOM-1107 rule (7)/KOM-1203 F2(d): `config.plyCap === null` (ligungi)
      // means there is no ply cap in winState(), so turns >= plyCap no
      // longer makes a terminal root the cap's doing by itself -- a
      // terminal root under a null-plyCap config is terminal for a genuine
      // reason (checkmate, repetition, insufficient material), not the cap,
      // so it falls through to the generic "no move to search" error below
      // instead.
      if (config.plyCap !== null && game.turns >= config.plyCap) {
        return { id: request.id, err: 'turn_cap' };
      }
      throw new Error(`root position is terminal at turns=${game.turns}: no move to search`);
    }

    const treeConfig: SearchConfig = { ...BASE_MATCH_CONFIG, playouts: sims };
    return {
      started,
      clock,
      t0,
      sims,
      discovery,
      draftmin,
      mlhConfig,
      rootBoard,
      stamp,
      state,
      treeConfig,
      turns: game.turns,
      config,
    };
  }

  // KOM-1105: the bias table persists across requests by design (the tree's
  // own nodes already clear every search, `search.ts`'s own `tree.clear()`);
  // this is the one thing that must NOT survive a `rules` change, since a
  // bias entry's mean WDL error was learned under the previous profile's
  // terminal scoring. `null` on the first-ever request (nothing stored yet,
  // and no tree to clear regardless).
  // KOM-1162: a `planeMapVersion` change clears it too -- the entries' WDL
  // errors were learned against the other encoding's network outputs.
  // KOM-1203 F2(c): keyed off `rulesConfigKey(config)`, not `_lastRules !==
  // rules` object/string identity -- a host rebuilding an equal-value
  // `RulesConfig` per request must not spuriously clear the bias table.
  // Unit 6: also folds in `hookKeyComponent(this._scoreTerminal)` -- empty
  // when this engine has no installed hook, so the key is byte-identical to
  // before unit 6 whenever no hook is installed (AC1).
  private resetBiasOnRulesChange(config: RulesConfig, planeMapVersion: PlaneMapVersion): void {
    const key = rulesConfigKey(config) + hookKeyComponent(this._scoreTerminal);
    if (
      this._tree !== null &&
      this._lastRulesKey !== null &&
      (this._lastRulesKey !== key || this._lastPlaneMapVersion !== planeMapVersion)
    ) {
      this._tree.bias.clear();
    }
    this._lastRulesKey = key;
    this._lastPlaneMapVersion = planeMapVersion;
  }

  // Installs the wrapped single-leaf evaluator -- everything `answer()` used
  // to do before calling `search()`. `wrap` is `handle`/`handleAsync`'s own
  // choice of wrapMlhEvaluator/wrapMlhEvaluatorAsync; `prepare()` itself
  // stays agnostic to sync vs async.
  private prepare(
    request: WorkerRequest,
    wrap: (
      inner: MlhLeafEvaluator | MlhLeafEvaluatorAsync,
      config: MlhConfig,
      ctx: MlhContext,
    ) => MaybeAsyncWdlEvaluator,
  ): PreparedSearch | WorkerReply {
    const built = this.buildRoot(request);
    if ('err' in built) return built;
    const { started, clock, t0, sims, discovery, draftmin, mlhConfig, rootBoard, stamp, state, treeConfig, turns, config } =
      built;
    this.resetBiasOnRulesChange(config, state.planeMapVersion);

    const mlhCtx = new MlhContext();
    const evaluator = wrap(this.leafEvaluator, mlhConfig, mlhCtx);
    // search() / searchAsync() clear the nodes themselves (search.ts's
    // tree.clear()); the bias table survives, as it does in the Python
    // server's retained tree.
    if (this._tree === null) {
      this._tree = new MatchTree(treeConfig, evaluator);
    } else {
      this._tree.config = treeConfig;
      this._tree.evaluator = evaluator;
    }

    const searchOpts: PreparedSearch['searchOpts'] = {
      discovery,
      ply: turns,
      tacticalTransitionCap: DEFAULT_TRANSITION_CAP,
    };
    if (clock !== null) searchOpts.clock = clock;
    return {
      tree: this._tree,
      state,
      searchConfig: { ...treeConfig, playouts: sims },
      searchOpts,
      rootBoard,
      stamp,
      sims,
      draftmin,
      discovery,
      mlhConfig,
      started,
      timing: clock === null ? null : { clock, t0, prepareMs: clock() - t0 },
      config,
    };
  }

  // KOM-1219 design section 2.1 step 2: `batch`/`maxMs` validation, shared
  // with `tryForced` rather than copied -- a width-1 request with a
  // malformed `batch`/`maxMs` gets the same `{id, err}` a width>1 request of
  // the same shape gets, instead of the field being silently ignored because
  // the search never ran.
  private validateFastParams(request: WorkerRequest): { batch: number; maxMs: number | undefined } {
    const batch = request.batch ?? 1;
    if (!Number.isInteger(batch) || batch < 1) {
      throw new Error(`batch must be a positive integer, got ${batch}`);
    }
    const maxMs = request.maxMs;
    if (maxMs !== undefined && (!Number.isFinite(maxMs) || maxMs < 0)) {
      throw new Error(`maxMs must be a non-negative number, got ${maxMs}`);
    }
    return { batch, maxMs };
  }

  // KOM-1219 design section 2.1: certifies a width-1 root off the legal mask
  // alone, before `ensureRootGen`'s one unconditional evaluator call --
  // AC1's "zero evaluator calls". Wraps its own `buildRoot`/validation work
  // in its own try/catch so a throw never escapes into `this.queue` (review
  // V3/V4: the design-review-era snippet let it escape and wedged every
  // later `handleFast` call). Order: `buildRoot`, then `soleLegalAction`
  // (width > 1 returns null before any batch/maxMs check, so that error
  // still comes from `prepareFast`'s own `validateFastParams` call), then --
  // width 1 only -- `validateFastParams`.
  private tryForced(request: WorkerRequest): WorkerReply | null {
    try {
      const built = this.buildRoot(request);
      if ('err' in built) return built;
      const action = soleLegalAction(built.state);
      if (action === null) return null;
      this.validateFastParams(request);
      // Design 2.1 step 4: a following `handlePonderNow` must see
      // `no_prior_reply` rather than resume from a stale standing snapshot
      // whose evaluation count a 0-evaluation forced reply leaves unchanged.
      // `tryForced` never calls `maybeStartPonder` (no root, nothing to
      // predict) and never touches the ponder cache.
      this._lastStanding = null;
      return this.finishForced(request, built, action);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      return { id: request.id, err: message };
    }
  }

  // Fast-mode U6a: like `prepare()`, but installs ONE shared `MlhContext` as
  // BOTH `tree.evaluator` (the root call) and `tree.batchEvaluator` (the
  // playout loop's batched leaves) -- design 6.3 S-1's normative fix: the
  // root call must go through the SAME wrapped batch evaluator, never an
  // independent single-leaf wrapper, or `ctx.gated` stops being
  // deterministic. Also validates `batch`/`maxMs` and builds the
  // `shouldStop` closure a deadline request needs (design 1.7 C3, 4.3).
  private prepareFast(
    request: WorkerRequest,
    batchEvaluator: MlhBatchEvaluator,
    target: StopTarget,
    deadlineStart: number,
  ): PreparedFastSearch | WorkerReply {
    const built = this.buildRoot(request);
    if ('err' in built) return built;
    const { started, clock, t0, sims, discovery, draftmin, mlhConfig, rootBoard, stamp, state, treeConfig, turns, config } =
      built;
    this.resetBiasOnRulesChange(config, state.planeMapVersion);

    const { batch, maxMs } = this.validateFastParams(request);

    // KOM-1195 design section 3 (U4a, re-review N3/advisor C1): the ONE
    // composed `shouldStop`, built from ordered contributors. A contributor
    // that fires writes its reason into `stopState` (read by `finish()` once
    // the search returns); the first one to fire wins.
    const stopState: { reason: StopReason | null } = { reason: target.analysis ? 'ceiling' : null };
    const contributors: Array<() => StopReason | null> = [];
    if (target.stoppable) contributors.push(() => (target.stopped ? 'stop' : null));
    if (maxMs !== undefined) contributors.push(() => (this.clock() - deadlineStart >= maxMs ? 'deadline' : null));
    const shouldStop =
      contributors.length === 0
        ? undefined
        : () => {
            for (const contributor of contributors) {
              const reason = contributor();
              if (reason !== null) {
                stopState.reason = reason;
                return true;
              }
            }
            return false;
          };

    const mlhCtx = new MlhContext();
    const wrappedBatch = wrapMlhBatch(batchEvaluator, mlhConfig, mlhCtx);
    const rootEvaluator: MaybeAsyncWdlEvaluator = (s: MatchState) =>
      Promise.resolve(wrappedBatch([s])).then((rows) => rows[0]!);

    if (this._tree === null) {
      this._tree = new MatchTree(treeConfig, rootEvaluator, wrappedBatch);
    } else {
      this._tree.config = treeConfig;
      this._tree.evaluator = rootEvaluator;
      this._tree.batchEvaluator = wrappedBatch;
    }

    const searchOpts: PreparedSearch['searchOpts'] = {
      discovery,
      ply: turns,
      tacticalTransitionCap: DEFAULT_TRANSITION_CAP,
    };
    if (clock !== null) searchOpts.clock = clock;
    if (shouldStop !== undefined) searchOpts.shouldStop = shouldStop;
    // KOM-1195 design section 3 (review F2, re-review N1): only an opted-in
    // request's async search yields to the macrotask queue between chunks --
    // an un-opted request (even with a `maxMs` deadline) keeps today's exact
    // timing (AC1).
    if (target.stoppable) searchOpts.yieldToHost = nextMacrotask;

    return {
      tree: this._tree,
      state,
      searchConfig: { ...treeConfig, playouts: sims },
      searchOpts,
      rootBoard,
      stamp,
      sims,
      draftmin,
      discovery,
      mlhConfig,
      started,
      timing: clock === null ? null : { clock, t0, prepareMs: clock() - t0 },
      config,
      batch,
      opted: target.stoppable,
      stopState,
    };
  }

  // KOM-1195 design section 3 (U4a; re-review N8/N10): opens this search's
  // `StopTarget`, synchronously, before any await -- a `stop` arriving while
  // the request is still queued/initialising is honoured at the first chunk
  // boundary once the search actually starts. A duplicate in-flight id is
  // rejected (the daemon's own `{ id, err }` shape) only when the live
  // request or this one opted in; when neither did, the new target is built
  // but left OUT of the map (so the live un-opted target stays discoverable
  // by `stop`, and this call's own `closeStopTarget` is a harmless no-op).
  openStopTarget(request: WorkerRequest | WorkerLevelRequest | WorkerAnalyzeRequest): StopTarget | WorkerReplyErr {
    const stoppable = isOptedIn(request);
    const analysis = request.analyze === true;
    const target: StopTarget = { id: request.id, stoppable, analysis, stopped: false };
    const existing = this._stopTargets.get(request.id);
    if (existing !== undefined) {
      if (existing.stoppable || stoppable) {
        return { id: request.id, err: 'duplicate in-flight search id' };
      }
      return target;
    }
    this._stopTargets.set(request.id, target);
    return target;
  }

  // Idempotent: deletes only if the map still holds this SAME object, so a
  // double close (the listener's own `finally` plus `handleFast`'s) is
  // harmless, and a close for a target that was never registered (the
  // duplicate-un-opted case above) is a no-op.
  closeStopTarget(target: StopTarget): void {
    if (this._stopTargets.get(target.id) === target) {
      this._stopTargets.delete(target.id);
    }
  }

  // KOM-1195 design section 3: `stop`'s own handler, same style as
  // `handlePonderSettings` -- never posts the TARGET's reply, only flips
  // `target.stopped`, which the composed `shouldStop`'s stop contributor
  // reads at the next chunk boundary.
  handleStop(request: WorkerStopRequest): WorkerStopReply {
    const target = this._stopTargets.get(request.target);
    if (target === undefined) return { id: request.id, type: 'stop', acknowledged: false, reason: 'unknown-target' };
    if (!target.stoppable) return { id: request.id, type: 'stop', acknowledged: false, reason: 'not-stoppable' };
    target.stopped = true;
    return { id: request.id, type: 'stop', acknowledged: true };
  }

  // Fast-mode U6a (design section 8 U6, 6.3 M-4): `handle()`'s batch/async
  // counterpart. `batchEvaluator` is the raw (pre-MLH) batch leaf evaluator
  // -- U6b's init/model/backend wiring decides what that is; this unit only
  // wires it through. Queued behind any earlier `handleFast` call so a
  // second request never starts a search on the one `MatchTree` mid-search.
  // Never throws, for the same reason `handle()` doesn't.
  // KOM-1195 design section 3: `target` lets a caller that already opened one
  // (the real `worker_host.ts` listener, U5) adopt it; called directly, as
  // every existing test does, this opens its own synchronously at entry,
  // before its first await.
  handleFast(
    request: WorkerRequest | WorkerLevelRequest | WorkerAnalyzeRequest,
    batchEvaluator: MlhBatchEvaluator,
    target?: StopTarget,
  ): Promise<WorkerReply> {
    const opened = target ?? this.openStopTarget(request);
    if ('err' in opened) return Promise.resolve(opened);
    const stopTarget = opened;
    // Design 5.3/6.3 M-4: "a real request always aborts pondering first via
    // the same signal" -- synchronously, before this request even joins the
    // queue, so pondering never races a real search on the one `MatchTree`.
    this.abortPonder();
    const run = this.queue
      .then(() => {
        // KOM-1105: validated before the ponder cache lookup AND `buildRoot`
        // (the latter re-validates inside `runFast` -> `attemptFast` ->
        // `prepareFast`, harmlessly redundant once this has already passed).
        // KOM-1202 (design section 3 "Resolution mechanics"): `resolveRequest`
        // runs here too, at the top of the queue callback -- before
        // `lookupPonder`, not in `prepareFast` (too late: a level request
        // would miss ponder hits and ponder uncapped, review F1/F2) -- and
        // sees `_backend`/`_threads` after any queued `init()` or earlier
        // fallback, not the values when the message arrived. The resolved
        // request, with `level` removed, is what both `lookupPonder` and
        // `runFast` get from here on.
        let resolved: WorkerRequest;
        try {
          toRulesConfig(request.rules);
          parsePlaneMapVersion(request.planeMapVersion);
          resolved = this.resolveRequest(request);
        } catch (error) {
          const message = error instanceof Error ? error.message : String(error);
          return { id: request.id, err: message };
        }
        // KOM-1195 design section 2 (retry hardening): validated BEFORE the
        // ponder cache lookup -- `ponderKey` does not fold in `topN`/
        // `progressEvery*`, so an invalid value here must never reach
        // `lookupPonder` and come back a truncated success.
        const invalid = validateProgressOptions(resolved);
        if (invalid !== null) return { id: request.id, err: invalid };
        // KOM-1219 design section 2.1 (D1/D2): checked before `lookupPonder`
        // on every `handleFast` path, ponder hit or not -- a width-1 root is
        // decided before the ponder cache is even consulted. Excluded for
        // analysis, a positive `topN`, or a progress cadence (section 5.1,
        // 7): those modes keep today's path unchanged, byte-identical.
        if (
          !stopTarget.analysis &&
          (resolved.topN === undefined || resolved.topN <= 0) &&
          resolved.progressEveryPlayouts === undefined &&
          resolved.progressEveryMs === undefined
        ) {
          const forced = this.tryForced(resolved);
          if (forced !== null) return forced;
        }
        // KOM-1195 design section 6 (review F4): an analysis never takes a
        // ponder-cache hit (nor starts pondering, `attemptFast`).
        const hit = stopTarget.analysis ? null : this.lookupPonder(resolved, batchEvaluator, stopTarget);
        if (hit !== null) return hit;
        return this.runFast(resolved, batchEvaluator, stopTarget);
      })
      // KOM-1195 design section 3: released on every path -- `prepareFast`
      // validation errors, evaluator errors, the `BackendError` fallback
      // retry, and a ponder-cache hit alike -- before the reply above
      // resolves, so no `stop` can be acknowledged for a search that has
      // already replied.
      .finally(() => this.closeStopTarget(stopTarget));
    this.queue = run.then(() => undefined);
    return run;
  }

  // KOM-1202 (design section 3, rules 1-4): resolves a raw `WorkerRequest`
  // (pass-through) or a `WorkerLevelRequest` (looked up against the
  // engine's currently selected backend) into one effective `WorkerRequest`
  // with `sims`/`batch`/`maxMs` set and no `level`. Throws on every
  // validation failure -- caught by `handleFast`'s queue callback, the same
  // way `toRulesConfig`/`parsePlaneMapVersion` already are.
  // KOM-1195 design section 6: widened to a three-way runtime discriminant
  // (an untyped host can send anything); an analysis resolves to an ordinary
  // `sims = nodeCeiling - 1` with `analyze`/`nodeCeiling` removed -- the
  // analysis bit itself travels on the `StopTarget`, opened on the raw
  // request before this runs.
  private resolveRequest(request: WorkerRequest | WorkerLevelRequest | WorkerAnalyzeRequest): WorkerRequest {
    const hasSims = request.sims !== undefined;
    const hasLevel = request.level !== undefined;
    const hasAnalyze = request.analyze === true;
    const modeCount = Number(hasSims) + Number(hasLevel) + Number(hasAnalyze);
    if (modeCount !== 1) {
      throw new Error('exactly one of sims, level, or analyze is required');
    }
    if (hasAnalyze) {
      const analyzeRequest = request as WorkerAnalyzeRequest;
      const nodeCeiling = analyzeRequest.nodeCeiling ?? ANALYSIS_DEFAULT_NODE_CEILING;
      if (!Number.isInteger(nodeCeiling) || nodeCeiling < 1 || nodeCeiling > ANALYSIS_MAX_NODE_CEILING) {
        throw new Error(
          `nodeCeiling must be a positive integer <= ${ANALYSIS_MAX_NODE_CEILING}, got ${analyzeRequest.nodeCeiling}`,
        );
      }
      const { analyze: _analyze, nodeCeiling: _nodeCeiling, ...rest } = analyzeRequest;
      return { ...rest, sims: nodeCeiling - 1 };
    }
    if (!hasLevel) {
      return request as WorkerRequest;
    }
    const levelRequest = request as WorkerLevelRequest;
    if (levelRequest.batch !== undefined || levelRequest.maxMs !== undefined) {
      throw new Error('level cannot be combined with batch or maxMs');
    }
    const name = parseLevel(levelRequest.level);
    if (this._backend === null) {
      throw new Error('level requires a selected backend; call init() first');
    }
    const { sims, batch, maxMs } = resolveLevel(name, { backend: this._backend, threads: this._threads });
    const { level: _level, ...rest } = levelRequest;
    return { ...rest, sims, batch, maxMs };
  }

  // Design 3.3/M-5: a mid-search backend failure (U4's BackendError, or any
  // session error the batch evaluator surfaces as one) abandons the
  // current search, restores the bias table to what it was BEFORE this
  // request started (N1's snapshot/restore, taken here -- before
  // `attemptFast` touches the tree at all, not after the failure), rebuilds
  // the next backend of the ladder via `env.createBatchBackend` (same
  // builder `init()` kept), and re-runs the whole request from scratch
  // (`attemptFast` -> `prepareFast` always installs a FRESH `MlhContext`,
  // M-5's own fix). Only possible when `init()` has run (`_fallbackBuilder`
  // set) and the failed backend still has a next rung; otherwise this
  // degrades to the same generic `{id, err}` reply `handle()` gives.
  // KOM-1195 design section 3 ("one flag across the fallback retry"): `target`
  // is passed unchanged into BOTH `attemptFast` calls below -- it is never
  // re-registered on the retry, so a `stop` that arrived during the failed
  // attempt still applies.
  private async runFast(
    request: WorkerRequest,
    batchEvaluator: MlhBatchEvaluator,
    target: StopTarget,
  ): Promise<WorkerReply> {
    // KOM-1102 item 3: `abortPonder()` (called synchronously by `handleFast`
    // before this ever runs) only flips the signal `runBatchedSearchAbortable`
    // checks before its NEXT `gen.next()` -- the batch already in flight on
    // this same ORT session when the real request arrived is still running,
    // and design 5.3's "an in-flight session.run cannot be cancelled and is
    // awaited regardless" means THIS search's own first session.run must not
    // start until that one has settled. `_ponderTask` never rejects
    // (`maybeStartPonder`'s own `.catch`), so this await adds no new failure
    // path, and it is a no-op (awaiting `null`) whenever nothing was pondering.
    await this._ponderTask;
    // KOM-1195 design section 6 (re-review N4): the whole bias state (table
    // AND rules key/plane-map version), taken before `attemptFast` touches
    // the tree. Restored by the `BackendError` fallback below, and -- for an
    // analysis, which must never change later play -- on every exit path.
    const preState = this.snapshotBiasState();
    try {
      return await this.runFastAttempts(request, batchEvaluator, target, preState);
    } finally {
      if (target.analysis) this.restoreBiasState(preState);
    }
  }

  // KOM-1195 design section 6 (re-review N4): see `BiasState`. A `null`
  // table means there was no tree yet; restoring it empties whatever table
  // the search's freshly built tree grew.
  private snapshotBiasState(): BiasState {
    return {
      table: this._tree !== null ? snapshotBias(this._tree) : null,
      rulesKey: this._lastRulesKey,
      planeMapVersion: this._lastPlaneMapVersion,
    };
  }

  private restoreBiasState(state: BiasState): void {
    if (this._tree !== null) restoreBias(this._tree, state.table ?? new Map<number, BiasEntry>());
    this._lastRulesKey = state.rulesKey;
    this._lastPlaneMapVersion = state.planeMapVersion;
  }

  private async runFastAttempts(
    request: WorkerRequest,
    batchEvaluator: MlhBatchEvaluator,
    target: StopTarget,
    preState: BiasState,
  ): Promise<WorkerReply> {
    // KOM-1202 unit 2c (design section 5, "Fallback does not restart the
    // deadline"): read once, before the first attempt, and reused on the
    // fallback retry -- a fresh `this.clock()` read in `prepareFast` on the
    // retry would give the retried search a second full `maxMs` budget on
    // top of however long the failed attempt already spent.
    const deadlineStart = this.clock();
    try {
      return await this.attemptFast(request, batchEvaluator, target, null, deadlineStart);
    } catch (error) {
      if (!(error instanceof BackendError) || this._fallbackBuilder === null || this._backend === null) {
        const message = error instanceof Error ? error.message : String(error);
        return { id: request.id, err: message };
      }
      const advanced = await this.fallbackLadder(this._fallbackBuilder, this._backend, this._threads, error.message);
      if (advanced === null) {
        return { id: request.id, err: error.message };
      }
      try {
        this.restoreBiasState(preState);
        this._backend = advanced.backend;
        this._threads = advanced.threads;
        this._fastEvaluator = advanced.evaluator;
        const fallbackReason = `${advanced.failedFrom}: ${advanced.message}`;
        // Exact mode (design 3.1 item 3): landing on wasm at 1 thread
        // always retries at batch 1, regardless of what the original
        // request asked for -- batching buys a 1-thread wasm session
        // nothing.
        const retryRequest =
          advanced.backend === 'wasm' && advanced.threads === 1 ? { ...request, batch: 1 } : request;
        return await this.attemptFast(
          retryRequest,
          advanced.evaluator,
          target,
          fallbackReason,
          deadlineStart,
          advanced.record,
        );
      } catch (innerError) {
        const message = innerError instanceof Error ? innerError.message : String(innerError);
        return { id: request.id, err: message };
      }
    }
  }

  // KOM-1096 item 2 (design 3.3): walks `nextLadderRung` from `failedBackend`/
  // `failedThreads` until a rung's `builder` call succeeds or the ladder is
  // exhausted (`null`) -- "every failure path ends at wasm 1 thread, never
  // at err while a lower rung remains," so a rung whose OWN session build
  // throws is skipped in favour of the next one rather than surfacing here.
  // Marks the cache (`writeFallbackCache`) only once a rung's build actually
  // succeeds, so the cache never points at a backend this call never
  // confirmed works.
  private async fallbackLadder(
    builder: BatchBackendBuilder,
    failedBackend: BackendKind,
    failedThreads: number,
    firstErrorMessage: string,
  ): Promise<FallbackAdvance | null> {
    let backend = failedBackend;
    let threads = failedThreads;
    for (;;) {
      const next = nextLadderRung(backend, threads);
      if (next === null) return null;
      try {
        const evaluator = await builder(next.backend, next.threads);
        const record = await this.writeFallbackCache(next);
        return { backend: next.backend, threads: next.threads, evaluator, failedFrom: failedBackend, message: firstErrorMessage, record };
      } catch {
        backend = next.backend;
        threads = next.threads;
      }
    }
  }

  // KOM-1096 item 2 / KOM-1101: builds the record this fallback rung marks --
  // same `backendCacheKey` `init()`'s own probe read (keyed by device shape,
  // not by backend), so a stored `storage` row lands the next `init()` on
  // `next` directly instead of re-probing the backend that just failed. Also
  // the record a caller with no `storage` (a Web Worker) must relay back as
  // `probeRecord` instead -- so this always computes and returns it, and
  // only ever uses `storage` for the side effect of persisting it here too.
  // `null` only when `init()` was never called (`_cacheCtx` unset).
  private async writeFallbackCache(next: { backend: BackendKind; threads: number }): Promise<BackendRecord | null> {
    const ctx = this._cacheCtx;
    if (ctx === null) return null;
    let record: BackendRecord;
    let key: string;
    try {
      const adapter = await resolveAdapterForCache(ctx.gpu);
      key = backendCacheKey({
        modelSha256: ctx.modelSha256,
        adapter,
        hardwareConcurrency: ctx.hardwareConcurrency,
        isolated: ctx.isolated,
      });
      record = {
        backend: next.backend,
        threads: next.threads,
        msPerPosition: {},
        chosenBatch: next.backend === 'wasm' && next.threads === 1 ? 1 : 4,
        probedAt: ctx.now(),
        key,
      };
    } catch {
      // requestAdapter (or key computation) can reject/throw -- losing the
      // cache mark only costs a re-probe next time, same tolerance as before
      // KOM-1101 split this out of the single enclosing try/catch.
      return null;
    }
    if (ctx.storage !== undefined) {
      try {
        ctx.storage.setItem(key, JSON.stringify(record));
      } catch {
        // storage can throw (quota, private mode) -- same tolerance backend.ts's own writeCache has.
      }
    }
    return record;
  }

  private async attemptFast(
    request: WorkerRequest,
    batchEvaluator: MlhBatchEvaluator,
    target: StopTarget,
    fallback: string | null,
    deadlineStart: number,
    probeRecord: BackendRecord | null = null,
  ): Promise<WorkerReply> {
    const prepared = this.prepareFast(request, batchEvaluator, target, deadlineStart);
    if (!('tree' in prepared)) return prepared;
    // KOM-1195 design section 4 (U4b): the cadence wrapper -- `onProgress`
    // itself is a pure counter (search/search.ts); this closure decides
    // WHEN to actually emit, against `progressEveryPlayouts`/
    // `progressEveryMs`, and only when this engine has a constructor
    // `onProgress` to call. No wrapper at all (and so no behavioural change)
    // for a request carrying neither cadence field.
    if (
      this._onProgress !== undefined &&
      (request.progressEveryPlayouts !== undefined || request.progressEveryMs !== undefined)
    ) {
      let lastPlayouts = 0;
      let lastMs = deadlineStart;
      prepared.searchOpts.onProgress = (playoutsUsed: number) => {
        const dueByPlayouts =
          request.progressEveryPlayouts !== undefined &&
          playoutsUsed - lastPlayouts >= request.progressEveryPlayouts;
        const dueByMs = request.progressEveryMs !== undefined && this.clock() - lastMs >= request.progressEveryMs;
        if (!dueByPlayouts && !dueByMs) return;
        lastPlayouts = playoutsUsed;
        lastMs = this.clock();
        this.emitProgress(request.id, prepared.tree, prepared.state.game, playoutsUsed, request.topN);
      };
    }
    const result = await searchFastAsync(
      prepared.tree,
      prepared.state,
      prepared.searchConfig,
      prepared.searchOpts,
      prepared.batch,
    );
    // Fast-mode U7 (design section 5): "after the engine replies, the
    // worker [predicts and ponders]" -- fire-and-forget, on `prepared.tree`
    // directly (this IS `this._tree`; `maybeStartPonder` only ever reads
    // its root/bias, never clears it). Only when the request carried an
    // explicit history array AND it resolves to a key (N5): a malformed
    // descriptor history is a ponder-start skip, same as no history at all,
    // never a throw into `snapshotStanding`'s `[...ctx.historyBase, san]`.
    // KOM-1195 design section 6 (review F4): never after an analysis -- that
    // session would ponder at the analysis budget with no `maxMs`.
    if (!target.analysis && Array.isArray(request.history)) {
      const key = historyKey(request.history, request.start, prepared.config);
      if (key !== null) {
        this.maybeStartPonder({
          tree: prepared.tree,
          batchEvaluator,
          preMoveGame: prepared.state.game,
          chosenAction: result.action,
          historyBase: key,
          start: request.start ?? ADVANCED_POSITION,
          rules: prepared.config,
          planeMapVersion: prepared.state.planeMapVersion,
          sims: prepared.sims,
          batch: prepared.batch,
          draftmin: prepared.draftmin,
          discovery: prepared.discovery,
          mlhConfig: prepared.mlhConfig,
          maxMs: request.maxMs ?? null,
          chunked: target.stoppable,
        });
      }
    }
    const predictedReply = this.predictReply(prepared.tree, prepared.state.game, result.action);
    return this.finish(request, prepared, result, fallback, probeRecord, predictedReply, {
      opted: prepared.opted,
      reason: prepared.stopState.reason,
    });
  }

  // KOM-1102 item 1 (F9/S-5, same as `maybeStartPonder`'s own call): the top
  // predicted human reply to `chosenAction`, read off `tree.root` with no
  // search and no evaluator call -- independent of `shouldPonder`/gating, so
  // this still reports a prediction when pondering itself is off or
  // suspended. `preMoveGame` is cloned (never mutated) so this never changes
  // the tree, the bias table, or the reply this search already computed.
  // `null` when `predictReplies` finds nothing (a terminal `chosenAction`, or
  // a chosen child with no visits).
  private predictReply(tree: MatchTree, preMoveGame: Game, chosenAction: number): PredictedReply | null {
    const root = tree.root;
    if (root === null) return null;
    const [replyAction] = predictReplies(root, chosenAction, 1);
    if (replyAction === undefined) return null;
    const game = preMoveGame.clone();
    game.play(chosenAction);
    const { struct } = decodeAction(game.board, replyAction);
    const san = buildSan(game.board, struct);
    const descriptor = describeRecord(game.board, struct);
    return { action: replyAction, san, descriptor };
  }

  // KOM-1195 design section 5 (U4b): shared by `finish()` (called only when
  // `request.topN > 0`, `rootWitness = tactical?.rootWitness ?? null`) and
  // `emitProgress` (`rootWitness = null`: progress is emitted only inside
  // the chunked loop, which the root-win path never enters). `n` is a
  // validated non-negative integer by the time either caller reaches here.
  private rankedRootMoves(
    tree: MatchTree,
    rootBoard: RootBoard,
    n: number,
    rootWitness: number | null,
  ): RootMoveStat[] {
    const root = tree.root;
    if (root === null) return [];
    let order = rankRootMoves(root, tree.config, tree.bias);
    if (rootWitness !== null) {
      // The witness is expected among `root.actions`; if it is not (-1),
      // `rankRootMoves`' order is kept unchanged rather than indexing -1.
      const w = root.actions.indexOf(rootWitness);
      if (w !== -1) order = [w, ...order.filter((i) => i !== w)];
    }
    const mean = childMeanWdl(root);
    const eligible = lcbEligible(root, tree.config);
    const scores = lcbScores(root, tree.config, tree.bias);
    return order.slice(0, n).map((i) => {
      const action = root.actions[i]!;
      const { struct } = decodeAction(rootBoard, action);
      return {
        action,
        san: buildSan(rootBoard, struct),
        descriptor: describeRecord(rootBoard, struct),
        visits: root.weight[i]!,
        wdlMean: [mean[i * 3]!, mean[i * 3 + 1]!, mean[i * 3 + 2]!] as [number, number, number],
        prior: root.prior[i]!,
        lcbEligible: eligible[i]!,
        lcbScore: eligible[i]! ? scores[i]! : null,
      };
    });
  }

  // KOM-1195 design section 4 (U4b): walks each node's chosen child's chosen
  // child, cloning `preMoveGame` forward the same way `predictReply`/
  // `snapshotStanding` already do. Stops at `width === 0`, depth 8, or --
  // before pushing that move's SAN -- a child the search has not actually
  // expanded (`node.children.get(idx)` undefined): a node with ANY weight at
  // all still has at least one real child, so in practice this only ever
  // truncates below the root, never at it (pv[0] is always the root's own
  // best move whenever this is called mid-search, i.e. with playoutsUsed >=
  // 1 -- see `emitProgress`).
  private buildPv(root: MatchNode, preMoveGame: Game, config: TreeConfig, bias: BiasReader): string[] {
    const pv: string[] = [];
    let node: MatchNode = root;
    let game: Game = preMoveGame;
    let cloned = false;
    for (let depth = 0; depth < 8; depth++) {
      if (node.width === 0) break;
      const idx = selectAction(node, config, { bias });
      const child = node.children.get(idx) as MatchNode | undefined;
      if (child === undefined) break;
      const action = node.actions[idx]!;
      const { struct } = decodeAction(game.board, action);
      pv.push(buildSan(game.board, struct));
      if (!cloned) {
        game = game.clone();
        cloned = true;
      }
      game.play(action);
      node = child;
    }
    return pv;
  }

  // KOM-1195 design section 4 (U4b): the real implementation behind
  // `SearchOptions.onProgress`'s pure counter -- `search/search.ts` has no
  // SAN encoder and no knowledge of `protocol.ts`, so `attemptFast`'s
  // cadence wrapper calls this with the live tree/root. `bestSan`/
  // `bestAction` are decoded exactly as `finish()` already does; `rootWdl`
  // is `buildResult`'s own formula (search/search.ts), duplicated here
  // rather than imported -- `buildResult` is an unexported function in a
  // file outside this unit's allowlist (packages/engine/src/engine.ts,
  // packages/engine/test/kom1195_topn.test.ts only).
  private emitProgress(
    id: string | number,
    tree: MatchTree,
    preMoveGame: Game,
    playoutsUsed: number,
    topN: number | undefined,
  ): void {
    if (this._onProgress === undefined) return;
    const root = tree.root;
    if (root === null) return;
    const rootBoard = preMoveGame.board;
    const bestIdx = selectAction(root, tree.config, { bias: tree.bias });
    const bestAction = root.actions[bestIdx]!;
    const { struct } = decodeAction(rootBoard, bestAction);
    const bestSan = buildSan(rootBoard, struct);
    const total = root.totalWeight();
    let rootWdl: [number, number, number];
    if (total > 0) {
      const sum = root.subtreeWdlSum();
      rootWdl = [sum[0]! / total, sum[1]! / total, sum[2]! / total];
    } else {
      rootWdl = [root.wdlNet[0]!, root.wdlNet[1]!, root.wdlNet[2]!];
    }
    const pv = this.buildPv(root, preMoveGame, tree.config, tree.bias);
    const event: WorkerProgressEvent = { id, type: 'progress', playouts: playoutsUsed, bestSan, bestAction, rootWdl, pv };
    if (topN !== undefined && topN > 0) {
      event.topMoves = this.rankedRootMoves(tree, rootBoard, topN, null);
    }
    this._onProgress(event);
  }

  // Design 5.3/6.3 M-4: stops whatever is pondering, if anything. The
  // in-flight batch (if any) is still awaited by `runBatchedSearchAbortable`
  // -- this only flips the flag it checks before the next `gen.next()`.
  private abortPonder(): void {
    if (this._ponderSignal !== null) this._ponderSignal.aborted = true;
    this._ponderSignal = null;
  }

  // KOM-1102 item 2: the standing-tree snapshot taken on every
  // `maybeStartPonder` call -- `ctx.preMoveGame`/`chosenAction` replayed the
  // same way `runPonderSession`'s own first two lines do, to get the SAME
  // resulting SAN/fen a continuing request would carry.
  private snapshotStanding(ctx: PonderStartCtx): StandingPonder {
    const game = ctx.preMoveGame.clone();
    const san = buildSan(game.board, decodeAction(game.board, ctx.chosenAction).struct);
    game.play(ctx.chosenAction);
    return {
      ctx,
      tree: ctx.tree,
      evaluationsAtReply: ctx.tree.evaluations,
      fen: boardToFen(game.board),
      history: [...ctx.historyBase, san],
    };
  }

  // Fast-mode U7 (design 5.1, 5.4): starts one ponder session predicting up
  // to `K` replies to `ctx.chosenAction`, each a full speculative search on
  // its own scratch tree. Synchronous up through reading the prediction and
  // snapshotting the bias table (5.2: "taken when pondering starts") --
  // everything that needs the just-finished search's root, which the next
  // search's `tree.clear()` would otherwise be free to discard first.
  private maybeStartPonder(ctx: PonderStartCtx): void {
    this._lastStanding = this.snapshotStanding(ctx);
    if (!shouldPonder(this._ponderSettings, this._ponderGating)) return;
    const root = ctx.tree.root;
    if (root === null) return;
    const predictions = predictReplies(root, ctx.chosenAction, this._ponderSettings.k);
    if (predictions.length === 0) return;

    const signal: PonderSignal = { aborted: false };
    this._ponderSignal = signal;
    const biasSnapshot = snapshotBias(ctx.tree);
    const mainTableEpoch = ctx.tree.evaluations;
    const configIdentity = configIdentityString(ctx.discovery, ctx.draftmin, ctx.mlhConfig);

    this._ponderTask = this.runPonderSession(ctx, predictions, signal, biasSnapshot, mainTableEpoch, configIdentity)
      // A ponder session never surfaces anywhere a real request can see it
      // (design 5.3: an aborted/failed speculative search is just
      // discarded) -- this `catch` is what keeps it from becoming an
      // unhandled rejection, not a reply path.
      .catch(() => {});
  }

  // Runs the K predicted-reply searches in sequence (design 5.4: "stops
  // after K replies" -- one scratch tree/bias copy each, design 5.2), over
  // the C4 abortable driver. Caches each one that finishes before `signal`
  // aborts; a thrown `PonderAborted` (or any other error) ends the whole
  // session right there -- design 5.3's "an aborted generator is dropped,
  // nothing else touched" extends to the rest of the K loop too.
  private async runPonderSession(
    ctx: PonderStartCtx,
    predictions: number[],
    signal: PonderSignal,
    biasSnapshot: Map<number, BiasEntry>,
    mainTableEpoch: number,
    configIdentity: string,
  ): Promise<void> {
    const chosenGame = ctx.preMoveGame.clone();
    const chosenSan = buildSan(chosenGame.board, decodeAction(chosenGame.board, ctx.chosenAction).struct);
    chosenGame.play(ctx.chosenAction);

    for (const replyAction of predictions) {
      if (signal.aborted) return;
      const game = chosenGame.clone();
      const replySan = buildSan(game.board, decodeAction(game.board, replyAction).struct);
      game.play(replyAction);

      const nextHistory = [...ctx.historyBase, chosenSan, replySan];
      const nextFen = boardToFen(game.board);
      const state = new MatchGameState(game, {
        minPlacements: ctx.draftmin,
        isRoot: true,
        planeMapVersion: ctx.planeMapVersion,
      });

      const mlhCtx = new MlhContext();
      const wrappedBatch = wrapMlhBatch(ctx.batchEvaluator, ctx.mlhConfig, mlhCtx);
      const rootEvaluator: MaybeAsyncWdlEvaluator = (s: MatchState) =>
        Promise.resolve(wrappedBatch([s])).then((rows) => rows[0]!);
      const scratchConfig: SearchConfig = { ...ctx.tree.config, playouts: ctx.sims };
      const scratchTree = new MatchTree(scratchConfig, rootEvaluator, wrappedBatch);
      // 5.2: "a copy of the main table taken when pondering starts" -- the
      // SAME snapshot for every one of the K searches, each restored into
      // its own tree so they can never see each other's backups.
      restoreBias(scratchTree, biasSnapshot);

      let shouldStop: (() => boolean) | undefined;
      if (ctx.maxMs !== null) {
        const deadlineStart = this.clock();
        const deadline = ctx.maxMs;
        shouldStop = () => this.clock() - deadlineStart >= deadline;
      } else if (ctx.chunked) {
        // KOM-1195 design section 6 (re-review N2): the no-deadline form of
        // section 3's composed `shouldStop` -- never fires, but its mere
        // presence sends an opted-in history play down the SAME chunked
        // loop its own live search would take, at batch 1 too.
        shouldStop = () => false;
      }
      const searchOpts: SearchOptions = {
        discovery: ctx.discovery,
        ply: game.turns,
        tacticalTransitionCap: DEFAULT_TRANSITION_CAP,
      };
      if (shouldStop !== undefined) searchOpts.shouldStop = shouldStop;

      const result = await runPonderSearch(
        scratchTree,
        state,
        { ...scratchConfig, playouts: ctx.sims },
        searchOpts,
        ctx.batch,
        signal,
      );

      const key = ponderKey({
        history: nextHistory,
        fen: nextFen,
        start: ctx.start,
        rules: rulesConfigKey(ctx.rules) + hookKeyComponent(this._scoreTerminal),
        planeMapVersion: ctx.planeMapVersion,
        sims: ctx.sims,
        batch: ctx.batch,
        configIdentity,
        backend: this._backend,
        modelSha256: this.modelSha256,
        maxMs: ctx.maxMs,
        mainTableEpoch,
        chunked: ctx.chunked,
      });
      this._ponderCache.set(key, {
        result,
        tree: scratchTree,
        context: {
          game,
          historyBase: nextHistory,
          start: ctx.start,
          rules: ctx.rules,
          planeMapVersion: ctx.planeMapVersion,
          sims: ctx.sims,
          batch: ctx.batch,
          draftmin: ctx.draftmin,
          discovery: ctx.discovery,
          mlhConfig: ctx.mlhConfig,
          maxMs: ctx.maxMs,
        },
      });
    }
  }

  // C2: checks the ponder cache for an entry whose key matches this
  // request's own (history, fen, sims, batch, config identity, backend,
  // modelSha256, maxMs, CURRENT mainTableEpoch) -- folding the live epoch
  // into the comparison is what makes "any main-tree search between ponder
  // start and lookup" a built-in miss (ponder.ts's own `ponderKey` doc).
  // `null` when the request carries no history array at all (nothing was
  // ever cached under such a key, design 5.1) or on any miss.
  private lookupPonder(request: WorkerRequest, batchEvaluator: MlhBatchEvaluator, target: StopTarget): WorkerReply | null {
    if (!Array.isArray(request.history)) return null;
    const { discovery, draftmin, mlhConfig } = this.effectiveIdentity(request);
    // Already validated by handleFast before this call; re-resolved here
    // only to get the effective value for the key (toRulesConfig is pure
    // and cheap, so re-running it is not a second validation pass in spirit).
    const config = toRulesConfig(request.rules);
    const planeMapVersion = parsePlaneMapVersion(request.planeMapVersion);
    const batch = request.batch ?? 1;
    const maxMs = request.maxMs ?? null;
    const configIdentity = configIdentityString(discovery, draftmin, mlhConfig);
    const mainTableEpoch = this._tree === null ? 0 : this._tree.evaluations;
    const historyForKey = historyKey(request.history, request.start, config);
    if (historyForKey === null) return null;
    const key = ponderKey({
      history: historyForKey,
      fen: request.fen,
      start: request.start ?? ADVANCED_POSITION,
      rules: rulesConfigKey(config) + hookKeyComponent(this._scoreTerminal),
      planeMapVersion,
      sims: request.sims,
      batch,
      configIdentity,
      backend: this._backend,
      modelSha256: this.modelSha256,
      maxMs,
      mainTableEpoch,
      chunked: target.stoppable,
    });
    const entry = this._ponderCache.get(key);
    if (entry === null) return null;

    // F8: the main tree ADOPTS the pondered one -- the next prediction (the
    // `maybeStartPonder` call below) reads this tree's root directly, and a
    // later real search's `tree.clear()`/bias table are this tree's own.
    this._tree = entry.tree;
    const prepared: PreparedSearch = {
      tree: entry.tree,
      state: new MatchGameState(entry.context.game, {
        minPlacements: entry.context.draftmin,
        isRoot: true,
        planeMapVersion: entry.context.planeMapVersion,
      }),
      searchConfig: { ...entry.tree.config, playouts: entry.context.sims },
      searchOpts: {
        discovery: entry.context.discovery,
        ply: entry.context.game.turns,
        tacticalTransitionCap: DEFAULT_TRANSITION_CAP,
      },
      rootBoard: entry.context.game.board,
      stamp: HISTORY_PRESENT,
      sims: entry.context.sims,
      draftmin: entry.context.draftmin,
      discovery: entry.context.discovery,
      mlhConfig: entry.context.mlhConfig,
      started: Date.now(),
      timing: null,
      config: entry.context.rules,
    };
    const predictedReply = this.predictReply(entry.tree, entry.context.game, entry.result.action);
    const reply = this.finish(request, prepared, entry.result, null, null, predictedReply);
    if (!('err' in reply)) {
      // N-6: a cached reply reports its own fresh `ms` (already true via
      // `finish`'s own `Date.now() - prepared.started`) and adds
      // `pondered: true`.
      reply.stats.pondered = true;
      this.maybeStartPonder({
        tree: entry.tree,
        batchEvaluator,
        preMoveGame: entry.context.game,
        chosenAction: entry.result.action,
        historyBase: entry.context.historyBase,
        start: entry.context.start,
        rules: entry.context.rules,
        planeMapVersion: entry.context.planeMapVersion,
        sims: entry.context.sims,
        batch: entry.context.batch,
        draftmin: entry.context.draftmin,
        discovery: entry.context.discovery,
        mlhConfig: entry.context.mlhConfig,
        maxMs: entry.context.maxMs,
        chunked: target.stoppable,
      });
    }
    return reply;
  }

  // M-6/F7: the page's ponder settings/gating message. Every field is a
  // partial update; disabling pondering, or gating it off mid-session,
  // aborts whatever is currently in flight immediately (design 5.4).
  handlePonderSettings(request: WorkerPonderSettingsRequest): WorkerPonderSettingsReply {
    if (request.enabled !== undefined) this._ponderSettings.enabled = request.enabled;
    if (request.k !== undefined) this._ponderSettings.k = clampPonderK(request.k);
    if (request.hidden !== undefined) this._ponderGating.hidden = request.hidden;
    if (request.discharging !== undefined) this._ponderGating.discharging = request.discharging;
    if (!shouldPonder(this._ponderSettings, this._ponderGating)) this.abortPonder();
    return {
      id: request.id,
      enabled: this._ponderSettings.enabled,
      k: this._ponderSettings.k,
      hidden: this._ponderGating.hidden,
      discharging: this._ponderGating.discharging,
    };
  }

  // KOM-1102 item 2 (design 5.1's ponder-start path, reused rather than
  // duplicated): resumes pondering from the standing tree the engine's last
  // reply left behind, for when the page's gating flags clear after that
  // reply instead of before it. Valid only when a prior reply actually
  // built a standing snapshot, nothing has searched the main tree since (the
  // SAME tree instance at the SAME evaluation count `maybeStartPonder`
  // observed -- any intervening real search, or a ponder hit that adopted a
  // different tree, fails this), the page's own fen/history still match what
  // that reply produced, pondering is enabled and ungated, and nothing is
  // already pondering (two ponder-now calls must not both start a session).
  // On every valid path this calls `maybeStartPonder` with the EXACT ctx the
  // original reply would have used -- same prediction, same cache key, same
  // bias snapshot -- never a parallel implementation of the ponder-start
  // logic. `started` reflects whether a session actually began (a valid
  // resume can still find nothing to predict, same as a reply-triggered
  // start would).
  handlePonderNow(request: WorkerPonderNowRequest): WorkerPonderNowReply {
    const standing = this._lastStanding;
    if (standing === null) return { id: request.id, started: false, reason: 'no_prior_reply' };
    if (standing.tree !== this._tree || standing.tree.evaluations !== standing.evaluationsAtReply) {
      return { id: request.id, started: false, reason: 'real_search_since' };
    }
    const history = historyKey(request.history ?? [], standing.ctx.start, standing.ctx.rules);
    if (
      history === null ||
      request.fen !== standing.fen ||
      JSON.stringify(history) !== JSON.stringify(standing.history)
    ) {
      return { id: request.id, started: false, reason: 'history_changed' };
    }
    if (!shouldPonder(this._ponderSettings, this._ponderGating)) {
      return { id: request.id, started: false, reason: 'gated' };
    }
    if (this._ponderSignal !== null) {
      return { id: request.id, started: false, reason: 'already_pondering' };
    }
    this.maybeStartPonder(standing.ctx);
    if (this._ponderSignal === null) return { id: request.id, started: false, reason: 'no_prediction' };
    return { id: request.id, started: true };
  }

  // Test/diagnostic only: resolves once the current ponder session (if any)
  // has settled -- never sleeps/polls, just awaits the one promise
  // `maybeStartPonder` already tracks. Never rejects (`runPonderSession`'s
  // own `.catch`).
  async ponderingPromise(): Promise<void> {
    const task = this._ponderTask;
    if (task !== null) await task;
  }

  // Test/diagnostic only.
  get ponderCacheSize(): number {
    return this._ponderCache.size;
  }

  // Test/diagnostic only (KOM-1195 design section 3 AC2(d)): every
  // `StopTarget` released on every path, so this returns to 0 once no fast
  // search is in flight.
  get stopTargetsSize(): number {
    return this._stopTargets.size;
  }

  // Everything `answer()` used to do after `search()` returned: the
  // evaluation-ceiling check, SAN, and the reply shape. Unchanged signature
  // and output (KOM-1219 unit 1, D3) -- the body now lives in `buildReply`,
  // shared with `finishForced`.
  private finish(
    request: WorkerRequest,
    prepared: PreparedSearch,
    result: MatchResult,
    fallback: string | null = null,
    probeRecord: BackendRecord | null = null,
    predictedReply: PredictedReply | null = null,
    // KOM-1195 design section 3 (review F12): present only through the fast
    // path's live-search call sites (`attemptFast`); `null` through
    // `handle()`/`handleAsync()` (no stop target at all) and a ponder-cache
    // hit (no live search this call). `opted` is `target.stoppable`.
    stopInfo: { opted: boolean; reason: StopReason | null } | null = null,
  ): WorkerReply {
    return this.buildReply(request, prepared, result, { nodes: prepared.tree.nodeCount }, fallback, probeRecord, predictedReply, stopInfo);
  }

  // KOM-1219 design section 2.2: `finish()`'s own reply construction,
  // factored out so `finishForced` never reads `prepared.tree.nodeCount`
  // (the PREVIOUS search's count -- `tryForced` never calls `prepareFast`,
  // so `this._tree` is untouched). `built` is `RootBuild` (from `buildRoot`,
  // `finishForced`'s own case -- no precomputed `timing`, derived here from
  // `clock`/`t0` the same way `prepare()`/`prepareFast()` already do) or
  // `PreparedSearch`/`PreparedFastSearch` (`finish()`'s case -- `timing`
  // already computed at prepare time, read as-is). `extra.nodes` is passed
  // explicitly rather than read off a shared tree; `extra.forced` sets
  // `stats.forced` only when given.
  private buildReply(
    request: WorkerRequest,
    built: RootBuild | PreparedSearch,
    result: MatchResult,
    extra: { nodes: number; forced?: true },
    fallback: string | null = null,
    probeRecord: BackendRecord | null = null,
    predictedReply: PredictedReply | null = null,
    stopInfo: { opted: boolean; reason: StopReason | null } | null = null,
  ): WorkerReply {
    const timing =
      'timing' in built
        ? built.timing
        : built.clock === null
          ? null
          : { clock: built.clock, t0: built.t0, prepareMs: built.clock() - built.t0 };
    const finishStart = timing === null ? 0 : timing.clock();
    const ceiling = built.sims + CEILING_SLACK;
    if (result.evaluationsUsed > ceiling) {
      throw new Error(
        `evaluation ceiling exceeded: ${result.evaluationsUsed} evaluations for a ${ceiling} ceiling`,
      );
    }

    const { struct } = decodeAction(built.rootBoard, result.action);
    const san = buildSan(built.rootBoard, struct);

    const stats: WorkerStats = {
      playouts: result.playoutsUsed,
      evaluations: result.evaluationsUsed,
      nodes: extra.nodes,
      rootWdl: Array.from(result.rootWdl),
      q: result.rootUtility,
      tactical: result.tactical,
      ms: Date.now() - built.started,
      model_sha256: this.modelSha256,
      sims: built.sims,
      draftmin: built.draftmin,
      discovery: discoveryLabel(built.discovery),
      mlh: formatMlhStamp(built.mlhConfig),
      // Design 1.7 C3: true only when a `shouldStop` deadline fired before
      // `config.playouts` was reached (searchFast/searchBatchedGen); always
      // false through search()/searchAsync(), which never see one.
      deadlineHit: result.deadlineHit,
    };
    if (extra.forced === true) stats.forced = true;
    if (this._backend !== null) stats.backend = this._backend;
    if (fallback !== null) stats.fallback = fallback;
    if (probeRecord !== null) stats.probeRecord = probeRecord;
    // KOM-1195 design section 3 (review F12): never set on an un-opted
    // request (AC1), and never set when the search ran to its playout cap
    // other than an analysis reaching its ceiling -- `stopState.reason`
    // stays `null` in that ordinary-completion case (U4a never seeds it for
    // a non-analysis target, and no contributor fires at the cap). Never
    // set for a forced reply either: `tryForced` never passes `stopInfo`.
    if (
      stopInfo !== null &&
      stopInfo.opted &&
      stopInfo.reason !== null &&
      (result.deadlineHit || (result.playoutsUsed > 0 && result.playoutsUsed >= built.sims))
    ) {
      stats.stopReason = stopInfo.reason;
    }
    // KOM-1105/KOM-1203 F2(b): echoed only when non-default (see `echoRules`
    // above), so a default-path reply stays byte-identical to before `rules`
    // existed.
    const echoedRules = echoRules(request.rules, built.config);
    if (echoedRules !== undefined) stats.rules = echoedRules;
    if (timing !== null && result.phases !== null) {
      const end = timing.clock();
      stats.phases = {
        prepareMs: timing.prepareMs,
        ...result.phases,
        finishMs: end - finishStart,
        totalMs: end - timing.t0,
      };
    }

    const reply: WorkerReplyOk = {
      id: request.id,
      san,
      action: result.action,
      stamp: built.stamp,
      stats,
    } as WorkerReplyOk;
    if (predictedReply !== null) reply.predictedReply = predictedReply;
    // KOM-1195 design section 2/5 (U4b): present only when `request.topN >
    // 0` (never assigned otherwise, same optional-assignment pattern as
    // `predictedReply`), and assigned BEFORE `descriptor`, which stays the
    // reply's last key (KOM-1198, N8). `finishForced` never sets a positive
    // `topN` request down this path (D2), so `'tree' in built` is really
    // only ever false there as a type guard, never an observed skip.
    if (request.topN !== undefined && request.topN > 0 && 'tree' in built) {
      reply.topMoves = this.rankedRootMoves(
        built.tree,
        built.rootBoard,
        request.topN,
        result.tactical?.rootWitness ?? null,
      );
    }
    reply.descriptor = describeRecord(built.rootBoard, struct);
    return reply;
  }

  // KOM-1219 design section 2.2: the forced-move reply's stats (AC1) --
  // `nodes: 0`, zero playouts/evaluations, and never a read of `this._tree`.
  private finishForced(request: WorkerRequest, built: RootBuild, action: number): WorkerReply {
    const forcedResult: MatchResult = {
      action,
      evaluationsUsed: 0,
      playoutsUsed: 0,
      nodesAtEntry: 0,
      rootWdl: new Float64Array([0, 0, 0]),
      rootUtility: 0,
      actions: Int32Array.from([action]),
      visits: new Int32Array([0]),
      weights: new Float64Array([0]),
      lcbScores: new Float64Array([0]),
      tactical: null,
      phases: built.clock === null ? null : ZERO_PHASES,
      deadlineHit: false,
    };
    return this.buildReply(request, built, forcedResult, { nodes: 0, forced: true });
  }
}

