// KOM-1199 unit U3a (design docs/design/kom1199-client.md sections 2, 3, 5):
// the main-thread `KomugiClient`. Owns at most one host-created worker,
// speaks protocol.ts's init/search messages to `installWorkerHost`, and
// resolves every `play` with a move, a typed failure or "stopped" -- never
// another engine's move (no fallback hook of any kind, section 1).
//
// Only `./client_failures.ts` is a value import: everything else is
// `import type`, so the main-thread bundle pulls in no search, rules or ORT
// code. Importing this module and constructing a client start nothing;
// `createWorker` is first called by `init()`/`play()`.

import type { BackendKind, BackendRecord } from './backend.ts';
import type {
  RequestConfig,
  WorkerAnalyzeRequest,
  WorkerInitReplyErr,
  WorkerInitRequest,
  WorkerPonderSettingsReplyOk,
  WorkerPonderSettingsRequest,
  WorkerProgressEvent,
  WorkerReplyErr,
  WorkerReplyOk,
  WorkerRequest,
  WorkerStopRequest,
} from './protocol.ts';
import type { RulesProfile } from './repetition.ts';
import type { RulesConfig } from './rules_config.ts';
import type { PlaneMapVersion } from './rule_features.ts';
import type { KomugiFailure } from './client_failures.ts';
import { classifyErrorReply, validateInitReply, validateSearchReply } from './client_failures.ts';

// constants.ts's ACTION_SIZE, re-spelled so this module keeps zero value
// imports outside client_failures.ts (test/client.test.ts pins the two equal).
const ACTION_SIZE = 37341;

const DEFAULT_INIT_TIMEOUT_MS = 60_000;
const DEFAULT_MOVE_TIMEOUT_GRACE_MS = 5_000;
const DEFAULT_MOVE_TIMEOUT_MS = 60_000;
const DEFAULT_MAX_CONSECUTIVE_CRASHES = 3;
const DEFAULT_STOP_GRACE_MS = 5_000;

// The minimum of a DOM `Worker` the client touches. A browser `Worker`
// satisfies it structurally; Node tests wrap `worker_threads.Worker`
// (section 6). 'error' and 'messageerror' are both treated as a crash.
export interface KomugiWorkerLike {
  postMessage(message: unknown): void;
  terminate(): void;
  addEventListener(type: 'message', listener: (event: { data: unknown }) => void): void;
  addEventListener(type: 'error' | 'messageerror', listener: (event: unknown) => void): void;
}

export type KomugiWorkerFactory = () => KomugiWorkerLike;

export interface KomugiModel {
  url: string;
  sha256: string; // lowercase or uppercase hex; the worker lowercases
  // KOM-1188 (model_registry.ts): set by resolveModelCard. Never sent at
  // init; every search is bound to this planeMapVersion.
  compatibility?: { planeMapVersion: PlaneMapVersion };
  // KOM-1200 section 3: owned by the host's loadModel() result, called
  // exactly once from dispose() (never stop()/retry()/a crash restart),
  // isolated through callHost. Never sent to the worker.
  release?: () => void;
}

export interface KomugiOrtAssets {
  wasm: { mjs: string; wasm: string };
  webgpu?: { mjs: string; wasm: string };
}

// KOM-1226 item 4 (D1): a partial update to the worker's pondering
// settings -- absent fields are left unchanged, mirroring
// WorkerPonderSettingsRequest's own partial-update fields.
export interface KomugiPonderOptions {
  enabled?: boolean;
  k?: number;
}

// KOM-1226 item 4 (D1): the full settings+gating state a ponderSettings
// reply carries back (k as the worker clamped it).
export interface KomugiPonderState {
  enabled: boolean;
  k: number;
  hidden: boolean;
  discharging: boolean;
}

// setPonder()'s own update shape (D3): KomugiPonderOptions plus the two
// page-relayed gating flags, same partial-update rule.
type PonderUpdate = KomugiPonderOptions & { hidden?: boolean; discharging?: boolean };

export interface KomugiClientOptions {
  createWorker: KomugiWorkerFactory;
  model: KomugiModel;
  ortAssets: KomugiOrtAssets;
  backend?: BackendKind; // forwarded as WorkerInitRequest.backend
  threads?: number; // forwarded as WorkerInitRequest.threads
  cachedRecord?: BackendRecord | null;
  // KOM-1226: forwarded as WorkerInitRequest.ortLogSeverityLevel. Absent
  // keeps the worker's own default (3, errors only).
  ortLogSeverityLevel?: number;
  onProbeRecord?: (record: BackendRecord) => void; // host persists it
  initTimeoutMs?: number; // default 60_000
  moveTimeoutGraceMs?: number; // default 5_000, added to maxMs
  moveTimeoutMs?: number; // default 60_000, used when maxMs absent
  maxConsecutiveCrashes?: number; // default 3, then the crash is latched (U3b)
  // KOM-1195 design section 7: grace period after moveNow() posts `stop`
  // before the client gives up and terminates (U7).
  stopGraceMs?: number; // default 5_000
  onProgress?: (event: WorkerProgressEvent) => void;
  // KOM-1226 item 4 (D2): the initial stored ponder settings. Absent posts
  // no ponderSettings message at all -- the worker's own default (on, k 1)
  // stays in force and existing message sequences are byte-identical.
  ponder?: KomugiPonderOptions;
}

// KOM-1228: the thinking budget a play() request carries -- sims with an
// optional time cap (today's form), or time only. "Both" is the first form
// with maxMs set; `{ }` (neither) is a type error, matching play()'s own
// runtime rejection below.
export type ThinkCap = { sims: number; maxMs?: number } | { sims?: undefined; maxMs: number };

// KOM-1228 (docs/design/kom1195-analysis-api.md's ANALYSIS_DEFAULT_NODE_CEILING):
// the memory bound for a time-only search, resolved client-side to
// `sims: TIME_ONLY_NODE_CEILING - 1` -- the real deadline (`maxMs`) is what
// actually stops the search. Each playout adds at most one node, about
// 3,000 B each, so this is the same value and "playouts = ceiling - 1" rule
// as the analysis design's default node ceiling.
export const TIME_ONLY_NODE_CEILING = 20_000;

interface PlayRequestFields {
  fen: string;
  history?: string[] | null;
  start?: string;
  // KOM-1226: widened to match protocol.ts's WorkerRequest.rules -- the
  // worker already accepts a RulesConfig object (KOM-1203); postSearch
  // passes this field through unchanged, so the client type must too.
  rules?: RulesProfile | RulesConfig;
  planeMapVersion?: PlaneMapVersion;
  batch?: number;
  threads?: number;
  config?: RequestConfig;
  // KOM-1195 design section 7: opts the search in to `moveNow()`/progress;
  // forwarded to the wire with no other new field (`isOptedIn`, section 2).
  stoppable?: true;
  topN?: number;
  progressEveryPlayouts?: number;
  progressEveryMs?: number;
}

export type PlayRequest = PlayRequestFields & ThinkCap;

// KOM-1195 design section 7: PlayRequest minus sims/maxMs/stoppable (an
// analysis always opts in and has no sims to bound it), plus nodeCeiling.
export type AnalyzeRequest = Omit<PlayRequest, 'sims' | 'maxMs' | 'stoppable'> & {
  nodeCeiling?: number;
};

export interface KomugiReady {
  backend: BackendKind;
  threads: number;
  batch: number;
  fallback?: { from: BackendKind; reason: string };
}

export type InitResult = { status: 'ready'; info: KomugiReady } | { status: 'failed'; failure: KomugiFailure };

export type PlayResult = { status: 'move'; reply: WorkerReplyOk } | { status: 'failed'; failure: KomugiFailure } | { status: 'stopped' };

export type KomugiClientState = 'idle' | 'starting' | 'ready' | 'searching' | 'failed' | 'disposed';

type Timer = ReturnType<typeof globalThis.setTimeout>;

interface PendingInit {
  id: string;
  settled: boolean;
  timer: Timer | null;
  // KOM-1214: true once this init already restarted after a threads-timeout.
  restarted: boolean;
  resolve: (result: InitResult) => void;
}

interface PendingPlay {
  // null until the search message is posted (the play is awaiting init).
  id: string | null;
  settled: boolean;
  timer: Timer | null;
  // KOM-1195 design section 7 (N9): armed when the `stop` message is
  // actually posted, cleared by settlePlay on every settle path.
  graceTimer: Timer | null;
  // Whether this play/analysis opted in to `moveNow()` (isOptedIn, section 2).
  stoppable: boolean;
  // Set by moveNow() called before the search message is posted (still
  // initialising); postSearch/postAnalyze sends the stop right after.
  stopRequested: boolean;
  resolve: (result: PlayResult) => void;
}

// KOM-1195 design section 2's isOptedIn, mirrored client-side: stoppable:
// true, a progressEvery* field, or topN > 0 (topN: 0 is absent) opts in.
function isOptedInPlay(request: PlayRequest): boolean {
  return (
    request.stoppable === true ||
    (request.topN !== undefined && request.topN > 0) ||
    request.progressEveryPlayouts !== undefined ||
    request.progressEveryMs !== undefined
  );
}

function isErrReply(data: object): data is WorkerReplyErr {
  return typeof (data as { err?: unknown }).err === 'string';
}

function isObject(value: unknown): value is object {
  return typeof value === 'object' && value !== null;
}

// A throwing host callback must not wedge the client's state machine: the
// error is re-thrown asynchronously instead, so it still surfaces.
function callHost(callback: () => void): void {
  try {
    callback();
  } catch (error) {
    globalThis.queueMicrotask(() => {
      throw error;
    });
  }
}

export class KomugiClient {
  private readonly options: KomugiClientOptions;
  private worker: KomugiWorkerLike | null = null;
  // Bumped on every spawn and terminate; listeners of an older generation
  // ignore their events.
  private generation = 0;
  private requestCounter = 0;
  private readyInfo: KomugiReady | null = null;
  private initPromise: Promise<InitResult> | null = null;
  private pendingInit: PendingInit | null = null;
  private pendingPlay: PendingPlay | null = null;
  private latched: KomugiFailure | null = null;
  private last: KomugiFailure | null = null;
  private probeRecord: BackendRecord | null = null;
  // KOM-1214: set to 1 after a threads-timeout and kept for every later
  // worker of this client (a device that timed out once stays capped).
  private maxThreads: number | null = null;
  private disposed = false;
  // Search-stage worker-crash/timeout/illegal-reply outcomes, reset on any
  // 'move'; latched failed at maxConsecutiveCrashes (section 3).
  private consecutiveCrashes = 0;
  // KOM-1226 item 4: every field ever set (constructor `ponder` option or
  // setPonder()), merged -- re-posted whole to every worker this client
  // spawns (D4). Empty means "nothing stored" (D2: post nothing, ever).
  private ponderStored: PonderUpdate;
  private readonly pendingPonder = new Map<string, { resolve: (state: KomugiPonderState | null) => void; reject: (error: Error) => void }>();

  constructor(options: KomugiClientOptions) {
    this.options = options;
    this.ponderStored = options.ponder !== undefined ? { ...options.ponder } : {};
  }

  get state(): KomugiClientState {
    if (this.disposed) return 'disposed';
    if (this.latched !== null) return 'failed';
    if (this.pendingPlay !== null && this.pendingPlay.id !== null) return 'searching';
    if (this.worker !== null) return this.readyInfo !== null ? 'ready' : 'starting';
    return this.pendingPlay !== null ? 'starting' : 'idle';
  }

  get lastFailure(): KomugiFailure | null {
    return this.last;
  }

  init(): Promise<InitResult> {
    if (this.disposed) return Promise.reject(new Error('KomugiClient: init() after dispose()'));
    if (this.latched !== null) return Promise.resolve({ status: 'failed', failure: this.latched });
    if (this.initPromise !== null) return this.initPromise;
    const init: PendingInit = { id: '', settled: false, timer: null, restarted: false, resolve: () => undefined };
    const promise = new Promise<InitResult>((resolve) => {
      init.resolve = resolve;
    });
    this.pendingInit = init;
    this.initPromise = promise;
    this.spawnAndInit();
    return promise;
  }

  play(request: PlayRequest): Promise<PlayResult> {
    if (this.disposed) return Promise.reject(new Error('KomugiClient: play() after dispose()'));
    if (this.pendingPlay !== null) return Promise.reject(new Error('KomugiClient: play() while a play is in flight'));
    if (typeof request.fen !== 'string' || request.fen.length === 0) {
      return Promise.reject(new Error('KomugiClient: play() needs a non-empty fen'));
    }
    if (request.sims === undefined && request.maxMs === undefined) {
      return Promise.reject(new Error('KomugiClient: play() needs sims, maxMs, or both'));
    }
    if (request.sims !== undefined) {
      if (!Number.isInteger(request.sims) || request.sims <= 0) {
        return Promise.reject(new Error('KomugiClient: play() needs sims to be a positive integer'));
      }
    } else if (!Number.isFinite(request.maxMs) || request.maxMs <= 0) {
      return Promise.reject(new Error('KomugiClient: play() needs maxMs to be a finite positive number'));
    }
    const play: PendingPlay = {
      id: null,
      settled: false,
      timer: null,
      graceTimer: null,
      stoppable: isOptedInPlay(request),
      stopRequested: false,
      resolve: () => undefined,
    };
    const promise = new Promise<PlayResult>((resolve) => {
      play.resolve = resolve;
    });
    this.pendingPlay = play;
    void this.init().then((result) => {
      if (play.settled) return; // stopped/disposed while starting
      if (result.status === 'failed') {
        this.settlePlay(play, { status: 'failed', failure: result.failure });
        return;
      }
      this.postSearch(play, request);
    });
    return promise;
  }

  // KOM-1195 design section 7: asks the pending stoppable search for its
  // best move so far. Returns false (and does nothing) when no stoppable
  // play/analysis is pending.
  moveNow(): boolean {
    const play = this.pendingPlay;
    if (play === null || !play.stoppable) return false;
    if (play.id === null) {
      // Still initialising: postSearch/postAnalyze sends the stop right
      // after the search message, once there is a target to name.
      play.stopRequested = true;
      return true;
    }
    this.sendStop(play);
    return true;
  }

  // KOM-1195 design section 7: runs an analysis until moveNow(), the node
  // ceiling, or stop() (abandon). No 60 s move timer (an analysis is
  // expected to run until told otherwise); only moveNow()'s grace timer
  // can fail it.
  analyze(request: AnalyzeRequest): Promise<PlayResult> {
    if (this.disposed) return Promise.reject(new Error('KomugiClient: analyze() after dispose()'));
    if (this.pendingPlay !== null) return Promise.reject(new Error('KomugiClient: analyze() while a play is in flight'));
    if (typeof request.fen !== 'string' || request.fen.length === 0) {
      return Promise.reject(new Error('KomugiClient: analyze() needs a non-empty fen'));
    }
    const play: PendingPlay = {
      id: null,
      settled: false,
      timer: null,
      graceTimer: null,
      stoppable: true,
      stopRequested: false,
      resolve: () => undefined,
    };
    const promise = new Promise<PlayResult>((resolve) => {
      play.resolve = resolve;
    });
    this.pendingPlay = play;
    void this.init().then((result) => {
      if (play.settled) return; // stopped/disposed while starting
      if (result.status === 'failed') {
        this.settlePlay(play, { status: 'failed', failure: result.failure });
        return;
      }
      this.postAnalyze(play, request);
    });
    return promise;
  }

  // KOM-1226 item 4 (D3): validates every present field first (nothing
  // stored or posted on failure), then merges the valid fields into the
  // stored settings. Posts ONE ponderSettings message and resolves with
  // the reply when a worker is ready (state 'ready'/'searching');
  // otherwise resolves null once the merge is stored.
  setPonder(update: PonderUpdate): Promise<KomugiPonderState | null> {
    if (this.disposed) return Promise.reject(new Error('KomugiClient: setPonder() after dispose()'));
    if (update.enabled !== undefined && typeof update.enabled !== 'boolean') {
      return Promise.reject(new Error('KomugiClient: setPonder() needs enabled to be a boolean'));
    }
    if (update.k !== undefined && !Number.isFinite(update.k)) {
      return Promise.reject(new Error('KomugiClient: setPonder() needs k to be a finite number'));
    }
    if (update.hidden !== undefined && typeof update.hidden !== 'boolean') {
      return Promise.reject(new Error('KomugiClient: setPonder() needs hidden to be a boolean'));
    }
    if (update.discharging !== undefined && typeof update.discharging !== 'boolean') {
      return Promise.reject(new Error('KomugiClient: setPonder() needs discharging to be a boolean'));
    }

    if (update.enabled !== undefined) this.ponderStored.enabled = update.enabled;
    if (update.k !== undefined) this.ponderStored.k = update.k;
    if (update.hidden !== undefined) this.ponderStored.hidden = update.hidden;
    if (update.discharging !== undefined) this.ponderStored.discharging = update.discharging;

    const worker = this.worker;
    const state = this.state;
    if (worker === null || (state !== 'ready' && state !== 'searching')) return Promise.resolve(null);
    return this.postPonderSettings(worker);
  }

  stop(): void {
    if (this.disposed) throw new Error('KomugiClient: stop() after dispose()');
    const play = this.pendingPlay;
    const init = this.pendingInit;
    if (play === null && (init === null || init.settled)) return; // nothing pending: keep the worker
    if (play !== null) this.settlePlay(play, { status: 'stopped' });
    if (init !== null) this.settleInit(init, { status: 'failed', failure: { code: 'worker-crash', stage: 'init', message: 'stopped' } });
    this.toIdle();
  }

  retry(): void {
    if (this.disposed) throw new Error('KomugiClient: retry() after dispose()');
    const play = this.pendingPlay;
    const init = this.pendingInit;
    if (play !== null) this.settlePlay(play, { status: 'stopped' });
    if (init !== null) this.settleInit(init, { status: 'failed', failure: { code: 'worker-crash', stage: 'init', message: 'stopped' } });
    this.toIdle();
    this.latched = null;
    this.last = null;
    this.consecutiveCrashes = 0;
  }

  // Idempotent: a second dispose() is a no-op rather than a rejection, so
  // hosts can call it unconditionally from cleanup paths.
  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    const play = this.pendingPlay;
    const init = this.pendingInit;
    if (play !== null) this.settlePlay(play, { status: 'stopped' });
    if (init !== null) this.settleInit(init, { status: 'failed', failure: { code: 'worker-crash', stage: 'init', message: 'disposed' } });
    this.toIdle();
    const release = this.options.model.release;
    if (release !== undefined) callHost(() => release());
  }

  private nextId(prefix: 'i' | 's' | 't' | 'p'): string {
    this.requestCounter += 1;
    return `${prefix}${this.requestCounter}`;
  }

  private buildPonderMessage(id: string): WorkerPonderSettingsRequest {
    const message: WorkerPonderSettingsRequest = { id, type: 'ponderSettings' };
    const stored = this.ponderStored;
    if (stored.enabled !== undefined) message.enabled = stored.enabled;
    if (stored.k !== undefined) message.k = stored.k;
    if (stored.hidden !== undefined) message.hidden = stored.hidden;
    if (stored.discharging !== undefined) message.discharging = stored.discharging;
    return message;
  }

  // D4: re-applied on every worker this client spawns, right after its
  // init succeeds and before any search message is posted to it -- a
  // fire-and-forget post (init() resolves without waiting for this
  // message's own reply; postMessage order is enough).
  private applyStoredPonderSettings(worker: KomugiWorkerLike): void {
    if (Object.keys(this.ponderStored).length === 0) return;
    worker.postMessage(this.buildPonderMessage(this.nextId('p')));
  }

  private postPonderSettings(worker: KomugiWorkerLike): Promise<KomugiPonderState | null> {
    const id = this.nextId('p');
    const message = this.buildPonderMessage(id);
    return new Promise<KomugiPonderState | null>((resolve, reject) => {
      this.pendingPonder.set(id, { resolve, reject });
      worker.postMessage(message);
    });
  }

  private spawnAndInit(): void {
    const init = this.pendingInit;
    if (init === null) return;
    let worker: KomugiWorkerLike;
    try {
      worker = this.options.createWorker();
    } catch (error) {
      this.failInit(init, { code: 'worker-crash', stage: 'init', message: `createWorker failed: ${String(error)}` });
      return;
    }
    this.generation += 1;
    const generation = this.generation;
    this.worker = worker;
    this.readyInfo = null;
    worker.addEventListener('message', (event) => {
      if (generation === this.generation) this.onMessage(event.data);
    });
    const onCrash = (event: unknown): void => {
      if (generation === this.generation) this.onCrash(event);
    };
    worker.addEventListener('error', onCrash);
    worker.addEventListener('messageerror', onCrash);

    const id = this.nextId('i');
    init.id = id;
    const message: WorkerInitRequest = {
      id,
      type: 'init',
      modelUrl: this.options.model.url,
      modelSha256: this.options.model.sha256,
      ortAssets: this.options.ortAssets,
    };
    if (this.options.backend !== undefined) message.backend = this.options.backend;
    if (this.options.threads !== undefined) message.threads = this.options.threads;
    if (this.options.ortLogSeverityLevel !== undefined) message.ortLogSeverityLevel = this.options.ortLogSeverityLevel;
    if (this.maxThreads !== null) message.maxThreads = this.maxThreads;
    const cachedRecord = this.probeRecord ?? this.options.cachedRecord;
    if (cachedRecord !== undefined) message.cachedRecord = cachedRecord;

    const timeoutMs = this.options.initTimeoutMs ?? DEFAULT_INIT_TIMEOUT_MS;
    init.timer = globalThis.setTimeout(() => {
      this.failInit(init, { code: 'timeout', stage: 'init', message: `worker init timed out after ${timeoutMs} ms` });
    }, timeoutMs);
    worker.postMessage(message);
  }

  private postSearch(play: PendingPlay, request: PlayRequest): void {
    // KOM-1188: the model's planeMapVersion is the default, and a request
    // contradicting it is refused before anything is posted. `rules` is
    // never taken from the model.
    const bound = this.options.model.compatibility?.planeMapVersion;
    if (bound !== undefined && request.planeMapVersion !== undefined && request.planeMapVersion !== bound) {
      this.failPlay(play, {
        code: 'incompatible-model',
        stage: 'search',
        message: `planeMapVersion: request says ${request.planeMapVersion}, model expects ${bound}`,
      });
      return;
    }
    const planeMapVersion = request.planeMapVersion ?? bound;
    const worker = this.worker;
    if (worker === null) {
      this.settlePlay(play, { status: 'failed', failure: { code: 'worker-crash', stage: 'search', message: 'worker gone before the search' } });
      return;
    }
    const id = this.nextId('s');
    // KOM-1228: time-only (request.sims undefined) resolves to the node
    // ceiling here, client-side only -- the wire message for a sims
    // request stays byte-identical to before.
    const sims = request.sims !== undefined ? request.sims : TIME_ONLY_NODE_CEILING - 1;
    const message: WorkerRequest = { id, fen: request.fen, sims };
    if (request.history !== undefined) message.history = request.history;
    if (request.start !== undefined) message.start = request.start;
    if (request.rules !== undefined) message.rules = request.rules;
    if (planeMapVersion !== undefined) message.planeMapVersion = planeMapVersion;
    if (request.maxMs !== undefined) message.maxMs = request.maxMs;
    if (request.batch !== undefined) message.batch = request.batch;
    if (request.threads !== undefined) message.threads = request.threads;
    if (request.config !== undefined) message.config = request.config;
    if (request.stoppable !== undefined) message.stoppable = request.stoppable;
    if (request.topN !== undefined) message.topN = request.topN;
    if (request.progressEveryPlayouts !== undefined) message.progressEveryPlayouts = request.progressEveryPlayouts;
    if (request.progressEveryMs !== undefined) message.progressEveryMs = request.progressEveryMs;

    const budgetMs =
      request.maxMs !== undefined
        ? request.maxMs + (this.options.moveTimeoutGraceMs ?? DEFAULT_MOVE_TIMEOUT_GRACE_MS)
        : (this.options.moveTimeoutMs ?? DEFAULT_MOVE_TIMEOUT_MS);
    play.id = id;
    play.timer = globalThis.setTimeout(() => {
      // A running search cannot be interrupted without KOM-1195: terminate.
      this.failSearch(play, { code: 'timeout', stage: 'search', message: `no move within ${budgetMs} ms` });
    }, budgetMs);
    worker.postMessage(message);
    if (play.stopRequested) this.sendStop(play);
  }

  // KOM-1195 design section 7: analyze()'s own search message -- no sims,
  // no move timer (an analysis runs until moveNow()/stop()/the ceiling).
  private postAnalyze(play: PendingPlay, request: AnalyzeRequest): void {
    const bound = this.options.model.compatibility?.planeMapVersion;
    if (bound !== undefined && request.planeMapVersion !== undefined && request.planeMapVersion !== bound) {
      this.failPlay(play, {
        code: 'incompatible-model',
        stage: 'search',
        message: `planeMapVersion: request says ${request.planeMapVersion}, model expects ${bound}`,
      });
      return;
    }
    const planeMapVersion = request.planeMapVersion ?? bound;
    const worker = this.worker;
    if (worker === null) {
      this.settlePlay(play, { status: 'failed', failure: { code: 'worker-crash', stage: 'search', message: 'worker gone before the search' } });
      return;
    }
    const id = this.nextId('s');
    const message: WorkerAnalyzeRequest = { id, fen: request.fen, analyze: true };
    if (request.history !== undefined) message.history = request.history;
    if (request.start !== undefined) message.start = request.start;
    if (request.rules !== undefined) message.rules = request.rules;
    if (planeMapVersion !== undefined) message.planeMapVersion = planeMapVersion;
    if (request.batch !== undefined) message.batch = request.batch;
    if (request.threads !== undefined) message.threads = request.threads;
    if (request.config !== undefined) message.config = request.config;
    if (request.topN !== undefined) message.topN = request.topN;
    if (request.nodeCeiling !== undefined) message.nodeCeiling = request.nodeCeiling;
    if (request.progressEveryPlayouts !== undefined) message.progressEveryPlayouts = request.progressEveryPlayouts;
    if (request.progressEveryMs !== undefined) message.progressEveryMs = request.progressEveryMs;

    play.id = id;
    worker.postMessage(message);
    if (play.stopRequested) this.sendStop(play);
  }

  // KOM-1195 design section 7: posts the `stop` message for play's search
  // and arms the grace timer -- called once play.id names a live search,
  // either from moveNow() directly or deferred from postSearch/postAnalyze.
  private sendStop(play: PendingPlay): void {
    const worker = this.worker;
    if (worker === null || play.id === null) return;
    const message: WorkerStopRequest = { id: this.nextId('t'), type: 'stop', target: play.id };
    worker.postMessage(message);
    this.armGraceTimer(play);
  }

  private armGraceTimer(play: PendingPlay): void {
    if (play.graceTimer !== null) return;
    const graceMs = this.options.stopGraceMs ?? DEFAULT_STOP_GRACE_MS;
    play.graceTimer = globalThis.setTimeout(() => {
      // No reply within the grace period: terminate and settle failed.
      // Never counts toward the crash latch (section 7); a worker 'error'
      // event keeps today's handling.
      this.toIdle();
      this.failPlay(play, { code: 'timeout', stage: 'search', message: `no reply within ${graceMs} ms of stop` });
    }, graceMs);
  }

  private onMessage(data: unknown): void {
    if (!isObject(data)) return;
    if ((data as { type?: unknown }).type === 'progress') {
      const play = this.pendingPlay;
      if (play !== null && play.id !== null && (data as { id?: unknown }).id === play.id) {
        const onProgress = this.options.onProgress;
        if (onProgress !== undefined) callHost(() => onProgress(data as WorkerProgressEvent));
      }
      return;
    }
    const id = (data as { id?: unknown }).id;
    if (typeof id === 'string') {
      const ponder = this.pendingPonder.get(id);
      if (ponder !== undefined) {
        this.pendingPonder.delete(id);
        if (isErrReply(data)) {
          ponder.reject(new Error((data as WorkerReplyErr).err));
        } else {
          const r = data as WorkerPonderSettingsReplyOk;
          ponder.resolve({ enabled: r.enabled, k: r.k, hidden: r.hidden, discharging: r.discharging });
        }
        return;
      }
    }
    const init = this.pendingInit;
    if (init !== null && !init.settled && id === init.id) {
      this.onInitReply(init, data);
      return;
    }
    const play = this.pendingPlay;
    if (play !== null && !play.settled && play.id !== null && id === play.id) this.onSearchReply(play, data);
  }

  private onInitReply(init: PendingInit, data: object): void {
    if (isErrReply(data)) {
      const reply = data as WorkerInitReplyErr;
      if (reply.code === 'threads-timeout') {
        const record = reply.probeRecord;
        if (isObject(record)) this.relayProbeRecord(record);
        if (!init.restarted) {
          // KOM-1214: ONE fresh Komugi worker capped at 1 thread.
          init.restarted = true;
          if (init.timer !== null) globalThis.clearTimeout(init.timer);
          init.timer = null;
          this.terminateWorker();
          this.maxThreads = 1;
          this.spawnAndInit();
          return;
        }
      }
      this.failInit(init, classifyErrorReply('init', reply));
      return;
    }
    const checked = validateInitReply(data, init.id);
    if (!checked.ok) {
      this.failInit(init, checked.failure);
      return;
    }
    const reply = checked.reply;
    const info: KomugiReady = { backend: reply.backend, threads: reply.threads, batch: reply.batch };
    if (reply.fallback !== undefined) info.fallback = reply.fallback;
    this.readyInfo = info;
    // D4: posted synchronously, before settleInit's resolve() can let any
    // `.then(() => postSearch(...))` continuation run.
    if (this.worker !== null) this.applyStoredPonderSettings(this.worker);
    this.settleInit(init, { status: 'ready', info });
    this.relayProbeRecord(reply.probeRecord);
  }

  private onSearchReply(play: PendingPlay, data: object): void {
    if (isErrReply(data)) {
      // Coded or uncoded, an error reply keeps the worker (section 3).
      this.failPlay(play, classifyErrorReply('search', data));
      return;
    }
    const checked = validateSearchReply(data, play.id ?? '', ACTION_SIZE);
    if (!checked.ok) {
      this.failSearch(play, checked.failure);
      return;
    }
    this.consecutiveCrashes = 0;
    this.settlePlay(play, { status: 'move', reply: checked.reply });
  }

  private onCrash(event: unknown): void {
    const message = event instanceof Error ? event.message : 'worker error';
    const init = this.pendingInit;
    if (init !== null && !init.settled) {
      this.failInit(init, { code: 'worker-crash', stage: 'init', message });
      return;
    }
    // A crash outside init is never trusted for a later move (Risk R2):
    // terminate and restart lazily on the next play.
    const play = this.pendingPlay;
    if (play !== null && play.id !== null) {
      this.failSearch(play, { code: 'worker-crash', stage: 'search', message });
      return;
    }
    this.toIdle();
  }

  private relayProbeRecord(record: BackendRecord): void {
    this.probeRecord = record;
    const onProbeRecord = this.options.onProbeRecord;
    if (onProbeRecord !== undefined) callHost(() => onProbeRecord(record));
  }

  private failInit(init: PendingInit, failure: KomugiFailure): void {
    if (init.settled) return;
    this.terminateWorker();
    this.initPromise = null;
    this.latched = failure;
    this.last = failure;
    this.settleInit(init, { status: 'failed', failure });
  }

  private failPlay(play: PendingPlay, failure: KomugiFailure): void {
    if (play.settled) return;
    this.last = failure;
    this.settlePlay(play, { status: 'failed', failure });
  }

  // Search-stage timeout/worker-crash/illegal-reply (section 3's "idle;
  // next play restarts" row): terminate, count toward the crash cap, and
  // latch failed once maxConsecutiveCrashes is reached (section 3).
  private failSearch(play: PendingPlay, failure: KomugiFailure): void {
    this.toIdle();
    this.consecutiveCrashes += 1;
    const max = this.options.maxConsecutiveCrashes ?? DEFAULT_MAX_CONSECUTIVE_CRASHES;
    if (this.consecutiveCrashes >= max) this.latched = failure;
    this.failPlay(play, failure);
  }

  private settleInit(init: PendingInit, result: InitResult): void {
    if (init.settled) return;
    init.settled = true;
    if (init.timer !== null) globalThis.clearTimeout(init.timer);
    init.timer = null;
    if (this.pendingInit === init) this.pendingInit = null;
    init.resolve(result);
  }

  private settlePlay(play: PendingPlay, result: PlayResult): void {
    if (play.settled) return;
    play.settled = true;
    if (play.timer !== null) globalThis.clearTimeout(play.timer);
    play.timer = null;
    if (play.graceTimer !== null) globalThis.clearTimeout(play.graceTimer);
    play.graceTimer = null;
    if (this.pendingPlay === play) this.pendingPlay = null;
    play.resolve(result);
  }

  private terminateWorker(): void {
    const worker = this.worker;
    this.generation += 1;
    this.worker = null;
    this.readyInfo = null;
    if (worker !== null) worker.terminate();
    if (this.pendingPonder.size > 0) {
      for (const pending of this.pendingPonder.values()) pending.resolve(null);
      this.pendingPonder.clear();
    }
  }

  private toIdle(): void {
    this.terminateWorker();
    this.initPromise = null;
  }
}
