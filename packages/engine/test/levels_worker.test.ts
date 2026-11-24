// KOM-1202 unit 2b (docs/design/kom1202-levels.md section 6/7): AC2 --
// a level request must resolve to exactly the same search as its raw twin
// `{ sims, batch, maxMs }` (resolveLevel's own output), through
// WorkerEngine.handleFast, including the ponder-hit (F1) and mid-search
// fallback (F2) paths where an earlier design draft diverged. Lives in its
// own file because worker_fast.test.ts has no `init()` fakes -- those are
// module-private in worker_init.test.ts, so this file brings its own
// minimal copy of only what it needs (no WebGPU/ORT/navigator.gpu is real
// anywhere below). KOM_SEARCH is required for the AC2 loop: it fails,
// never skips, without it (B1, search_batch.test.ts's own convention) --
// served by loadSearchFixture (docs/design/kom1196-golden-fixtures.md
// section 5), which falls back to the committed golden sample by default.

import assert from 'node:assert/strict';
import { test } from 'node:test';

import type { BackendKind, GpuLike, ProbeSession, SessionFactory } from '../src/backend.ts';
import { BackendError } from '../src/batch_evaluator.ts';
import type { MlhBatchEvaluator } from '../src/batch_evaluator.ts';
import { ACTION_SIZE, ADVANCED_POSITION } from '../src/constants.ts';
import { WorkerEngine } from '../src/engine.ts';
import type { WorkerInitEnv } from '../src/engine.ts';
import { makeFakeMlhEvaluator } from '../src/fake_evaluator.ts';
import { boardToFen } from '../src/fen.ts';
import { LEVEL_NAMES, resolveLevel } from '../src/levels.ts';
import type { LevelName } from '../src/levels.ts';
import type { WorkerInitReply, WorkerLevelRequest, WorkerReply, WorkerRequest } from '../src/protocol.ts';
import { gameFromHistory } from '../src/root.ts';
import { sanRecords } from '../src/san.ts';
import { loadSearchFixture } from './fixture.ts';

type SearchRecord = { position: { fen: string } };
type SearchFixture = { searches: SearchRecord[] };

// Only the FEN strings are used: this suite tests resolveLevel's parity
// through WorkerEngine, not against the Python fixture's own search
// ground truth (search_batch.test.ts/tree.test.ts already pin that).
// Fixture positions are played past the draft, so the refutation survey
// runs rather than being skipped as it is at the draft start.
function loadFixturePositions(): string[] {
  const fixture = loadSearchFixture<SearchFixture>();
  const fens = fixture.searches.map((rec) => rec.position.fen);
  assert.ok(fens.length > 0, 'no search records in the fixture');
  return fens;
}

function isOk(reply: WorkerReply): reply is WorkerReply & { stats: Record<string, unknown> } {
  return !('err' in reply);
}

function isInitOk(reply: WorkerInitReply): reply is WorkerInitReply & { backend: BackendKind; threads: number } {
  return !('err' in reply);
}

// Drops `ms` (always differs) and `phases` (only present when a request
// sets `timing: true`, which none here do -- kept for symmetry with the
// AC2 bullet's own wording).
function stripped(reply: WorkerReply & { stats: Record<string, unknown> }): Record<string, unknown> {
  const stats = { ...reply.stats };
  delete stats.ms;
  delete stats.phases;
  return { ...reply, stats };
}

function fakeBatchEvaluator(): MlhBatchEvaluator {
  const leaf = makeFakeMlhEvaluator(ACTION_SIZE);
  return (states) => states.map((state) => leaf(state));
}

function cleanBackend(): WorkerInitEnv['createBatchBackend'] {
  return async () => fakeBatchEvaluator();
}

// F2: a `createBatchBackend` whose 'webgpu' evaluator throws a
// BackendError once it has already answered `throwAfter` batches -- "a
// backend fails mid-search", not at session-build time. Same convention as
// worker_init.test.ts's own `throwingWebgpuBackend` (module-private there,
// so copied here).
function throwingWebgpuBackend(throwAfter: number): WorkerInitEnv['createBatchBackend'] {
  let calls = 0;
  const leaf = makeFakeMlhEvaluator(ACTION_SIZE);
  return async (backend: BackendKind) => {
    if (backend !== 'webgpu') return fakeBatchEvaluator();
    return (states) => {
      calls += 1;
      if (calls > throwAfter) throw new BackendError('fake: webgpu device lost mid-search');
      return states.map((state) => leaf(state));
    };
  };
}

function realAdapterGpu(): GpuLike {
  return { requestAdapter: async () => ({ isFallbackAdapter: false }) };
}

// Same `now` convention as worker_init.test.ts's own `makeFakeFactory`: the
// clock `probeBackend` itself reads, so the ladder can actually tell a 1ms
// backend from a 14ms one across these synchronous fakes.
function makeFakeFactory(opts: { webgpuMsPerPosition?: number; wasmMsPerPosition?: number }): {
  factory: SessionFactory;
  now: () => number;
} {
  const webgpuMs = opts.webgpuMsPerPosition ?? 1;
  const wasmMs = opts.wasmMsPerPosition ?? 14;
  let clock = 0;
  const factory: SessionFactory = {
    setNumThreads(): void {},
    async createWebgpu(): Promise<ProbeSession> {
      return {
        async warmup(): Promise<void> {},
        async run(batch: number): Promise<void> {
          clock += webgpuMs * batch;
        },
      };
    },
    async createWasm(): Promise<ProbeSession> {
      return {
        async warmup(): Promise<void> {},
        async run(batch: number): Promise<void> {
          clock += wasmMs * batch;
        },
      };
    },
  };
  return { factory, now: () => clock };
}

type BackendCaseName = 'webgpu' | 'wasm-mt' | 'wasm-1t';

// The three backend cases section 7's AC2 bullet requires, each driven
// through a real `WorkerEngine.init()` call (never `probeBackend` directly)
// so `_backend`/`_threads` land exactly where `resolveRequest` reads them.
function initFor(
  kase: BackendCaseName,
  engine: WorkerEngine,
  id: number,
  createBatchBackend: WorkerInitEnv['createBatchBackend'] = cleanBackend(),
): Promise<WorkerInitReply> {
  if (kase === 'webgpu') {
    const { factory, now } = makeFakeFactory({ webgpuMsPerPosition: 1, wasmMsPerPosition: 14 });
    return engine.init(
      { id, modelUrl: 'model.onnx', modelSha256: `sha-${kase}-${id}` },
      { sessionFactory: factory, gpu: realAdapterGpu(), isolated: true, hardwareConcurrency: 8, now, createBatchBackend },
    );
  }
  if (kase === 'wasm-mt') {
    const { factory, now } = makeFakeFactory({ wasmMsPerPosition: 14 });
    return engine.init(
      { id, modelUrl: 'model.onnx', modelSha256: `sha-${kase}-${id}` },
      { sessionFactory: factory, isolated: true, hardwareConcurrency: 8, now, createBatchBackend },
    );
  }
  const { factory, now } = makeFakeFactory({ wasmMsPerPosition: 14 });
  return engine.init(
    { id, modelUrl: 'model.onnx', modelSha256: `sha-${kase}-${id}` },
    { sessionFactory: factory, isolated: false, hardwareConcurrency: 16, now, createBatchBackend },
  );
}

const BACKEND_CASES: BackendCaseName[] = ['webgpu', 'wasm-mt', 'wasm-1t'];

test('AC2 parity: a level request resolves to exactly its raw twin, for every level and backend case, across the fixture positions', async () => {
  const fens = loadFixturePositions();
  let checked = 0;
  let nextId = 1;
  for (const level of LEVEL_NAMES) {
    for (const kase of BACKEND_CASES) {
      const engineA = new WorkerEngine({ clock: () => 0 });
      const engineB = new WorkerEngine({ clock: () => 0 });
      const initA = await initFor(kase, engineA, nextId++);
      const initB = await initFor(kase, engineB, nextId++);
      assert.ok(isInitOk(initA), `${level}/${kase}: engine A init failed: ${JSON.stringify(initA)}`);
      assert.ok(isInitOk(initB), `${level}/${kase}: engine B init failed: ${JSON.stringify(initB)}`);
      if (!isInitOk(initA) || !isInitOk(initB)) continue;

      const resolved = resolveLevel(level, { backend: initB.backend, threads: initB.threads });

      for (const fen of fens) {
        const id = nextId++;
        const reqA: WorkerLevelRequest = { id, fen, level };
        const reqB: WorkerRequest = { id, fen, sims: resolved.sims, batch: resolved.batch, maxMs: resolved.maxMs };

        const replyA = await engineA.handleFast(reqA, engineA.currentBatchEvaluator!);
        const replyB = await engineB.handleFast(reqB, engineB.currentBatchEvaluator!);
        assert.ok(isOk(replyA), `${level}/${kase}: level reply errored: ${JSON.stringify(replyA)}`);
        assert.ok(isOk(replyB), `${level}/${kase}: raw-twin reply errored: ${JSON.stringify(replyB)}`);
        if (!isOk(replyA) || !isOk(replyB)) continue;

        assert.equal(replyA.stats.deadlineHit, false, `${level}/${kase}/${fen}: level deadlineHit`);
        assert.equal(replyB.stats.deadlineHit, false, `${level}/${kase}/${fen}: raw-twin deadlineHit`);
        assert.deepEqual(stripped(replyA), stripped(replyB), `${level}/${kase}/${fen}: replies diverged`);
        checked++;
      }
    }
  }
  assert.ok(checked > 0, 'no fixture position was checked');
});

test('exact on wasm 1 thread: a maxMs deadline that never fires is exact (ticket point)', async () => {
  const withCap = new WorkerEngine();
  const noCap = new WorkerEngine();
  const requestWithCap: WorkerRequest = { id: 1, fen: ADVANCED_POSITION, sims: 64, batch: 1, maxMs: 1e12 };
  const requestNoCap: WorkerRequest = { id: 1, fen: ADVANCED_POSITION, sims: 64, batch: 1 };

  const replyWithCap = await withCap.handleFast(requestWithCap, fakeBatchEvaluator());
  const replyNoCap = await noCap.handleFast(requestNoCap, fakeBatchEvaluator());
  assert.ok(isOk(replyWithCap), JSON.stringify(replyWithCap));
  assert.ok(isOk(replyNoCap), JSON.stringify(replyNoCap));
  if (!isOk(replyWithCap) || !isOk(replyNoCap)) return;

  assert.equal(replyWithCap.stats.deadlineHit, false, 'a deadline 1e12ms away must never fire');
  assert.deepEqual(stripped(replyWithCap), stripped(replyNoCap));
});

// Same convention as ponder.test.ts/ponder_predicted.test.ts's own
// `buildHistory`: the current position's first enumerated legal move,
// every ply.
function buildHistory(n: number): { history: string[]; fen: string } {
  const history: string[] = [];
  let board = gameFromHistory(history).board;
  for (let i = 0; i < n; i++) {
    const records = sanRecords(board);
    assert.ok(records.length > 0, `no legal move at ply ${i}`);
    history.push(records[0].san);
    board = gameFromHistory(history).board;
  }
  return { history, fen: boardToFen(board) };
}

const BASE = buildHistory(2);

test('F1 ponder-hit path: a level request and its raw twin produce the same pondered second reply', async () => {
  let nextId = 1;
  for (const level of LEVEL_NAMES) {
    const engineA = new WorkerEngine({ clock: () => 0 });
    const engineB = new WorkerEngine({ clock: () => 0 });
    const initA = await initFor('wasm-mt', engineA, nextId++);
    const initB = await initFor('wasm-mt', engineB, nextId++);
    assert.ok(isInitOk(initA) && isInitOk(initB), `${level}: init failed`);
    if (!isInitOk(initA) || !isInitOk(initB)) continue;

    const resolved = resolveLevel(level, { backend: initB.backend, threads: initB.threads });

    const req1Id = nextId++;
    const req1A: WorkerLevelRequest = { id: req1Id, fen: BASE.fen, history: BASE.history, level };
    const req1B: WorkerRequest = {
      id: req1Id,
      fen: BASE.fen,
      history: BASE.history,
      sims: resolved.sims,
      batch: resolved.batch,
      maxMs: resolved.maxMs,
    };
    const reply1A = await engineA.handleFast(req1A, engineA.currentBatchEvaluator!);
    const reply1B = await engineB.handleFast(req1B, engineB.currentBatchEvaluator!);
    assert.ok(isOk(reply1A) && isOk(reply1B), `${level}: first reply errored`);
    if (!isOk(reply1A) || !isOk(reply1B)) continue;

    await engineA.ponderingPromise();
    await engineB.ponderingPromise();

    assert.ok(reply1A.predictedReply !== undefined, `${level}: expected a predictedReply to drive the second request`);
    if (reply1A.predictedReply === undefined) continue;
    const nextHistoryA = [...BASE.history, reply1A.san, reply1A.predictedReply.san];
    const nextFenA = boardToFen(gameFromHistory(nextHistoryA).board);

    assert.ok(reply1B.predictedReply !== undefined, `${level}: expected a predictedReply on the raw twin too`);
    if (reply1B.predictedReply === undefined) continue;
    const nextHistoryB = [...BASE.history, reply1B.san, reply1B.predictedReply.san];
    const nextFenB = boardToFen(gameFromHistory(nextHistoryB).board);

    const req2Id = nextId++;
    const req2A: WorkerLevelRequest = { id: req2Id, fen: nextFenA, history: nextHistoryA, level };
    const req2B: WorkerRequest = {
      id: req2Id,
      fen: nextFenB,
      history: nextHistoryB,
      sims: resolved.sims,
      batch: resolved.batch,
      maxMs: resolved.maxMs,
    };
    const reply2A = await engineA.handleFast(req2A, engineA.currentBatchEvaluator!);
    const reply2B = await engineB.handleFast(req2B, engineB.currentBatchEvaluator!);
    assert.ok(isOk(reply2A) && isOk(reply2B), `${level}: second reply errored`);
    if (!isOk(reply2A) || !isOk(reply2B)) continue;

    assert.equal(reply2A.stats.pondered, true, `${level}: level request's second reply must be a ponder hit`);
    assert.equal(reply2B.stats.pondered, true, `${level}: raw twin's second reply must be a ponder hit`);
    assert.deepEqual(stripped(reply2A), stripped(reply2B), `${level}: pondered replies diverged`);
  }
});

test('F2 fallback path: a level request that hits a mid-search backend failure resolves identically to its raw twin', async () => {
  let nextId = 1;
  for (const level of LEVEL_NAMES) {
    const engineA = new WorkerEngine({ clock: () => 0 });
    const engineB = new WorkerEngine({ clock: () => 0 });
    // WebGPU without cross-origin isolation (threads 1) -- the common site
    // case where a mid-search failure must drop to wasm, and (landing at 1
    // thread) force batch 1 for the retry, identically for both requests.
    const { factory: factoryA, now: nowA } = makeFakeFactory({ webgpuMsPerPosition: 1, wasmMsPerPosition: 14 });
    const { factory: factoryB, now: nowB } = makeFakeFactory({ webgpuMsPerPosition: 1, wasmMsPerPosition: 14 });
    // Same modelSha256 for both engines -- a fallback reply's stats carry
    // `probeRecord.key`, which is derived from it (backendCacheKey), and
    // that key must match between the level reply and its raw twin just
    // like every other stats field.
    const modelSha256 = `sha-f2-${level}`;
    const initA = await engineA.init(
      { id: nextId++, modelUrl: 'model.onnx', modelSha256 },
      {
        sessionFactory: factoryA,
        gpu: realAdapterGpu(),
        isolated: false,
        hardwareConcurrency: 8,
        now: nowA,
        createBatchBackend: throwingWebgpuBackend(1),
      },
    );
    const initB = await engineB.init(
      { id: nextId++, modelUrl: 'model.onnx', modelSha256 },
      {
        sessionFactory: factoryB,
        gpu: realAdapterGpu(),
        isolated: false,
        hardwareConcurrency: 8,
        now: nowB,
        createBatchBackend: throwingWebgpuBackend(1),
      },
    );
    assert.ok(isInitOk(initA) && initA.backend === 'webgpu', `${level}: expected init A to settle on webgpu, got ${JSON.stringify(initA)}`);
    assert.ok(isInitOk(initB) && initB.backend === 'webgpu', `${level}: expected init B to settle on webgpu, got ${JSON.stringify(initB)}`);
    if (!isInitOk(initA) || !isInitOk(initB)) continue;

    const resolved = resolveLevel(level, { backend: initB.backend, threads: initB.threads });

    const id = nextId++;
    const reqA: WorkerLevelRequest = { id, fen: ADVANCED_POSITION, level };
    const reqB: WorkerRequest = { id, fen: ADVANCED_POSITION, sims: resolved.sims, batch: resolved.batch, maxMs: resolved.maxMs };

    const replyA = await engineA.handleFast(reqA, engineA.currentBatchEvaluator!);
    const replyB = await engineB.handleFast(reqB, engineB.currentBatchEvaluator!);
    assert.ok(isOk(replyA), `${level}: level reply errored: ${JSON.stringify(replyA)}`);
    assert.ok(isOk(replyB), `${level}: raw-twin reply errored: ${JSON.stringify(replyB)}`);
    if (!isOk(replyA) || !isOk(replyB)) continue;

    assert.equal(replyA.stats.backend, 'wasm', `${level}: level reply must land on the fallback backend`);
    assert.equal(replyB.stats.backend, 'wasm', `${level}: raw-twin reply must land on the fallback backend`);
    assert.equal(typeof replyA.stats.fallback, 'string', `${level}: level reply must stamp a fallback reason`);
    assert.equal(typeof replyB.stats.fallback, 'string', `${level}: raw-twin reply must stamp a fallback reason`);
    assert.deepEqual(stripped(replyA), stripped(replyB), `${level}: fallback replies diverged`);
  }
});
