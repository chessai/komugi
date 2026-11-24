// KOM-1195 unit U4a (docs/design/kom1195-analysis-api.md section 3, section
// 9 U4a): stop targets, opt-in, the composed `shouldStop`, the macrotask
// yield and the chunked ponder key. Fake evaluator throughout, same
// convention as worker_fast.test.ts/deadline_batch.test.ts -- real-net
// parity is a later unit's job.
//
// U4a2 hardening (docs/reviews/2026-11-23-kom1195-u4a-code-review.md): the
// F1-F8 tests below kill mutants the original U4a suite left surviving.

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { decodeAction } from '../src/actions.ts';
import type { GpuLike, ProbeSession, SessionFactory } from '../src/backend.ts';
import { BackendError } from '../src/batch_evaluator.ts';
import type { MlhBatchEvaluator } from '../src/batch_evaluator.ts';
import { ACTION_SIZE, ADVANCED_POSITION } from '../src/constants.ts';
import { WorkerEngine } from '../src/engine.ts';
import type { WorkerInitEnv } from '../src/engine.ts';
import { makeFakeMlhEvaluator } from '../src/fake_evaluator.ts';
import { boardToFen } from '../src/fen.ts';
import { loadFixture } from './fixture.ts';
import { predictReplies } from '../src/ponder.ts';
import type { WorkerReply, WorkerRequest } from '../src/protocol.ts';
import { gameFromHistory } from '../src/root.ts';
import { buildSan, sanRecords } from '../src/san.ts';
import type { MatchNode } from '../src/search/node.ts';

function isOk(
  reply: WorkerReply,
): reply is WorkerReply & { action: number; san: string; stats: Record<string, unknown> } {
  return !('err' in reply);
}

function statsOf(reply: WorkerReply): Record<string, unknown> {
  const stats = { ...(reply as { stats: Record<string, unknown> }).stats };
  delete stats.ms;
  delete stats.pondered;
  return stats;
}

function fakeBatchEvaluator(): MlhBatchEvaluator {
  const leaf = makeFakeMlhEvaluator(ACTION_SIZE);
  return (states) => states.map((state) => leaf(state));
}

// A fake batch evaluator that calls `onCall(count)` BEFORE answering every
// batch -- the same convention deadline_batch.test.ts's fake clock uses to
// arm a deadline deterministically, here used to arm `handleStop` on exactly
// the Nth evaluator call regardless of `batch`.
function fakeBatchEvaluatorCounting(onCall: (count: number) => void): MlhBatchEvaluator {
  const leaf = makeFakeMlhEvaluator(ACTION_SIZE);
  let count = 0;
  return (states) => {
    count += 1;
    onCall(count);
    return states.map((state) => leaf(state));
  };
}

// A short real, legal history from the advanced start (root.test.ts's own
// convention, reused by ponder.test.ts): always the current position's
// first enumerated legal move.
function buildHistory(n: number): { history: string[]; fen: string } {
  const history: string[] = [];
  let board = gameFromHistory(history).board;
  for (let i = 0; i < n; i++) {
    const records = sanRecords(board);
    assert.ok(records.length > 0, `no legal move at ply ${i}`);
    history.push(records[0]!.san);
    board = gameFromHistory(history).board;
  }
  return { history, fen: boardToFen(board) };
}

// KOM-1195 U4a2 hardening (review F1): same construction as the reviewer's
// probe_h.test.ts -- varies which legal move is taken at each ply
// (`recs[(i * 7) % len]`), unlike `buildHistory`'s always-first-move walk,
// so the search explores enough different histories to find one where the
// chunked ponder reply actually differs from the unpondered continuous one.
function varyingHistory(n: number): string[] {
  const history: string[] = [];
  let board = gameFromHistory(history).board;
  for (let i = 0; i < n; i++) {
    const records = sanRecords(board);
    assert.ok(records.length > 0, `no legal move at ply ${i}`);
    history.push(records[(i * 7) % records.length]!.san);
    board = gameFromHistory(history).board;
  }
  return history;
}

const BASE = buildHistory(2);

function baseRequest(id: number | string, overrides: Partial<WorkerRequest> = {}): WorkerRequest {
  return { id, fen: BASE.fen, history: BASE.history, sims: 16, batch: 4, ...overrides };
}

function nextRequest(
  id: number | string,
  history: readonly string[],
  engineSan: string,
  replySan: string,
  overrides: Partial<WorkerRequest> = {},
): WorkerRequest {
  const nextHistory = [...history, engineSan, replySan];
  const fen = boardToFen(gameFromHistory(nextHistory).board);
  return { id, fen, history: nextHistory, sims: 16, batch: 4, ...overrides };
}

function topPredictionSan(
  root: MatchNode,
  afterEngineMoveBoard: ReturnType<typeof gameFromHistory>['board'],
  chosenAction: number,
): string {
  const [predicted] = predictReplies(root, chosenAction, 1);
  assert.ok(predicted !== undefined, 'expected at least one predicted reply under the chosen child');
  return buildSan(afterEngineMoveBoard, decodeAction(afterEngineMoveBoard, predicted).struct);
}

// KOM-1195 U4a2 hardening (review F3): the init harness levels_cap.test.ts
// already uses, reused verbatim so a BackendError fallback can be exercised
// against a stop target here too.
function realAdapterGpu(): GpuLike {
  return { requestAdapter: async () => ({ isFallbackAdapter: false }) };
}

function fakeProbeFactory(): { factory: SessionFactory; now: () => number } {
  let clock = 0;
  const factory: SessionFactory = {
    setNumThreads(): void {},
    async createWebgpu(): Promise<ProbeSession> {
      return {
        async warmup(): Promise<void> {},
        async run(batch: number): Promise<void> {
          clock += 1 * batch;
        },
      };
    },
    async createWasm(): Promise<ProbeSession> {
      return {
        async warmup(): Promise<void> {},
        async run(batch: number): Promise<void> {
          clock += 14 * batch;
        },
      };
    },
  };
  return { factory, now: () => clock };
}

// ---- (d) release on error / normal search: _stopTargets returns to 0 ------

test('(d) a prepareFast rejection releases its target: stop then gives unknown-target, size 0', async () => {
  const engine = new WorkerEngine();
  const rejected = await engine.handleFast(
    { id: 1, fen: ADVANCED_POSITION, sims: 16, batch: 0, stoppable: true },
    fakeBatchEvaluator(),
  );
  assert.ok('err' in rejected, JSON.stringify(rejected));
  assert.equal(engine.stopTargetsSize, 0);
  const stopReply = engine.handleStop({ id: 's', type: 'stop', target: 1 });
  assert.equal(stopReply.acknowledged, false);
  assert.equal((stopReply as { reason: string }).reason, 'unknown-target');
});

test('(d) a normal unstopped opted-in search releases its target: stop then gives unknown-target, size 0', async () => {
  const engine = new WorkerEngine();
  const reply = await engine.handleFast(
    { id: 2, fen: ADVANCED_POSITION, sims: 16, batch: 1, stoppable: true },
    fakeBatchEvaluator(),
  );
  assert.ok(isOk(reply), JSON.stringify(reply));
  assert.equal(engine.stopTargetsSize, 0);
  const stopReply = engine.handleStop({ id: 's', type: 'stop', target: 2 });
  assert.equal(stopReply.acknowledged, false);
  assert.equal((stopReply as { reason: string }).reason, 'unknown-target');
});

// ---- F3: BackendError fallback also releases the target (section 3 AC2d) --

test('F3 a stop during a failing attempt applies on the BackendError retry; size 0 after', async () => {
  const engine = new WorkerEngine();
  const { factory, now } = fakeProbeFactory();
  const leaf = makeFakeMlhEvaluator(ACTION_SIZE);
  let gpuCalls = 0;
  const createBatchBackend: WorkerInitEnv['createBatchBackend'] = async (backend) => {
    if (backend === 'webgpu') {
      return (states) => {
        gpuCalls += 1;
        if (gpuCalls === 3) {
          engine.handleStop({ id: 's', type: 'stop', target: 'fb' });
          throw new BackendError('fake device lost');
        }
        return states.map((state) => leaf(state));
      };
    }
    return (states) => states.map((state) => leaf(state));
  };
  const init = await engine.init(
    { id: 1, type: 'init', modelUrl: 'm.onnx', modelSha256: 'sha' },
    { sessionFactory: factory, gpu: realAdapterGpu(), isolated: true, hardwareConcurrency: 8, now, createBatchBackend },
  );
  assert.ok(!('err' in init) && init.backend === 'webgpu', JSON.stringify(init));
  const req: WorkerRequest = { id: 'fb', fen: ADVANCED_POSITION, sims: 400, batch: 4, stoppable: true };
  const reply = await engine.handleFast(req, engine.currentBatchEvaluator!);
  assert.ok(isOk(reply), JSON.stringify(reply));
  assert.equal(typeof reply.stats.fallback, 'string');
  assert.equal(reply.stats.deadlineHit, true);
  assert.equal(reply.stats.stopReason, 'stop');
  assert.equal(engine.stopTargetsSize, 0);
});

test('F3 a BackendError fallback with no stop leaves size 0 and sets no stopReason', async () => {
  const engine = new WorkerEngine();
  const { factory, now } = fakeProbeFactory();
  const leaf = makeFakeMlhEvaluator(ACTION_SIZE);
  let gpuCalls = 0;
  const createBatchBackend: WorkerInitEnv['createBatchBackend'] = async (backend) => {
    if (backend === 'webgpu') {
      return (states) => {
        gpuCalls += 1;
        if (gpuCalls > 2) throw new BackendError('fake device lost');
        return states.map((state) => leaf(state));
      };
    }
    return (states) => states.map((state) => leaf(state));
  };
  await engine.init(
    { id: 1, type: 'init', modelUrl: 'm.onnx', modelSha256: 'sha' },
    { sessionFactory: factory, gpu: realAdapterGpu(), isolated: true, hardwareConcurrency: 8, now, createBatchBackend },
  );
  const reply = await engine.handleFast(
    { id: 'fb2', fen: ADVANCED_POSITION, sims: 32, batch: 4, stoppable: true },
    engine.currentBatchEvaluator!,
  );
  assert.ok(isOk(reply), JSON.stringify(reply));
  assert.equal(typeof reply.stats.fallback, 'string');
  assert.equal('stopReason' in reply.stats, false);
  assert.equal(engine.stopTargetsSize, 0);
});

// ---- (e) stop on a non-opted-in in-flight request: not-stoppable -----------

test('(e) stop for a non-opted-in in-flight request gives not-stoppable, and the reply equals the unstopped one', async () => {
  const engine = new WorkerEngine();
  let stopResult: { acknowledged: boolean; reason?: string } | null = null;
  const request: WorkerRequest = { id: 3, fen: ADVANCED_POSITION, sims: 400, batch: 1 };
  const evaluator = fakeBatchEvaluatorCounting((count) => {
    if (count === 3) stopResult = engine.handleStop({ id: 's', type: 'stop', target: 3 });
  });
  const reply = await engine.handleFast(request, evaluator);
  assert.ok(isOk(reply), JSON.stringify(reply));
  assert.ok(stopResult !== null, 'expected handleStop to have run during the search');
  assert.equal(stopResult!.acknowledged, false);
  assert.equal(stopResult!.reason, 'not-stoppable');

  const baseline = await new WorkerEngine().handleFast(request, fakeBatchEvaluator());
  assert.ok(isOk(baseline), JSON.stringify(baseline));
  assert.equal(reply.action, baseline.action);
  assert.equal(reply.san, baseline.san);
  assert.deepEqual(statsOf(reply), statsOf(baseline));
});

// ---- The mandatory batch-400 stop test, plus its batch-1 (C1) variant ------

test('stoppable: true, sims: 400, stop on the third evaluator call -> deadlineHit, stopReason stop, partial playouts', async () => {
  const engine = new WorkerEngine();
  const request: WorkerRequest = { id: 4, fen: ADVANCED_POSITION, sims: 400, batch: 8, stoppable: true };
  const evaluator = fakeBatchEvaluatorCounting((count) => {
    if (count === 3) engine.handleStop({ id: 's', type: 'stop', target: 4 });
  });
  const reply = await engine.handleFast(request, evaluator);
  assert.ok(isOk(reply), JSON.stringify(reply));
  assert.equal(reply.stats.deadlineHit, true);
  assert.equal(reply.stats.stopReason, 'stop');
  assert.ok((reply.stats.playouts as number) < 400, `expected a partial playout count, got ${reply.stats.playouts}`);
});

test('C1 (batch 1 via the re-review fix): the same request at batch 1 with no maxMs also stops at the third evaluator call', async () => {
  const engine = new WorkerEngine();
  const request: WorkerRequest = { id: 5, fen: ADVANCED_POSITION, sims: 400, batch: 1, stoppable: true };
  const evaluator = fakeBatchEvaluatorCounting((count) => {
    if (count === 3) engine.handleStop({ id: 's', type: 'stop', target: 5 });
  });
  const reply = await engine.handleFast(request, evaluator);
  assert.ok(isOk(reply), JSON.stringify(reply));
  assert.equal(reply.stats.deadlineHit, true);
  assert.equal(reply.stats.stopReason, 'stop');
  assert.ok(
    (reply.stats.playouts as number) < 400,
    `expected a partial playout count (proving the chunked loop ran), got ${reply.stats.playouts}`,
  );
});

// ---- F6: contributor precedence -- stop beats deadline in the same chunk --

test('F6 precedence: deadline and stop firing in the same chunk resolve to stop', async () => {
  let now = 0;
  const engine = new WorkerEngine({ clock: () => now });
  const evaluator = fakeBatchEvaluatorCounting((count) => {
    if (count === 3) {
      now = 1000;
      engine.handleStop({ id: 's', type: 'stop', target: 'pr' });
    }
  });
  const reply = await engine.handleFast(
    { id: 'pr', fen: ADVANCED_POSITION, sims: 400, batch: 4, stoppable: true, maxMs: 10 },
    evaluator,
  );
  assert.ok(isOk(reply));
  assert.equal(reply.stats.deadlineHit, true);
  assert.equal(reply.stats.stopReason, 'stop');
});

// ---- F5: an opted-in deadline reports 'deadline'; an un-opted one reports --
// ---- no stopReason key at all ---------------------------------------------

test("F5 opted-in deadline sets stopReason 'deadline'; un-opted deadline carries no stopReason key", async () => {
  for (const opted of [true, false]) {
    let now = 0;
    const engine = new WorkerEngine({ clock: () => now });
    const evaluator = fakeBatchEvaluatorCounting((count) => {
      if (count === 3) now = 1000;
    });
    const request: WorkerRequest = { id: `dl-${opted}`, fen: ADVANCED_POSITION, sims: 400, batch: 4, maxMs: 10 };
    if (opted) request.stoppable = true;
    const reply = await engine.handleFast(request, evaluator);
    assert.ok(isOk(reply));
    assert.equal(reply.stats.deadlineHit, true);
    if (opted) assert.equal(reply.stats.stopReason, 'deadline');
    else assert.equal('stopReason' in reply.stats, false, JSON.stringify(reply.stats));
  }
});

// ---- F7: finish() must not stamp a stopReason on a root-win reply ---------

test('F7 an analyze-only request on a root-win position carries no stopReason; analyze+sims is rejected', async () => {
  const engine = new WorkerEngine();
  const fen = '8m/8D/9/9/9/9/9/9/4M4 -/- w 1 - 1';
  const req = {
    id: 'rw',
    fen,
    batch: 4,
    analyze: true,
  } as unknown as WorkerRequest;
  const reply = await engine.handleFast(req, fakeBatchEvaluator());
  assert.ok(isOk(reply), JSON.stringify(reply));
  if (!isOk(reply)) return;
  assert.equal(reply.stats.playouts, 0, 'expected a root-win scan with no playouts');
  assert.equal(reply.stats.deadlineHit, false);
  assert.equal('stopReason' in reply.stats, false, JSON.stringify(reply.stats));

  const bothReq = { id: 'rw2', fen, sims: 16, batch: 4, analyze: true } as unknown as WorkerRequest;
  const bothReply = await engine.handleFast(bothReq, fakeBatchEvaluator());
  assert.ok('err' in bothReply, JSON.stringify(bothReply));
  assert.match((bothReply as { err: string }).err, /exactly one of sims, level, or analyze is required/);
});

// ---- (f) yield is result-neutral (review F8: parity positions, stronger) --

test('(f) an opted-in request with no maxMs equals the same request with maxMs: 1e12 and no opt-in, over parity positions at batch 1 and 8', async () => {
  const positions = loadFixture()
    .filter((rec) => !rec.terminal)
    .slice(0, 5);
  assert.ok(positions.length > 0, 'no non-terminal parity positions');
  for (const pos of positions) {
    for (const batch of [1, 8]) {
      const opted: WorkerRequest = { id: `f-opt-${pos.id}-${batch}`, fen: pos.fen, sims: 32, batch, stoppable: true };
      const plain: WorkerRequest = { id: `f-plain-${pos.id}-${batch}`, fen: pos.fen, sims: 32, batch, maxMs: 1e12 };
      const optedReply = await new WorkerEngine().handleFast(opted, fakeBatchEvaluator());
      const plainReply = await new WorkerEngine().handleFast(plain, fakeBatchEvaluator());
      assert.ok(isOk(optedReply) && isOk(plainReply), JSON.stringify({ optedReply, plainReply }));
      if (!isOk(optedReply) || !isOk(plainReply)) continue;
      assert.equal(optedReply.action, plainReply.action, `${pos.id} batch ${batch}`);
      assert.equal(optedReply.san, plainReply.san, `${pos.id} batch ${batch}`);
      assert.ok(
        !('stopReason' in optedReply.stats),
        `${pos.id} batch ${batch}: unexpected stopReason on the opted-in reply`,
      );
      assert.deepEqual(statsOf(optedReply), statsOf(plainReply), `${pos.id} batch ${batch}`);
    }
  }
});

// ---- F4: the macrotask yield actually runs, on both branches --------------

test('F4 yield: a stop armed mid-search via setTimeout(0) on a microtask-only evaluator lands within 2 chunks', async () => {
  const engine = new WorkerEngine();
  const leaf = makeFakeMlhEvaluator(ACTION_SIZE);
  let calls = 0;
  let armedAt = -1;
  let k = -1;
  const evaluator: MlhBatchEvaluator = (states) => {
    calls += 1;
    if (calls === 4 && armedAt < 0) {
      armedAt = calls;
      setTimeout(() => {
        k = calls;
        engine.handleStop({ id: 's', type: 'stop', target: 'y' });
      }, 0);
    }
    return Promise.resolve(states.map((state) => leaf(state)));
  };
  const reply = await engine.handleFast(
    { id: 'y', fen: ADVANCED_POSITION, sims: 5000, batch: 1, stoppable: true },
    evaluator,
  );
  assert.ok(isOk(reply));
  assert.equal(reply.stats.deadlineHit, true, `playouts ${reply.stats.playouts}`);
  assert.ok(k > 0 && calls - k <= 2, `calls ${calls} k ${k}`);
});

test('F4 fresh-MessageChannel branch (setImmediate unavailable): completes, stop lands, no hang', async () => {
  const g = globalThis as { setImmediate?: unknown };
  const saved = g.setImmediate;
  g.setImmediate = undefined;
  try {
    const engine = new WorkerEngine();
    const leaf = makeFakeMlhEvaluator(ACTION_SIZE);
    let calls = 0;
    let k = -1;
    const evaluator: MlhBatchEvaluator = (states) => {
      calls += 1;
      if (calls === 4) {
        setTimeout(() => {
          k = calls;
          engine.handleStop({ id: 's', type: 'stop', target: 'mc' });
        }, 0);
      }
      return Promise.resolve(states.map((state) => leaf(state)));
    };
    const reply = await engine.handleFast(
      { id: 'mc', fen: ADVANCED_POSITION, sims: 5000, batch: 1, stoppable: true },
      evaluator,
    );
    assert.ok(isOk(reply));
    assert.equal(reply.stats.deadlineHit, true);
    assert.ok(calls - k <= 2, `calls ${calls} k ${k}`);
    const full = await new WorkerEngine().handleFast(
      { id: 'mc2', fen: ADVANCED_POSITION, sims: 64, batch: 4, stoppable: true },
      (states) => Promise.resolve(states.map((state) => leaf(state))),
    );
    assert.ok(isOk(full));
  } finally {
    g.setImmediate = saved;
  }
});

// ---- (g) duplicate in-flight ids (re-review N10) ----------------------------

test('(g) two concurrent un-opted handleFast calls with the same id both reply normally, equal to sequential replies', async () => {
  const request = (id: number | string): WorkerRequest => ({ id, fen: ADVANCED_POSITION, sims: 16, batch: 4 });

  const sequential = new WorkerEngine();
  const seq1 = await sequential.handleFast(request('seq'), fakeBatchEvaluator());
  const seq2 = await sequential.handleFast(request('seq'), fakeBatchEvaluator());

  const concurrent = new WorkerEngine();
  const [conc1, conc2] = await Promise.all([
    concurrent.handleFast(request('dup'), fakeBatchEvaluator()),
    concurrent.handleFast(request('dup'), fakeBatchEvaluator()),
  ]);
  assert.ok(isOk(seq1) && isOk(seq2) && isOk(conc1) && isOk(conc2));
  if (!isOk(seq1) || !isOk(seq2) || !isOk(conc1) || !isOk(conc2)) return;
  assert.deepEqual(statsOf(conc1), statsOf(seq1));
  assert.deepEqual(statsOf(conc2), statsOf(seq2));
  assert.equal(concurrent.stopTargetsSize, 0);
});

test('(g) an opted-in request whose id is live gets the duplicate error, and so does an un-opted one against a live opted-in id', async () => {
  const engine = new WorkerEngine();
  let duplicateReply: WorkerReply | null = null;
  const live: WorkerRequest = { id: 'dup2', fen: ADVANCED_POSITION, sims: 64, batch: 8, stoppable: true };
  const liveEvaluator = fakeBatchEvaluatorCounting((count) => {
    if (count === 2 && duplicateReply === null) {
      // Fired from inside the live search's own evaluator -- synchronous
      // with respect to the live target still being registered.
      void engine
        .handleFast({ id: 'dup2', fen: ADVANCED_POSITION, sims: 16, batch: 1 }, fakeBatchEvaluator())
        .then((r) => {
          duplicateReply = r;
        });
    }
  });
  const liveReply = await engine.handleFast(live, liveEvaluator);
  assert.ok(isOk(liveReply), JSON.stringify(liveReply));
  assert.ok(duplicateReply !== null, 'expected the duplicate-id call to have resolved');
  assert.ok(duplicateReply !== null && 'err' in duplicateReply, JSON.stringify(duplicateReply));
  if (duplicateReply !== null && 'err' in duplicateReply) {
    assert.equal(duplicateReply.err, 'duplicate in-flight search id');
  }

  // The reverse: a live opted-in target, probed synchronously via
  // `openStopTarget` directly (no evaluator timing games needed) -- an
  // un-opted request against that same id is rejected too.
  const target = engine.openStopTarget({ id: 'dup3', fen: ADVANCED_POSITION, sims: 16, stoppable: true });
  assert.ok(!('err' in target));
  if ('err' in target) return;
  const rejected = engine.openStopTarget({ id: 'dup3', fen: ADVANCED_POSITION, sims: 16 });
  assert.ok('err' in rejected);
  if ('err' in rejected) assert.equal(rejected.err, 'duplicate in-flight search id');
  engine.closeStopTarget(target);
});

// ---- F2: (g) hardening -- a NEW opted-in request against a live id --------

test('F2 an opted-in request against a live un-opted id is rejected; opted-in vs opted-in too', () => {
  const engine = new WorkerEngine();
  const live = engine.openStopTarget({ id: 'u', fen: ADVANCED_POSITION, sims: 16 });
  assert.ok(!('err' in live));
  const dup = engine.openStopTarget({ id: 'u', fen: ADVANCED_POSITION, sims: 16, stoppable: true });
  assert.ok('err' in dup, 'an opted-in new request against a live un-opted id must be rejected');
  if (!('err' in live)) engine.closeStopTarget(live);

  const live2 = engine.openStopTarget({ id: 'o', fen: ADVANCED_POSITION, sims: 16, topN: 2 });
  const dup2 = engine.openStopTarget({ id: 'o', fen: ADVANCED_POSITION, sims: 16, progressEveryMs: 5 });
  assert.ok('err' in dup2);
  if (!('err' in live2)) engine.closeStopTarget(live2);
  assert.equal(engine.stopTargetsSize, 0);
});

// ---- (h) ponder keys record opt-in (section 6, re-review N2) --------------

test('F1 (h) at sims 96/batch 8, with a precondition that the two replies differ, a chunked ponder hit equals the unpondered stoppable reply', async () => {
  let found = false;
  for (let n = 2; n <= 40 && !found; n += 2) {
    const history = varyingHistory(n);
    const fen = boardToFen(gameFromHistory(history).board);
    const req1: WorkerRequest = { id: 1, fen, history, sims: 96, batch: 8, stoppable: true };
    const engine = new WorkerEngine();
    const reply1 = await engine.handleFast(req1, fakeBatchEvaluator());
    if (!isOk(reply1)) continue;
    await engine.ponderingPromise();
    if (engine.ponderCacheSize === 0) continue;

    const afterEngineMove = gameFromHistory([...history, reply1.san]).board;
    const [predicted] = predictReplies(engine.tree!.root!, reply1.action, 1);
    if (predicted === undefined) continue;
    const replySan = buildSan(afterEngineMove, decodeAction(afterEngineMove, predicted).struct);
    const req2 = nextRequest(2, history, reply1.san, replySan, { stoppable: true, sims: 96, batch: 8 });

    // Precondition (so this case can go red): the unpondered chunked
    // (stoppable) reply at req2 must differ from the unpondered continuous
    // (plain) reply -- otherwise a dropped chunked branch would be
    // invisible to the hit assertion below.
    const offStoppable = new WorkerEngine();
    offStoppable.handlePonderSettings({ id: 's', type: 'ponderSettings', enabled: false });
    await offStoppable.handleFast(req1, fakeBatchEvaluator());
    const unpStoppable = await offStoppable.handleFast(req2, fakeBatchEvaluator());

    const offPlain = new WorkerEngine();
    offPlain.handlePonderSettings({ id: 's', type: 'ponderSettings', enabled: false });
    await offPlain.handleFast({ ...req1, stoppable: undefined } as WorkerRequest, fakeBatchEvaluator());
    const { stoppable: _stoppable, ...plainReq2 } = req2;
    const unpPlain = await offPlain.handleFast(plainReq2 as WorkerRequest, fakeBatchEvaluator());
    if (!isOk(unpStoppable) || !isOk(unpPlain)) continue;
    if (
      unpStoppable.action === unpPlain.action &&
      unpStoppable.san === unpPlain.san &&
      JSON.stringify(statsOf(unpStoppable)) === JSON.stringify(statsOf(unpPlain))
    ) {
      continue;
    }

    found = true;
    const hit = await engine.handleFast(req2, fakeBatchEvaluator());
    assert.ok(isOk(hit), JSON.stringify(hit));
    if (!isOk(hit)) continue;
    assert.equal(hit.stats.pondered, true, 'expected the chunked-opted continuation to hit the chunked entry');
    assert.equal(hit.action, unpStoppable.action);
    assert.equal(hit.san, unpStoppable.san);
    assert.deepEqual(
      statsOf(hit),
      statsOf(unpStoppable),
      'chunked ponder hit must equal the unpondered stoppable reply',
    );
  }
  assert.ok(found, 'no differing position found in the explored history-length range');
});

test('(h) an opted-in continuation never hits an un-opted (continuous-path) ponder entry', async () => {
  const engine = new WorkerEngine();
  const req1 = baseRequest(1);
  const reply1 = await engine.handleFast(req1, fakeBatchEvaluator());
  assert.ok(isOk(reply1), JSON.stringify(reply1));
  if (!isOk(reply1)) return;

  await engine.ponderingPromise();
  assert.ok(engine.ponderCacheSize > 0, 'expected at least one cached predicted reply');

  const afterEngineMove = gameFromHistory([...BASE.history, reply1.san]).board;
  const root = engine.tree!.root!;
  const replySan = topPredictionSan(root, afterEngineMove, reply1.action);
  // Same history/fen as the cached entry, but now opted in -- the chunked
  // flag must make this a miss, never a cross-kind hit.
  const req2 = nextRequest(2, BASE.history, reply1.san, replySan, { stoppable: true });

  const reply2 = await engine.handleFast(req2, fakeBatchEvaluator());
  assert.ok(isOk(reply2), JSON.stringify(reply2));
  if (!isOk(reply2)) return;
  assert.notEqual(reply2.stats.pondered, true, 'an opted-in request must never hit a continuous-path entry');
});
