// KOM-1195 unit U4c (docs/design/kom1195-analysis-api.md section 6, section
// 9 U4c): analysis mode -- the three-way sims/level/analyze discriminant,
// `sims = nodeCeiling - 1`, the ceiling constants and their byte budget, the
// whole-bias-state snapshot/restore (around an analysis AND in the
// `BackendError` fallback), no pondering after analysis, the exact-mode
// rejection, and section 3's case a (stop mid-analysis). Fake evaluator
// throughout. Run with `node --expose-gc --test` to exercise the
// bytes-per-node pin; without the flag the pin is a silent no-op (the
// project's full-regression gate never passes --expose-gc, so a hard
// failure there would be permanent, not a signal of a forgotten flag).

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { decodeAction } from '../src/actions.ts';
import type { GpuLike, ProbeSession, SessionFactory } from '../src/backend.ts';
import { BackendError } from '../src/batch_evaluator.ts';
import type { MlhBatchEvaluator } from '../src/batch_evaluator.ts';
import { ACTION_SIZE } from '../src/constants.ts';
import {
  ANALYSIS_DEFAULT_NODE_CEILING,
  ANALYSIS_MAX_NODE_CEILING,
  ANALYSIS_NODE_BYTES_ESTIMATE,
  WorkerEngine,
} from '../src/engine.ts';
import type { WorkerInitEnv } from '../src/engine.ts';
import { makeFakeMlhEvaluator } from '../src/fake_evaluator.ts';
import { boardToFen } from '../src/fen.ts';
import { predictReplies } from '../src/ponder.ts';
import type {
  RootMoveStat,
  WorkerAnalyzeRequest,
  WorkerProgressEvent,
  WorkerReply,
  WorkerRequest,
} from '../src/protocol.ts';
import { gameFromHistory } from '../src/root.ts';
import { buildSan, sanRecords } from '../src/san.ts';
import type { BiasEntry } from '../src/search/bias.ts';

function isOk(
  reply: WorkerReply,
): reply is WorkerReply & { action: number; san: string; stats: Record<string, unknown> } {
  return !('err' in reply);
}

function fakeBatchEvaluator(): MlhBatchEvaluator {
  const leaf = makeFakeMlhEvaluator(ACTION_SIZE);
  return (states) => states.map((state) => leaf(state));
}

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

const START = buildHistory(0);
const BASE = buildHistory(2);
const MARSHAL_CAPTURE_FEN = '8m/8D/9/9/9/9/9/9/4M4 -/- w 1 - 1';

function quietEngine(opts: ConstructorParameters<typeof WorkerEngine>[0] = {}): WorkerEngine {
  const engine = new WorkerEngine(opts);
  engine.handlePonderSettings({ id: 'p', type: 'ponderSettings', enabled: false });
  return engine;
}

// A plain-data copy of the bias table, comparable with deepEqual.
function biasOf(engine: WorkerEngine): Array<[number, number[], number, number]> {
  const entries: Map<number, BiasEntry> = engine.tree?.bias.entries ?? new Map();
  return [...entries.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([key, e]) => [key, Array.from(e.errSum), e.weightSum, e.gen]);
}

// The init harness kom1195_stop.test.ts/levels_cap.test.ts already use.
function realAdapterGpu(): GpuLike {
  return { requestAdapter: async () => ({ isFallbackAdapter: false }) };
}

function fakeProbeFactory(): { factory: SessionFactory; now: () => number } {
  let clock = 0;
  const session = (cost: number): ProbeSession => ({
    async warmup(): Promise<void> {},
    async run(batch: number): Promise<void> {
      clock += cost * batch;
    },
  });
  const factory: SessionFactory = {
    setNumThreads(): void {},
    async createWebgpu(): Promise<ProbeSession> {
      return session(1);
    },
    async createWasm(): Promise<ProbeSession> {
      return session(14);
    },
  };
  return { factory, now: () => clock };
}

// ---- ceiling by construction -----------------------------------------------

test('nodeCeiling 50: ok reply, nodes <= 50, playouts 49, stopReason ceiling, deadlineHit false', async () => {
  const engine = quietEngine();
  const req: WorkerAnalyzeRequest = { id: 'c50', fen: START.fen, analyze: true, nodeCeiling: 50, batch: 4 };
  const reply = await engine.handleFast(req, fakeBatchEvaluator());
  assert.ok(isOk(reply), JSON.stringify(reply));
  if (!isOk(reply)) return;
  assert.ok((reply.stats.nodes as number) <= 50, `nodes ${String(reply.stats.nodes)}`);
  assert.equal(reply.stats.playouts, 49);
  assert.equal(reply.stats.sims, 49);
  assert.equal(reply.stats.stopReason, 'ceiling');
  assert.equal(reply.stats.deadlineHit, false);
  assert.equal(engine.stopTargetsSize, 0);
});

test('nodeCeiling 50 at batch 1 also runs exactly 49 playouts with stopReason ceiling', async () => {
  const engine = quietEngine();
  const reply = await engine.handleFast(
    { id: 'c50b1', fen: START.fen, analyze: true, nodeCeiling: 50 },
    fakeBatchEvaluator(),
  );
  assert.ok(isOk(reply), JSON.stringify(reply));
  if (!isOk(reply)) return;
  assert.equal(reply.stats.playouts, 49);
  assert.ok((reply.stats.nodes as number) <= 50);
  assert.equal(reply.stats.stopReason, 'ceiling');
});

// ---- no ponder after analysis (and no ponder hit for one) ------------------

test('an analysis with history leaves no ponder entry: ponderingPromise settles within 2 s, cache size 0', async () => {
  const engine = new WorkerEngine();
  const reply = await engine.handleFast(
    { id: 'ph', fen: BASE.fen, history: BASE.history, analyze: true, nodeCeiling: 50, batch: 4 },
    fakeBatchEvaluator(),
  );
  assert.ok(isOk(reply), JSON.stringify(reply));
  let timer: ReturnType<typeof setTimeout> | undefined;
  const timeout = new Promise<'timeout'>((resolve) => {
    timer = setTimeout(() => resolve('timeout'), 2000);
  });
  const settled = await Promise.race([engine.ponderingPromise().then(() => 'settled' as const), timeout]);
  clearTimeout(timer);
  assert.equal(settled, 'settled');
  assert.equal(engine.ponderCacheSize, 0);

  // Control: the same request as a plain stoppable play DOES ponder (so the
  // assertion above can go red).
  const control = new WorkerEngine();
  const r2 = await control.handleFast(
    { id: 'ph2', fen: BASE.fen, history: BASE.history, sims: 49, batch: 4, stoppable: true },
    fakeBatchEvaluator(),
  );
  assert.ok(isOk(r2), JSON.stringify(r2));
  await control.ponderingPromise();
  assert.ok(control.ponderCacheSize > 0, 'control: a stoppable history play should leave a ponder entry');
});

test('an analysis of a predicted position never takes the ponder-cache hit a stoppable play would', async () => {
  const setup = async (): Promise<{ engine: WorkerEngine; next: { history: string[]; fen: string } }> => {
    const engine = new WorkerEngine();
    const r1 = await engine.handleFast(
      { id: 1, fen: BASE.fen, history: BASE.history, sims: 16, batch: 4, stoppable: true },
      fakeBatchEvaluator(),
    );
    assert.ok(isOk(r1), JSON.stringify(r1));
    if (!isOk(r1)) throw new Error('unreachable');
    await engine.ponderingPromise();
    assert.ok(engine.ponderCacheSize > 0, 'expected a ponder entry');
    const after = gameFromHistory([...BASE.history, r1.san]).board;
    const [predicted] = predictReplies(engine.tree!.root!, r1.action, 1);
    assert.ok(predicted !== undefined);
    const replySan = buildSan(after, decodeAction(after, predicted!).struct);
    const history = [...BASE.history, r1.san, replySan];
    return { engine, next: { history, fen: boardToFen(gameFromHistory(history).board) } };
  };

  // Control: the stoppable play at sims 16 is a hit (precondition).
  const a = await setup();
  const hit = await a.engine.handleFast(
    { id: 2, fen: a.next.fen, history: a.next.history, sims: 16, batch: 4, stoppable: true },
    fakeBatchEvaluator(),
  );
  assert.ok(isOk(hit), JSON.stringify(hit));
  assert.equal(hit.stats.pondered, true, 'precondition: the stoppable play should be a ponder hit');

  // The analysis resolving to the same sims (nodeCeiling 17) is not.
  const b = await setup();
  const analysis = await b.engine.handleFast(
    { id: 2, fen: b.next.fen, history: b.next.history, analyze: true, nodeCeiling: 17, batch: 4 },
    fakeBatchEvaluator(),
  );
  assert.ok(isOk(analysis), JSON.stringify(analysis));
  assert.equal('pondered' in analysis.stats, false, JSON.stringify(analysis.stats));
  assert.equal(analysis.stats.stopReason, 'ceiling');
});

// ---- bias isolation --------------------------------------------------------

test('two 2000-node analyses leave the bias table (keys and contents) exactly as before them', async () => {
  const engine = quietEngine();
  const play = await engine.handleFast({ id: 'p0', fen: START.fen, sims: 64, batch: 4 }, fakeBatchEvaluator());
  assert.ok(isOk(play), JSON.stringify(play));
  const before = biasOf(engine);
  assert.ok(before.length > 0, 'precondition: a non-empty bias table');
  for (const id of ['a1', 'a2']) {
    const reply = await engine.handleFast(
      { id, fen: BASE.fen, analyze: true, nodeCeiling: 2000, batch: 8 },
      fakeBatchEvaluator(),
    );
    assert.ok(isOk(reply), JSON.stringify(reply));
    assert.equal((reply as { stats: { playouts: number } }).stats.playouts, 1999);
  }
  assert.deepEqual(biasOf(engine), before);
});

test('rules key: (play A, analyse B, play A) equals (play A, play A); (play A, analyse B, play B) equals a fresh (play B)', async () => {
  const playA: WorkerRequest = { id: 'A', fen: START.fen, sims: 32, batch: 4 };
  const playB: WorkerRequest = { id: 'B', fen: START.fen, sims: 32, batch: 4, rules: 'ligungi' };
  const analyseB: WorkerAnalyzeRequest = {
    id: 'aB',
    fen: BASE.fen,
    analyze: true,
    nodeCeiling: 200,
    batch: 4,
    rules: 'ligungi',
  };
  const run = async (reqs: Array<WorkerRequest | WorkerAnalyzeRequest>): Promise<WorkerEngine> => {
    const engine = quietEngine();
    for (const req of reqs) {
      const reply = await engine.handleFast(req, fakeBatchEvaluator());
      assert.ok(isOk(reply), JSON.stringify(reply));
    }
    return engine;
  };

  const baselineAA = biasOf(await run([playA, playA]));
  const withAnalysisAA = biasOf(await run([playA, analyseB, playA]));
  assert.notDeepEqual(baselineAA, biasOf(await run([playA])), 'precondition: a second A play changes the table');
  assert.deepEqual(withAnalysisAA, baselineAA);

  const aOnly = biasOf(await run([playA]));
  const baselineB = biasOf(await run([playB]));
  const withAnalysisAB = biasOf(await run([playA, analyseB, playB]));
  assert.deepEqual(withAnalysisAB, baselineB);
  const bKeys = new Set(baselineB.map(([key]) => key));
  const aEraKeys = aOnly.map(([key]) => key).filter((key) => !bKeys.has(key));
  const left = new Set(withAnalysisAB.map(([key]) => key));
  for (const key of aEraKeys) assert.equal(left.has(key), false, `A-era key ${key} survived under B`);
});

test('fallback: (play A, rules-B request whose first attempt throws BackendError) leaves no A-era state after the retry', async () => {
  const makeEngine = async (): Promise<{ engine: WorkerEngine; failNext: { on: boolean } }> => {
    const engine = quietEngine();
    const { factory, now } = fakeProbeFactory();
    const leaf = makeFakeMlhEvaluator(ACTION_SIZE);
    const failNext = { on: false };
    let calls = 0;
    const createBatchBackend: WorkerInitEnv['createBatchBackend'] = async (backend) => {
      if (backend === 'webgpu') {
        return (states) => {
          if (failNext.on) {
            calls += 1;
            if (calls === 3) throw new BackendError('fake device lost');
          }
          return states.map((state) => leaf(state));
        };
      }
      return (states) => states.map((state) => leaf(state));
    };
    const init = await engine.init(
      { id: 'i', type: 'init', modelUrl: 'm.onnx', modelSha256: 'sha' },
      { sessionFactory: factory, gpu: realAdapterGpu(), isolated: true, hardwareConcurrency: 8, now, createBatchBackend },
    );
    assert.ok(!('err' in init) && init.backend === 'webgpu', JSON.stringify(init));
    return { engine, failNext };
  };
  const playA: WorkerRequest = { id: 'A', fen: START.fen, sims: 32, batch: 1 };
  const playB: WorkerRequest = { id: 'B', fen: START.fen, sims: 32, batch: 1, rules: 'ligungi' };

  const { engine, failNext } = await makeEngine();
  const rA = await engine.handleFast(playA, engine.currentBatchEvaluator!);
  assert.ok(isOk(rA), JSON.stringify(rA));
  const aOnly = biasOf(engine);
  assert.ok(aOnly.length > 0);
  failNext.on = true;
  const rB = await engine.handleFast(playB, engine.currentBatchEvaluator!);
  assert.ok(isOk(rB), JSON.stringify(rB));
  if (!isOk(rB)) return;
  assert.equal(typeof rB.stats.fallback, 'string', 'precondition: the first attempt fell back');

  const fresh = await makeEngine();
  const rB2 = await fresh.engine.handleFast(playB, fresh.engine.currentBatchEvaluator!);
  assert.ok(isOk(rB2), JSON.stringify(rB2));
  const baselineB = biasOf(fresh.engine);
  const after = biasOf(engine);
  assert.deepEqual(after, baselineB);
  const bKeys = new Set(baselineB.map(([key]) => key));
  const left = new Set(after.map(([key]) => key));
  for (const [key] of aOnly) {
    if (!bKeys.has(key)) assert.equal(left.has(key), false, `A-era key ${key} survived the fallback retry`);
  }
});

// ---- bytes-per-node pin (N7) -----------------------------------------------

test('bytes per node over a nodeCeiling 5000 analysis (read before the bias restore) stay within the estimate', async () => {
  const gc = (globalThis as { gc?: () => void }).gc;
  if (typeof gc !== 'function') return;
  let peak: number | null = null;
  const engine = quietEngine({
    onProgress: (event) => {
      if (event.playouts === 4999) {
        gc();
        peak = process.memoryUsage().heapUsed;
      }
    },
  });
  gc();
  const base = process.memoryUsage().heapUsed;
  const reply = await engine.handleFast(
    { id: 'bpn', fen: START.fen, analyze: true, nodeCeiling: 5000, batch: 8, progressEveryPlayouts: 4999 },
    fakeBatchEvaluator(),
  );
  assert.ok(isOk(reply), JSON.stringify(reply));
  if (!isOk(reply)) return;
  assert.equal(reply.stats.playouts, 4999);
  assert.ok(peak !== null, 'expected the final progress event at the cap');
  const nodes = reply.stats.nodes as number;
  const perNode = ((peak as unknown as number) - base) / nodes;
  assert.ok(perNode <= ANALYSIS_NODE_BYTES_ESTIMATE, `measured ${perNode.toFixed(0)} B/node over ${nodes} nodes`);
  assert.ok(ANALYSIS_DEFAULT_NODE_CEILING * ANALYSIS_NODE_BYTES_ESTIMATE <= 64 * 1024 * 1024);
  assert.ok(ANALYSIS_MAX_NODE_CEILING * ANALYSIS_NODE_BYTES_ESTIMATE <= 256 * 1024 * 1024);
});

// ---- ordering (N8) ---------------------------------------------------------

test('an analysis with topN 3 keeps descriptor last; a root-win analysis has no stopReason', async () => {
  const engine = quietEngine();
  const reply = await engine.handleFast(
    { id: 'tn', fen: START.fen, analyze: true, nodeCeiling: 50, batch: 4, topN: 3 },
    fakeBatchEvaluator(),
  );
  assert.ok(isOk(reply), JSON.stringify(reply));
  const keys = Object.keys(reply as unknown as Record<string, unknown>);
  assert.equal(keys[keys.length - 1], 'descriptor');
  assert.ok(keys.includes('topMoves'));

  const rw = await engine.handleFast(
    { id: 'rw', fen: MARSHAL_CAPTURE_FEN, analyze: true, nodeCeiling: 17, batch: 4 },
    fakeBatchEvaluator(),
  );
  assert.ok(isOk(rw), JSON.stringify(rw));
  if (!isOk(rw)) return;
  assert.equal(rw.stats.playouts, 0, 'expected a root-win scan');
  assert.equal('stopReason' in rw.stats, false, JSON.stringify(rw.stats));
});

// ---- validation ------------------------------------------------------------

test('handle()/handleAsync() reject analyze: true with an err reply', async () => {
  const req = { id: 'x', fen: START.fen, analyze: true, nodeCeiling: 50 } as unknown as WorkerRequest;
  const syncReply = new WorkerEngine().handle(req);
  assert.ok('err' in syncReply, JSON.stringify(syncReply));
  const asyncReply = await new WorkerEngine().handleAsync(req);
  assert.ok('err' in asyncReply, JSON.stringify(asyncReply));
});

test('nodeCeiling 0, -1, 1.5 and MAX + 1 give err replies', async () => {
  for (const nodeCeiling of [0, -1, 1.5, ANALYSIS_MAX_NODE_CEILING + 1]) {
    const engine = quietEngine();
    const reply = await engine.handleFast(
      { id: `nc-${nodeCeiling}`, fen: START.fen, analyze: true, nodeCeiling, batch: 4 },
      fakeBatchEvaluator(),
    );
    assert.ok('err' in reply, `nodeCeiling ${nodeCeiling}: ${JSON.stringify(reply)}`);
    assert.match((reply as { err: string }).err, /nodeCeiling/);
    assert.equal(engine.stopTargetsSize, 0);
  }
});

test('three-way discriminant: exactly one of sims/level/analyze is accepted; zero, two or three give err', async () => {
  const makeEngine = async (): Promise<WorkerEngine> => {
    const engine = quietEngine();
    const { factory, now } = fakeProbeFactory();
    const createBatchBackend: WorkerInitEnv['createBatchBackend'] = async () => fakeBatchEvaluator();
    const init = await engine.init(
      { id: 'i', type: 'init', modelUrl: 'm.onnx', modelSha256: 'sha' },
      { sessionFactory: factory, gpu: realAdapterGpu(), isolated: true, hardwareConcurrency: 8, now, createBatchBackend },
    );
    assert.ok(!('err' in init), JSON.stringify(init));
    return engine;
  };
  const modes: Record<string, Record<string, unknown>> = {
    sims: { sims: 16, batch: 4 },
    level: { level: 'casual' },
    analyze: { analyze: true, nodeCeiling: 17, batch: 4 },
  };
  const names = Object.keys(modes);
  const subsets: string[][] = [[]];
  for (const name of names) {
    for (const s of [...subsets]) subsets.push([...s, name]);
  }
  for (const subset of subsets) {
    const engine = await makeEngine();
    // Combined subsets drop batch so a level combination is not rejected
    // for its batch first -- the discriminant must reject it on its own.
    const fields: Record<string, unknown> = {};
    for (const name of subset) Object.assign(fields, modes[name]);
    if (subset.length > 1) delete fields.batch;
    const req = { id: `d-${subset.join('+')}`, fen: START.fen, ...fields } as unknown as WorkerRequest;
    const reply = await engine.handleFast(req, engine.currentBatchEvaluator!);
    if (subset.length === 1) {
      assert.ok(isOk(reply), `${subset.join('+')}: ${JSON.stringify(reply)}`);
    } else {
      assert.ok('err' in reply, `${subset.join('+') || 'none'}: ${JSON.stringify(reply)}`);
      assert.match((reply as { err: string }).err, /exactly one of sims/);
    }
  }
});

// ---- section 3 case a: stop mid-analysis -----------------------------------

test('(a) stop mid-analysis: one reply, stopReason stop, playouts >= the last progress count, nothing after', async () => {
  const events: WorkerProgressEvent[] = [];
  let replied = false;
  let eventsAfterReply = 0;
  let stopAck: boolean | null = null;
  const engine = new WorkerEngine({
    onProgress: (event) => {
      if (replied) eventsAfterReply += 1;
      events.push(event);
      if (events.length === 2) {
        stopAck = engine.handleStop({ id: 's', type: 'stop', target: 'an' }).acknowledged;
      }
    },
  });
  const replies: WorkerReply[] = [];
  const reply = await engine
    .handleFast({ id: 'an', fen: START.fen, analyze: true, progressEveryPlayouts: 1 }, fakeBatchEvaluator())
    .then((r) => {
      replied = true;
      replies.push(r);
      return r;
    });
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(stopAck, true);
  assert.equal(replies.length, 1);
  assert.ok(isOk(reply), JSON.stringify(reply));
  if (!isOk(reply)) return;
  assert.equal(reply.stats.stopReason, 'stop');
  assert.equal(reply.stats.deadlineHit, true);
  assert.ok(events.length >= 2);
  assert.ok((reply.stats.playouts as number) >= events[events.length - 1]!.playouts);
  assert.ok((reply.stats.playouts as number) < ANALYSIS_DEFAULT_NODE_CEILING - 1);
  assert.equal(eventsAfterReply, 0);
  assert.ok(events.every((e) => e.id === 'an'));
  assert.equal(engine.stopTargetsSize, 0);
  assert.equal(engine.handleStop({ id: 's2', type: 'stop', target: 'an' }).acknowledged, false);
});

// ---- rankedRootMoves: a witness missing from root.actions keeps the order --

test('rankedRootMoves keeps rankRootMoves order when the root-win witness is not in root.actions', async () => {
  const engine = quietEngine();
  const reply = await engine.handleFast({ id: 'rk', fen: START.fen, sims: 32, batch: 4 }, fakeBatchEvaluator());
  assert.ok(isOk(reply), JSON.stringify(reply));
  const ranked = (
    engine as unknown as {
      rankedRootMoves(tree: unknown, board: unknown, n: number, witness: number | null): RootMoveStat[];
    }
  ).rankedRootMoves.bind(engine);
  const board = gameFromHistory([]).board;
  const tree = engine.tree!;
  assert.equal(tree.root!.actions.indexOf(ACTION_SIZE + 7), -1, 'precondition: witness absent');
  const plain = ranked(tree, board, 5, null);
  const missing = ranked(tree, board, 5, ACTION_SIZE + 7);
  assert.ok(plain.length > 0);
  assert.deepEqual(missing, plain);
});
