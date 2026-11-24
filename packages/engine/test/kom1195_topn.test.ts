// KOM-1195 unit U4b (docs/design/kom1195-analysis-api.md section 4/5, section
// 9 U4b): progress emission (the `onProgress` constructor option,
// `attemptFast`'s cadence wrapper, `emitProgress`), top-N root moves
// (`rankedRootMoves` from `finish()`), and the shared `topN`/
// `progressEveryPlayouts`/`progressEveryMs` validation (AC3). Fake evaluator
// throughout, same convention as kom1195_stop.test.ts -- real-net parity is a
// later unit's job.

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { decodeAction } from '../src/actions.ts';
import type { MlhBatchEvaluator } from '../src/batch_evaluator.ts';
import { ACTION_SIZE, ADVANCED_POSITION } from '../src/constants.ts';
import { WorkerEngine } from '../src/engine.ts';
import { makeFakeMlhEvaluator } from '../src/fake_evaluator.ts';
import { boardToFen } from '../src/fen.ts';
import { predictReplies } from '../src/ponder.ts';
import type { WorkerProgressEvent, WorkerReply, WorkerRequest } from '../src/protocol.ts';
import { gameFromHistory } from '../src/root.ts';
import { buildSan, sanRecords } from '../src/san.ts';
import type { MatchTactical } from '../src/search/search.ts';
import { lcbEligible, selectAction } from '../src/search/select.ts';
import { loadFixture } from './fixture.ts';

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

const MARSHAL_CAPTURE_FEN = '8m/8D/9/9/9/9/9/9/4M4 -/- w 1 - 1';

// ---- order against selectAction directly (discovery off) -----------------

test('topMoves order matches selectAction directly (discovery off), over parity positions', async () => {
  const positions = loadFixture()
    .filter((rec) => !rec.terminal)
    .slice(0, 8);
  assert.ok(positions.length > 0, 'no non-terminal parity positions');
  for (const pos of positions) {
    const engine = new WorkerEngine();
    engine.handlePonderSettings({ id: 'p', type: 'ponderSettings', enabled: false });
    const request: WorkerRequest = {
      id: `order-${pos.id}`,
      fen: pos.fen,
      sims: 96,
      batch: 1,
      topN: 1000,
      config: { discovery: false },
    };
    const reply = await engine.handleFast(request, fakeBatchEvaluator());
    assert.ok(isOk(reply), `${pos.id}: ${JSON.stringify(reply)}`);
    if (!isOk(reply)) continue;
    const root = engine.tree!.root!;
    const config = engine.tree!.config;
    const bias = engine.tree!.bias;
    const want = selectAction(root, config, { bias });
    const topMoves = (reply as unknown as { topMoves: Array<{ action: number; visits: number }> }).topMoves;
    assert.ok(Array.isArray(topMoves) && topMoves.length > 0, `${pos.id}: expected topMoves`);
    assert.equal(topMoves[0]!.action, root.actions[want]!, `${pos.id}: entry 0 must equal selectAction's own choice`);
    if (topMoves.length >= root.width) {
      const sum = topMoves.reduce((acc, m) => acc + m.visits, 0);
      assert.equal(sum, reply.stats.playouts as number, `${pos.id}: visits must sum to playouts at topN >= width`);
    }
  }
});

// ---- the survey-free corollary: topMoves[0] === reply.action when no -------
// ---- overriding/witnessing tactical fired ----------------------------------

test('topMoves[0].action equals reply.action exactly when tactical is null or (overridden, rootWitness both null)', async () => {
  const positions = loadFixture()
    .filter((rec) => !rec.terminal)
    .slice(0, 8);
  for (const pos of positions) {
    const engine = new WorkerEngine();
    engine.handlePonderSettings({ id: 'p', type: 'ponderSettings', enabled: false });
    const request: WorkerRequest = { id: `corollary-${pos.id}`, fen: pos.fen, sims: 96, batch: 8, topN: 1000 };
    const reply = await engine.handleFast(request, fakeBatchEvaluator());
    assert.ok(isOk(reply), `${pos.id}: ${JSON.stringify(reply)}`);
    if (!isOk(reply)) continue;
    const tactical = reply.stats.tactical as MatchTactical | null;
    const topMoves = (reply as unknown as { topMoves: Array<{ action: number }> }).topMoves;
    const expectEqual = tactical === null || (tactical.overridden === null && tactical.rootWitness === null);
    if (expectEqual) {
      assert.equal(topMoves[0]!.action, reply.action, `${pos.id}: expected topMoves[0] to equal reply.action`);
    }
  }
});

// ---- root-win case, built inline (N6/F9) -----------------------------------

test('root-win scan: topN:3 puts the witness first, descriptor stays last, lcbScore null iff ineligible', async () => {
  const engine = new WorkerEngine();
  const request: WorkerRequest = { id: 'rw', fen: MARSHAL_CAPTURE_FEN, sims: 16, batch: 4, topN: 3 };
  const reply = await engine.handleFast(request, fakeBatchEvaluator());
  assert.ok(isOk(reply), JSON.stringify(reply));
  if (!isOk(reply)) return;

  const tactical = reply.stats.tactical as MatchTactical | null;
  assert.ok(tactical !== null && tactical.rootWitness !== null, 'expected a root-win witness (precondition)');
  if (tactical === null || tactical.rootWitness === null) return;

  const topMoves = (reply as unknown as { topMoves: Array<{ action: number; visits: number; lcbEligible: boolean; lcbScore: number | null }> }).topMoves;
  assert.ok(Array.isArray(topMoves) && topMoves.length > 0);
  assert.equal(topMoves[0]!.action, tactical.rootWitness, 'expected the witness listed first');

  const keys = Object.keys(reply as unknown as Record<string, unknown>);
  assert.equal(keys[keys.length - 1], 'descriptor', 'descriptor must stay the reply`s last key');

  const root = engine.tree!.root!;
  if (topMoves.length >= root.width) {
    const sum = topMoves.reduce((acc, m) => acc + m.visits, 0);
    assert.equal(sum, reply.stats.playouts as number);
  }
  for (const m of topMoves) {
    assert.equal(m.lcbScore === null, !m.lcbEligible, 'lcbScore must be null exactly when !lcbEligible');
  }
});

// ---- a standalone lcbScore-null-iff-ineligible check over real searches ---

test('lcbScore is null exactly when !lcbEligible, over parity positions at topN = width', async () => {
  const positions = loadFixture()
    .filter((rec) => !rec.terminal)
    .slice(0, 8);
  for (const pos of positions) {
    const engine = new WorkerEngine();
    engine.handlePonderSettings({ id: 'p', type: 'ponderSettings', enabled: false });
    const reply = await engine.handleFast(
      { id: `lcb-${pos.id}`, fen: pos.fen, sims: 32, batch: 4, topN: 1000 },
      fakeBatchEvaluator(),
    );
    assert.ok(isOk(reply), `${pos.id}: ${JSON.stringify(reply)}`);
    if (!isOk(reply)) continue;
    const root = engine.tree!.root!;
    const eligible = lcbEligible(root, engine.tree!.config);
    const topMoves = (reply as unknown as { topMoves: Array<{ action: number; lcbEligible: boolean; lcbScore: number | null }> }).topMoves;
    for (const m of topMoves) {
      const idx = root.actions.indexOf(m.action);
      assert.equal(m.lcbEligible, eligible[idx]!, `${pos.id}: lcbEligible mismatch`);
      assert.equal(m.lcbScore === null, !eligible[idx]!, `${pos.id}: lcbScore must be null exactly when !lcbEligible`);
    }
  }
});

// ---- reply with no topN gets no topMoves field, un-opted stays untouched --

test('a reply with no topN (or topN: 0) carries no topMoves field', async () => {
  for (const topN of [undefined, 0]) {
    const engine = new WorkerEngine();
    const request: WorkerRequest = { id: `none-${String(topN)}`, fen: ADVANCED_POSITION, sims: 16, batch: 4 };
    if (topN !== undefined) request.topN = topN;
    const reply = await engine.handleFast(request, fakeBatchEvaluator());
    assert.ok(isOk(reply), JSON.stringify(reply));
    if (!isOk(reply)) continue;
    assert.equal('topMoves' in reply, false, JSON.stringify(reply));
  }
});

// ---- validation: topN/progressEvery* on every entry point reaching -------
// ---- finish() (retry BLOCKER) ----------------------------------------------

test('invalid topN/progressEvery* give err replies on handleFast, never a truncated slice', async () => {
  const cases: Array<Partial<WorkerRequest>> = [
    { topN: -1 },
    { topN: 1.5 },
    { progressEveryPlayouts: 0 },
    { progressEveryMs: -1 },
  ];
  for (const overrides of cases) {
    const engine = new WorkerEngine();
    const reply = await engine.handleFast(
      { id: `bad-${JSON.stringify(overrides)}`, fen: ADVANCED_POSITION, sims: 16, batch: 4, ...overrides },
      fakeBatchEvaluator(),
    );
    assert.ok('err' in reply, `expected an err reply for ${JSON.stringify(overrides)}, got ${JSON.stringify(reply)}`);
  }
});

test('invalid topN gives err replies on exact-mode handle()/handleAsync() too', async () => {
  for (const topN of [-1, 1.5]) {
    const engine = new WorkerEngine();
    const syncReply = engine.handle({ id: `sync-${topN}`, fen: ADVANCED_POSITION, sims: 16, topN });
    assert.ok('err' in syncReply, `handle(): expected err for topN ${topN}, got ${JSON.stringify(syncReply)}`);

    const asyncEngine = new WorkerEngine();
    const asyncReply = await asyncEngine.handleAsync({ id: `async-${topN}`, fen: ADVANCED_POSITION, sims: 16, topN });
    assert.ok('err' in asyncReply, `handleAsync(): expected err for topN ${topN}, got ${JSON.stringify(asyncReply)}`);
  }
});

// ---- REQUIRED retry test: validation runs BEFORE the ponder cache lookup --
// ---- (the ponder key folds in neither topN nor progressEvery*) ------------

test('an invalid topN/progressEvery* value on a request matching a live ponder entry still errs (never a ponder-cache hit)', async () => {
  const engine = new WorkerEngine();
  const base = buildHistory(2);
  const req1: WorkerRequest = { id: 1, fen: base.fen, history: base.history, sims: 16, batch: 4 };
  const reply1 = await engine.handleFast(req1, fakeBatchEvaluator());
  assert.ok(isOk(reply1), JSON.stringify(reply1));
  if (!isOk(reply1)) return;

  await engine.ponderingPromise();
  assert.ok(engine.ponderCacheSize > 0, 'expected a ponder entry to have been cached');

  const afterEngineMove = gameFromHistory([...base.history, reply1.san]).board;
  const [predicted] = predictReplies(engine.tree!.root!, reply1.action, 1);
  assert.ok(predicted !== undefined, 'expected a predicted reply under the chosen child');
  const replySan = buildSan(afterEngineMove, decodeAction(afterEngineMove, predicted!).struct);

  const invalidCases: Array<Partial<WorkerRequest>> = [
    { topN: -1 },
    { topN: 1.5 },
    { progressEveryPlayouts: 0 },
    { progressEveryMs: -1 },
  ];
  for (const overrides of invalidCases) {
    const req2 = nextRequest(`bad-${JSON.stringify(overrides)}`, base.history, reply1.san, replySan, overrides);
    const reply2 = await engine.handleFast(req2, fakeBatchEvaluator());
    assert.ok(
      'err' in reply2,
      `expected an err reply for ${JSON.stringify(overrides)} against a live ponder entry, got ${JSON.stringify(reply2)}`,
    );
  }
});

// ---- AC3's determinism check: both opted in, byte-identical ---------------

test('determinism: progressEveryPlayouts:1 vs stoppable-only (both opted in) give byte-identical replies', async () => {
  const positions = loadFixture()
    .filter((rec) => !rec.terminal)
    .slice(0, 3);
  for (const pos of positions) {
    for (const batch of [1, 8]) {
      const withProgress: WorkerRequest = {
        id: `det-p-${pos.id}-${batch}`,
        fen: pos.fen,
        sims: 32,
        batch,
        progressEveryPlayouts: 1,
      };
      const withStop: WorkerRequest = { id: `det-s-${pos.id}-${batch}`, fen: pos.fen, sims: 32, batch, stoppable: true };
      const e1 = new WorkerEngine({ onProgress: () => {} });
      e1.handlePonderSettings({ id: 'p', type: 'ponderSettings', enabled: false });
      const r1 = await e1.handleFast(withProgress, fakeBatchEvaluator());
      const e2 = new WorkerEngine();
      e2.handlePonderSettings({ id: 'p', type: 'ponderSettings', enabled: false });
      const r2 = await e2.handleFast(withStop, fakeBatchEvaluator());
      assert.ok(isOk(r1) && isOk(r2), `${pos.id} batch ${batch}: ${JSON.stringify({ r1, r2 })}`);
      if (!isOk(r1) || !isOk(r2)) continue;
      assert.equal(r1.action, r2.action, `${pos.id} batch ${batch}`);
      assert.equal(r1.san, r2.san, `${pos.id} batch ${batch}`);
      assert.deepEqual(statsOf(r1), statsOf(r2), `${pos.id} batch ${batch}`);
    }
  }
});

// ---- progress emission: the onProgress constructor option -----------------

test('onProgress fires on progressEveryPlayouts cadence with a well-formed WorkerProgressEvent (pv, topMoves)', async () => {
  const events: WorkerProgressEvent[] = [];
  const engine = new WorkerEngine({ onProgress: (e) => events.push(e) });
  engine.handlePonderSettings({ id: 'p', type: 'ponderSettings', enabled: false });
  const request: WorkerRequest = {
    id: 'prog',
    fen: ADVANCED_POSITION,
    sims: 64,
    batch: 1,
    progressEveryPlayouts: 4,
    topN: 3,
  };
  const reply = await engine.handleFast(request, fakeBatchEvaluator());
  assert.ok(isOk(reply), JSON.stringify(reply));
  if (!isOk(reply)) return;

  assert.ok(events.length > 0, 'expected at least one progress event');
  let last = 0;
  for (const event of events) {
    assert.equal(event.type, 'progress');
    assert.equal(event.id, 'prog');
    assert.ok(event.playouts > last, `playouts must strictly increase: ${last} -> ${event.playouts}`);
    last = event.playouts;
    assert.equal(typeof event.bestSan, 'string');
    assert.equal(typeof event.bestAction, 'number');
    assert.equal(event.rootWdl.length, 3);
    assert.ok(Array.isArray(event.pv) && event.pv.length >= 1 && event.pv.length <= 8);
    assert.equal(event.pv[0], event.bestSan, 'pv[0] must equal the best move at the root');
    assert.ok(event.topMoves !== undefined && event.topMoves.length <= 3);
  }
  assert.ok(last <= (reply.stats.playouts as number));
});

test('onProgress fires on progressEveryMs cadence (fake clock), and never when no progressEvery* field is set', async () => {
  let now = 0;
  const events: WorkerProgressEvent[] = [];
  const engine = new WorkerEngine({ clock: () => now, onProgress: (e) => events.push(e) });
  engine.handlePonderSettings({ id: 'p', type: 'ponderSettings', enabled: false });
  let calls = 0;
  const leaf = makeFakeMlhEvaluator(ACTION_SIZE);
  const evaluator: MlhBatchEvaluator = (states) => {
    calls += 1;
    now += 10;
    return states.map((state) => leaf(state));
  };
  const request: WorkerRequest = { id: 'progms', fen: ADVANCED_POSITION, sims: 64, batch: 1, progressEveryMs: 15 };
  const reply = await engine.handleFast(request, evaluator);
  assert.ok(isOk(reply), JSON.stringify(reply));
  assert.ok(events.length > 0, 'expected at least one progress event from the ms cadence');
  assert.ok(calls > 0);

  const events2: WorkerProgressEvent[] = [];
  const engine2 = new WorkerEngine({ onProgress: (e) => events2.push(e) });
  const reply2 = await engine2.handleFast(
    { id: 'noprog', fen: ADVANCED_POSITION, sims: 64, batch: 1, stoppable: true },
    fakeBatchEvaluator(),
  );
  assert.ok(isOk(reply2), JSON.stringify(reply2));
  assert.equal(events2.length, 0, 'no progressEvery* field must mean no progress events at all');
});
