// KOM-1219 unit 1 (docs/design/kom1219-adaptive-time.md section 2, section
// 6 AC1, section 7 unit 1): `tryForced`'s own check, before `lookupPonder`,
// in `handleFast`'s queue callback -- a width-1 root is answered with zero
// evaluator calls, on every `handleFast` path, ponder hit or not. Fake
// evaluator throughout, same convention as kom1195_stop.test.ts.

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { ACTION_SIZE, ADVANCED_POSITION } from '../src/constants.ts';
import { WorkerEngine } from '../src/engine.ts';
import { makeFakeMlhEvaluator } from '../src/fake_evaluator.ts';
import type { MlhBatchEvaluator } from '../src/batch_evaluator.ts';
import type { WorkerAnalyzeRequest, WorkerReply, WorkerRequest } from '../src/protocol.ts';
import { rootGame } from '../src/root.ts';
import { sanRecords } from '../src/san.ts';
import { boardFromFen, boardToFen } from '../src/fen.ts';
import { Game } from '../src/game.ts';
import { RepetitionRule } from '../src/repetition.ts';
import { loadOracleFixture, positionRecords } from './oracle_helpers.ts';

// D6: the committed oracle record `pos:soak-beginner-s7/g020/p096` -- under
// `rules: 'ligungi'` this FEN has exactly 1 legal action (id 5650, verified
// against MatchGameState.legalMask directly); under the default rules the
// SAME FEN has 58 legal actions, the natural not-forced control.
const FORCED_FEN =
  'j6g1/2r|A:N|mf1|s:s|1/d2Ad3d/1r|f:G|3|I:a|2/1w1|S:W|2d2/j|D:n|7/J1F2|W:n|2n/5wM|R:N|1/S5J1F -/- w 1 - 49';

function isOk(reply: WorkerReply): reply is WorkerReply & { stats: Record<string, unknown> } {
  return !('err' in reply);
}

function fakeBatchEvaluator(): MlhBatchEvaluator {
  const leaf = makeFakeMlhEvaluator(ACTION_SIZE);
  return (states) => states.map((state) => leaf(state));
}

function fakeCountingBatchEvaluator(): { evaluator: MlhBatchEvaluator; calls: () => number } {
  const leaf = makeFakeMlhEvaluator(ACTION_SIZE);
  let calls = 0;
  return {
    evaluator: (states) => {
      calls += 1;
      return states.map((state) => leaf(state));
    },
    calls: () => calls,
  };
}

test('D6 fixture: FORCED_FEN matches pos:soak-beginner-s7/g020/p096 komugiFen', () => {
  const record = positionRecords(loadOracleFixture()).find((r) => r.id === 'pos:soak-beginner-s7/g020/p096');
  assert.ok(record !== undefined, 'fixture record not found');
  assert.equal(record.komugiFen, FORCED_FEN);
});

// ---- (a) width-1 root: zero evaluator calls -------------------------------

test('(a) a width-1 root replies forced, evaluations 0, playouts 0, nodes 0, 0 batch-evaluator calls', async () => {
  const engine = new WorkerEngine();
  const { evaluator, calls } = fakeCountingBatchEvaluator();
  const request: WorkerRequest = { id: 'a', fen: FORCED_FEN, rules: 'ligungi', sims: 4 };
  const reply = await engine.handleFast(request, evaluator);
  assert.ok(isOk(reply), JSON.stringify(reply));
  if (!isOk(reply)) return;
  assert.equal(reply.stats.forced, true);
  assert.equal(reply.stats.evaluations, 0);
  assert.equal(reply.stats.playouts, 0);
  assert.equal(reply.stats.nodes, 0);
  assert.equal(calls(), 0);
});

// ---- (b) stale-tree case ---------------------------------------------------

test('(b) a width>1 reply then a forced reply: nodes 0, forced true, and handlePonderNow sees no_prior_reply', async () => {
  const engine = new WorkerEngine();
  const evaluator = fakeBatchEvaluator();
  // A non-forced request carrying a history array: `attemptFast` calls
  // `maybeStartPonder`, which always sets `_lastStanding` (gated pondering
  // or not) -- the snapshot `tryForced` must clear on the next request.
  const first: WorkerRequest = { id: 'b1', fen: ADVANCED_POSITION, history: [], sims: 4 };
  const firstReply = await engine.handleFast(first, evaluator);
  assert.ok(isOk(firstReply), JSON.stringify(firstReply));

  const second: WorkerRequest = { id: 'b2', fen: FORCED_FEN, rules: 'ligungi', sims: 4 };
  const secondReply = await engine.handleFast(second, evaluator);
  assert.ok(isOk(secondReply), JSON.stringify(secondReply));
  if (!isOk(secondReply)) return;
  assert.equal(secondReply.stats.nodes, 0);
  assert.equal(secondReply.stats.forced, true);

  const pondered = engine.handlePonderNow({ id: 'pn', type: 'ponderNow', fen: FORCED_FEN });
  assert.equal(pondered.started, false);
  assert.equal(pondered.reason, 'no_prior_reply');
});

// ---- (c) D8: lookupPonder spy -- forced sits before the ponder lookup ----

test('(c) D8: a width-1 history request never reaches lookupPonder; the 58-move control does', async () => {
  const engine = new WorkerEngine();
  const evaluator = fakeBatchEvaluator();
  let calls = 0;
  const sentinel: WorkerReply = {
    id: 'sentinel',
    san: 'sentinel-san',
    action: 0,
    stamp: 'absent',
    stats: {
      playouts: 0,
      evaluations: 0,
      nodes: 0,
      rootWdl: [0, 0, 0],
      q: 0,
      tactical: null,
      ms: 0,
      model_sha256: null,
      sims: 0,
      draftmin: 0,
      discovery: 'on',
      mlh: '0/0/0/0',
      deadlineHit: false,
    },
  } as WorkerReply;
  (engine as unknown as { lookupPonder: (...args: unknown[]) => WorkerReply | null }).lookupPonder = (
    ..._args: unknown[]
  ) => {
    calls += 1;
    return sentinel;
  };

  const forcedRequest: WorkerRequest = { id: 'c1', fen: FORCED_FEN, rules: 'ligungi', sims: 4, history: ['e2e4'] };
  const forcedReply = await engine.handleFast(forcedRequest, evaluator);
  assert.ok(isOk(forcedReply), JSON.stringify(forcedReply));
  if (!isOk(forcedReply)) return;
  assert.equal(forcedReply.stats.forced, true);
  assert.equal(forcedReply.stats.evaluations, 0);
  assert.equal(calls, 0, 'lookupPonder must not be reached on the forced path');

  const controlRequest: WorkerRequest = { id: 'c2', fen: FORCED_FEN, sims: 4, history: ['e2e4'] };
  const controlReply = await engine.handleFast(controlRequest, evaluator);
  assert.equal(controlReply, sentinel);
  assert.equal(calls, 1, 'lookupPonder must be reached on the 58-move control');
});

// ---- (d) timing: true gives zeroed phases ---------------------------------

test('(d) timing: true gives phases with every search-phase key 0', async () => {
  const engine = new WorkerEngine();
  const evaluator = fakeBatchEvaluator();
  const request: WorkerRequest = { id: 'd', fen: FORCED_FEN, rules: 'ligungi', sims: 4, timing: true };
  const reply = await engine.handleFast(request, evaluator);
  assert.ok(isOk(reply), JSON.stringify(reply));
  if (!isOk(reply)) return;
  const phases = reply.stats.phases as Record<string, number> | undefined;
  assert.ok(phases !== undefined, 'expected stats.phases');
  for (const key of [
    'setupMs',
    'rootEvalMs',
    'rootWinMs',
    'playoutsMs',
    'selectMs',
    'surveyMs',
    'resultMs',
    'searchMs',
    'evaluateMs',
    'evaluatorCalls',
  ]) {
    assert.equal(phases![key], 0, key);
  }
});

// ---- (e) a >= 2-move reply has no forced key ------------------------------

test('(e) a >= 2-move reply carries no forced key', async () => {
  const engine = new WorkerEngine();
  const evaluator = fakeBatchEvaluator();
  const request: WorkerRequest = { id: 'e', fen: ADVANCED_POSITION, sims: 4 };
  const reply = await engine.handleFast(request, evaluator);
  assert.ok(isOk(reply), JSON.stringify(reply));
  if (!isOk(reply)) return;
  assert.ok(!('forced' in reply.stats));
});

// ---- (f) malformed fen: {id, err}, queue still resolves -------------------

test('(f) a malformed (non-string) fen is an err reply, and the next queued handleFast call still resolves', async () => {
  const engine = new WorkerEngine();
  const evaluator = fakeBatchEvaluator();
  // @ts-expect-error -- deliberately wrong shape.
  const bad: WorkerRequest = { id: 'f1', fen: 12345, sims: 4 };
  const reply = await engine.handleFast(bad, evaluator);
  assert.ok('err' in reply, JSON.stringify(reply));
  if ('err' in reply) assert.equal(reply.err, 'fen must be a string');

  const next: WorkerRequest = { id: 'f2', fen: ADVANCED_POSITION, sims: 4 };
  const nextReply = await engine.handleFast(next, evaluator);
  assert.ok(isOk(nextReply), JSON.stringify(nextReply));
});

// ---- (g) malformed sims: width-1 err equals the 58-move control's --------

test('(g) a malformed sims at a width-1 root gives the same err as the 58-move control, for -1 and 1.5', async () => {
  const engine = new WorkerEngine();
  const evaluator = fakeBatchEvaluator();
  for (const badSims of [-1, 1.5]) {
    const forced: WorkerRequest = { id: `g1-${badSims}`, fen: FORCED_FEN, rules: 'ligungi', sims: badSims };
    const forcedReply = await engine.handleFast(forced, evaluator);
    assert.ok('err' in forcedReply, JSON.stringify(forcedReply));

    const control: WorkerRequest = { id: `g2-${badSims}`, fen: FORCED_FEN, sims: badSims };
    const controlReply = await engine.handleFast(control, evaluator);
    assert.ok('err' in controlReply, JSON.stringify(controlReply));

    if ('err' in forcedReply && 'err' in controlReply) {
      assert.equal(forcedReply.err, controlReply.err);
    }
  }
  const next: WorkerRequest = { id: 'g3', fen: ADVANCED_POSITION, sims: 4 };
  const nextReply = await engine.handleFast(next, evaluator);
  assert.ok(isOk(nextReply), JSON.stringify(nextReply));
});

// ---- (h) terminal root: err, queue still resolves -------------------------

test('(h) a history reaching the ply cap gives err: turn_cap, and the next queued call still resolves', async () => {
  const engine = new WorkerEngine();
  const evaluator = fakeBatchEvaluator();
  const game = new Game(boardFromFen(ADVANCED_POSITION), RepetitionRule.PLACEMENT);
  const history: string[] = [];
  for (let i = 0; i < 512; i++) {
    const records = sanRecords(game.board);
    assert.ok(records.length > 0, `no legal move at ply ${i}`);
    const { id, san } = records[0];
    history.push(san);
    game.play(id);
  }
  const fen = boardToFen(game.board);
  const request: WorkerRequest = { id: 'h1', fen, history, sims: 4 };
  const reply = await engine.handleFast(request, evaluator);
  assert.ok('err' in reply, JSON.stringify(reply));
  if ('err' in reply) assert.equal(reply.err, 'turn_cap');

  const next: WorkerRequest = { id: 'h2', fen: ADVANCED_POSITION, sims: 4 };
  const nextReply = await engine.handleFast(next, evaluator);
  assert.ok(isOk(nextReply), JSON.stringify(nextReply));
});

// ---- (i) D9: non-replaying history: stamp matches the base path ----------

test('(i) D9: a width-1 request with a non-replaying history gets the same stamp as the 58-move control, and the queue still resolves', async () => {
  const engine = new WorkerEngine();
  const evaluator = fakeBatchEvaluator();
  const forced: WorkerRequest = { id: 'i1', fen: FORCED_FEN, rules: 'ligungi', sims: 4, history: ['not-a-legal-san'] };
  const forcedReply = await engine.handleFast(forced, evaluator);
  assert.ok(isOk(forcedReply), JSON.stringify(forcedReply));
  if (!isOk(forcedReply)) return;
  assert.equal(forcedReply.stats.forced, true);

  const control: WorkerRequest = { id: 'i2', fen: FORCED_FEN, sims: 4, history: ['not-a-legal-san'] };
  const controlReply = await engine.handleFast(control, evaluator);
  assert.ok(isOk(controlReply), JSON.stringify(controlReply));
  if (!isOk(controlReply)) return;

  assert.equal(forcedReply.stamp, controlReply.stamp);
  assert.equal(forcedReply.stamp, 'mismatch');

  const next: WorkerRequest = { id: 'i3', fen: ADVANCED_POSITION, sims: 4 };
  const nextReply = await engine.handleFast(next, evaluator);
  assert.ok(isOk(nextReply), JSON.stringify(nextReply));
});

// ---- (j) a stoppable width-1 request still takes the forced path --------

test('(j) a stoppable: true width-1 request is forced, evaluations 0, and carries no stopReason', async () => {
  const engine = new WorkerEngine();
  const evaluator = fakeBatchEvaluator();
  const request: WorkerRequest = { id: 'j', fen: FORCED_FEN, rules: 'ligungi', sims: 4, stoppable: true };
  const reply = await engine.handleFast(request, evaluator);
  assert.ok(isOk(reply), JSON.stringify(reply));
  if (!isOk(reply)) return;
  assert.equal(reply.stats.forced, true);
  assert.equal(reply.stats.evaluations, 0);
  assert.ok(!('stopReason' in reply.stats));
});

// ---- (k) analysis / topN > 0 never take the forced path -------------------

test('(k) an analysis request at a width-1 root does not take the forced path', async () => {
  const engine = new WorkerEngine();
  const evaluator = fakeBatchEvaluator();
  const request: WorkerAnalyzeRequest = { id: 'k1', fen: FORCED_FEN, rules: 'ligungi', analyze: true };
  const reply = await engine.handleFast(request, evaluator);
  assert.ok(isOk(reply), JSON.stringify(reply));
  if (!isOk(reply)) return;
  assert.ok(!('forced' in reply.stats));
  assert.equal(reply.stats.evaluations, 1);
});

test('(k) a topN: 1 request at a width-1 root does not take the forced path', async () => {
  const engine = new WorkerEngine();
  const evaluator = fakeBatchEvaluator();
  const request: WorkerRequest = { id: 'k2', fen: FORCED_FEN, rules: 'ligungi', sims: 4, topN: 1 };
  const reply = await engine.handleFast(request, evaluator);
  assert.ok(isOk(reply), JSON.stringify(reply));
  if (!isOk(reply)) return;
  assert.ok(!('forced' in reply.stats));
  assert.equal(reply.stats.evaluations, 1);
});

// ---- D10/D11: validateFastParams is shared, not copied --------------------

test('D10: a malformed batch or maxMs at a width-1 root gives the same err as the 58-move control', async () => {
  const engine = new WorkerEngine();
  const evaluator = fakeBatchEvaluator();
  const badFields: Array<Pick<WorkerRequest, 'batch' | 'maxMs'>> = [
    { batch: 0 },
    { batch: 1.5 },
    { maxMs: -1 },
    { maxMs: Infinity },
  ];
  for (const field of badFields) {
    const forced: WorkerRequest = { id: `d10f-${JSON.stringify(field)}`, fen: FORCED_FEN, rules: 'ligungi', sims: 4, ...field };
    const forcedReply = await engine.handleFast(forced, evaluator);
    assert.ok('err' in forcedReply, JSON.stringify(forcedReply));

    const control: WorkerRequest = { id: `d10c-${JSON.stringify(field)}`, fen: FORCED_FEN, sims: 4, ...field };
    const controlReply = await engine.handleFast(control, evaluator);
    assert.ok('err' in controlReply, JSON.stringify(controlReply));

    if ('err' in forcedReply && 'err' in controlReply) {
      assert.equal(forcedReply.err, controlReply.err, JSON.stringify(field));
    }
  }
});

// ---- D5/D13: measured cost of the extra rootGame replay -------------------

test('D5/D13: measured cost of the extra rootGame replay on a non-forced request (reported, not asserted)', () => {
  const history: string[] = [];
  const game = new Game(boardFromFen(ADVANCED_POSITION), RepetitionRule.PLACEMENT);
  for (let i = 0; i < 20; i++) {
    const records = sanRecords(game.board);
    const { id, san } = records[0]!;
    history.push(san);
    game.play(id);
  }
  const fen = boardToFen(game.board);
  const samples: number[] = [];
  const N = 25;
  for (let i = 0; i < N; i++) {
    const t0 = performance.now();
    rootGame(fen, history);
    samples.push(performance.now() - t0);
  }
  samples.sort((a, b) => a - b);
  const median = samples[Math.floor(N / 2)]!;
  // eslint-disable-next-line no-console
  console.log(`[D5/D13] extra rootGame replay (history length ${history.length}): median ${median.toFixed(4)} ms over ${N} runs`);
  assert.ok(median >= 0);
});
