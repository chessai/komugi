// KOM-1102 item 2 (design 5.1's ponder-start path, resumed instead of
// duplicated): "ponder now" -- resuming pondering from the standing tree a
// gated reply left behind, once the page's gating flags clear. Fake
// evaluator throughout, same convention as ponder.test.ts.

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { decodeAction } from '../src/actions.ts';
import type { MlhBatchEvaluator } from '../src/batch_evaluator.ts';
import { ACTION_SIZE } from '../src/constants.ts';
import { makeFakeMlhEvaluator } from '../src/fake_evaluator.ts';
import { boardToFen } from '../src/fen.ts';
import { predictReplies } from '../src/ponder.ts';
import type { WorkerPonderNowRequest, WorkerReply, WorkerRequest } from '../src/protocol.ts';
import { gameFromHistory } from '../src/root.ts';
import { buildSan, sanRecords } from '../src/san.ts';
import type { MatchNode } from '../src/search/node.ts';
import { WorkerEngine } from '../src/engine.ts';

function isOk(reply: WorkerReply): reply is WorkerReply & { action: number; san: string; stats: Record<string, unknown> } {
  return !('err' in reply);
}

function statsOf(reply: WorkerReply): Record<string, unknown> {
  const stats = { ...(reply as { stats: Record<string, unknown> }).stats };
  delete stats.ms;
  delete stats.phases;
  delete stats.pondered;
  return stats;
}

type BiasPlainEntry = [number, number[], number, number];

function biasPlain(table: Map<number, { errSum: Float64Array; weightSum: number; gen: number }>): BiasPlainEntry[] {
  return Array.from(table.entries())
    .sort(([a], [b]) => a - b)
    .map(([key, entry]) => [key, Array.from(entry.errSum), entry.weightSum, entry.gen]);
}

function fakeBatchEvaluator(): MlhBatchEvaluator {
  const leaf = makeFakeMlhEvaluator(ACTION_SIZE);
  return (states) => states.map((state) => leaf(state));
}

// A gate-controllable evaluator: `close()` makes the NEXT call pause until
// `open()` resolves it -- used to hold a resumed ponder's first evaluator
// call in flight so a real request's abort can be observed landing on it,
// rather than racing a fake evaluator that always resolves instantly.
function gatedBatchEvaluator(): { evaluator: MlhBatchEvaluator; close: () => void; open: () => void } {
  const leaf = makeFakeMlhEvaluator(ACTION_SIZE);
  let gate: Promise<void> | null = null;
  let release: (() => void) | null = null;
  return {
    evaluator: async (states) => {
      if (gate !== null) await gate;
      return states.map((state) => leaf(state));
    },
    close: () => {
      gate = new Promise((resolve) => {
        release = resolve;
      });
    },
    open: () => {
      const r = release;
      gate = null;
      release = null;
      if (r !== null) r();
    },
  };
}

// A short real, legal history from the advanced start (ponder.test.ts's own
// convention).
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

// What the page would know as "the standing position" right after an
// engine reply -- the same computation `snapshotStanding` (worker.ts) does
// internally, used here only to build a correct `ponderNow` request.
function standingAfter(history: readonly string[], engineSan: string): { fen: string; history: string[] } {
  const nextHistory = [...history, engineSan];
  return { fen: boardToFen(gameFromHistory(nextHistory).board), history: nextHistory };
}

function topPredictionSan(root: MatchNode, afterEngineMoveBoard: ReturnType<typeof gameFromHistory>['board'], chosenAction: number): string {
  const [predicted] = predictReplies(root, chosenAction, 1);
  assert.ok(predicted !== undefined, 'expected at least one predicted reply under the chosen child');
  return buildSan(afterEngineMoveBoard, decodeAction(afterEngineMoveBoard, predicted).struct);
}

function ponderNowRequest(id: number | string, standing: { fen: string; history: string[] }): WorkerPonderNowRequest {
  return { id, type: 'ponderNow', fen: standing.fen, history: standing.history };
}

test('gated reply -> flags clear -> ponder-now -> a matching human move is a hit equal to the unpondered reply (bias table identical)', async () => {
  const engine = new WorkerEngine();
  engine.handlePonderSettings({ id: 's1', type: 'ponderSettings', hidden: true });

  const req1 = baseRequest(1);
  const reply1 = await engine.handleFast(req1, fakeBatchEvaluator());
  assert.ok(isOk(reply1), JSON.stringify(reply1));
  if (!isOk(reply1)) return;

  // Gated: no pondering actually started yet.
  await engine.ponderingPromise();
  assert.equal(engine.ponderCacheSize, 0);

  engine.handlePonderSettings({ id: 's2', type: 'ponderSettings', hidden: false });

  const standing = standingAfter(BASE.history, reply1.san);
  const resumed = engine.handlePonderNow(ponderNowRequest('pn1', standing));
  assert.equal(resumed.started, true, JSON.stringify(resumed));

  await engine.ponderingPromise();
  assert.ok(engine.ponderCacheSize > 0, 'expected the resumed ponder to have cached a prediction');

  const afterEngineMove = gameFromHistory(standing.history).board;
  const root = engine.tree!.root!;
  const replySan = topPredictionSan(root, afterEngineMove, reply1.action);
  const req2 = nextRequest(2, BASE.history, reply1.san, replySan);

  const reply2 = await engine.handleFast(req2, fakeBatchEvaluator());
  assert.ok(isOk(reply2), JSON.stringify(reply2));
  if (!isOk(reply2)) return;
  assert.equal(reply2.stats.pondered, true);

  // N-6 baseline: a fresh engine, pondering disabled, processing the SAME
  // two requests for real.
  const plain = new WorkerEngine();
  plain.handlePonderSettings({ id: 's3', type: 'ponderSettings', enabled: false });
  await plain.handleFast(req1, fakeBatchEvaluator());
  const plainReply2 = await plain.handleFast(req2, fakeBatchEvaluator());
  assert.ok(isOk(plainReply2), JSON.stringify(plainReply2));
  if (!isOk(plainReply2)) return;

  assert.equal(reply2.action, plainReply2.action);
  assert.equal(reply2.san, plainReply2.san);
  assert.deepEqual(statsOf(reply2), statsOf(plainReply2));
  assert.deepEqual(biasPlain(engine.tree!.bias.entries), biasPlain(plain.tree!.bias.entries));
});

test('ponder-now after the history changed is a no-op with a reason', async () => {
  const engine = new WorkerEngine();
  engine.handlePonderSettings({ id: 's1', type: 'ponderSettings', hidden: true });

  const req1 = baseRequest(1);
  const reply1 = await engine.handleFast(req1, fakeBatchEvaluator());
  assert.ok(isOk(reply1), JSON.stringify(reply1));
  if (!isOk(reply1)) return;

  engine.handlePonderSettings({ id: 's2', type: 'ponderSettings', hidden: false });

  const mismatched = { fen: 'not-the-real-fen', history: [...BASE.history, reply1.san] };
  const resumed = engine.handlePonderNow(ponderNowRequest('pn1', mismatched));
  assert.equal(resumed.started, false);
  assert.equal(resumed.reason, 'history_changed');

  await engine.ponderingPromise();
  assert.equal(engine.ponderCacheSize, 0);
});

test('ponder-now while still gated is a no-op', async () => {
  const engine = new WorkerEngine();
  engine.handlePonderSettings({ id: 's1', type: 'ponderSettings', hidden: true });

  const req1 = baseRequest(1);
  const reply1 = await engine.handleFast(req1, fakeBatchEvaluator());
  assert.ok(isOk(reply1), JSON.stringify(reply1));
  if (!isOk(reply1)) return;

  const standing = standingAfter(BASE.history, reply1.san);
  const resumed = engine.handlePonderNow(ponderNowRequest('pn1', standing));
  assert.equal(resumed.started, false);
  assert.equal(resumed.reason, 'gated');

  await engine.ponderingPromise();
  assert.equal(engine.ponderCacheSize, 0);
});

test('ponder-now twice does not start two ponders', async () => {
  const engine = new WorkerEngine();
  engine.handlePonderSettings({ id: 's1', type: 'ponderSettings', hidden: true });

  const req1 = baseRequest(1);
  const reply1 = await engine.handleFast(req1, fakeBatchEvaluator());
  assert.ok(isOk(reply1), JSON.stringify(reply1));
  if (!isOk(reply1)) return;

  engine.handlePonderSettings({ id: 's2', type: 'ponderSettings', hidden: false });

  const standing = standingAfter(BASE.history, reply1.san);
  const first = engine.handlePonderNow(ponderNowRequest('pn1', standing));
  assert.equal(first.started, true, JSON.stringify(first));

  const second = engine.handlePonderNow(ponderNowRequest('pn2', standing));
  assert.equal(second.started, false);
  assert.equal(second.reason, 'already_pondering');

  await engine.ponderingPromise();
  assert.equal(engine.ponderCacheSize, 1, 'exactly one ponder session should have run, not two');
});

test('a real request during a resumed ponder aborts it with no trace', async () => {
  const engine = new WorkerEngine();
  engine.handlePonderSettings({ id: 's1', type: 'ponderSettings', hidden: true });

  const gated = gatedBatchEvaluator();
  const req1 = baseRequest(1);
  const reply1 = await engine.handleFast(req1, gated.evaluator);
  assert.ok(isOk(reply1), JSON.stringify(reply1));
  if (!isOk(reply1)) return;

  engine.handlePonderSettings({ id: 's2', type: 'ponderSettings', hidden: false });

  // Hold the resumed ponder's own evaluator calls in flight.
  gated.close();
  const standing = standingAfter(BASE.history, reply1.san);
  const resumed = engine.handlePonderNow(ponderNowRequest('pn1', standing));
  assert.equal(resumed.started, true, JSON.stringify(resumed));

  // A real request arrives while the resumed ponder's batch is still
  // pending on the gate -- it must abort the ponder first (synchronously,
  // via handleFast's own abortPonder() call). Fen-only (no `history`), so
  // this request's OWN reply never starts a fresh ponder session itself
  // (worker.ts only calls maybeStartPonder when `history` is an array) --
  // isolating the cache-size check below to the EARLIER, resumed session.
  const req2: WorkerRequest = { id: 2, fen: BASE.fen, sims: 16, batch: 4 };
  const req2Promise = engine.handleFast(req2, fakeBatchEvaluator());

  // Release the held ponder batch so its awaited promise can settle and
  // observe the abort.
  gated.open();

  const reply2 = await req2Promise;
  assert.ok(isOk(reply2), JSON.stringify(reply2));

  await engine.ponderingPromise();
  assert.equal(engine.ponderCacheSize, 0, 'an aborted resumed ponder must leave no cache trace');
});
