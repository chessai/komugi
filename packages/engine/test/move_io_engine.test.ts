// KOM-1198 unit 4 (docs/design/kom1198-move-io.md section 5/6/7): engine
// wiring for the reply descriptor fields and descriptor-history ponder
// continuity. Fake evaluator throughout, same convention as
// ponder.test.ts/ponder_predicted.test.ts.

import assert from 'node:assert/strict';
import { test } from 'node:test';

import type { MlhBatchEvaluator } from '../src/batch_evaluator.ts';
import { ACTION_SIZE } from '../src/constants.ts';
import { WorkerEngine } from '../src/engine.ts';
import { makeFakeMlhEvaluator } from '../src/fake_evaluator.ts';
import { boardFromFen, boardToFen } from '../src/fen.ts';
import { resolveMoveDescriptor } from '../src/move_io.ts';
import type { WorkerPonderNowRequest, WorkerReply, WorkerRequest } from '../src/protocol.ts';
import { gameFromHistory } from '../src/root.ts';
import { sanRecords } from '../src/san.ts';

function isOk(
  reply: WorkerReply,
): reply is WorkerReply & { action: number; san: string; stats: Record<string, unknown> } {
  return !('err' in reply);
}

function fakeBatchEvaluator(): MlhBatchEvaluator {
  const leaf = makeFakeMlhEvaluator(ACTION_SIZE);
  return (states) => states.map((state) => leaf(state));
}

// Same convention as ponder.test.ts's own buildHistory: the current
// position's first enumerated legal move, every ply.
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

test('reply.descriptor resolves back to reply.action, and predictedReply.descriptor resolves back to predictedReply.action on the post-move board', async () => {
  const engine = new WorkerEngine();
  engine.handlePonderSettings({ id: 's', type: 'ponderSettings', enabled: false });
  const reply = await engine.handleFast(baseRequest(1), fakeBatchEvaluator());
  assert.ok(isOk(reply), JSON.stringify(reply));
  if (!isOk(reply)) return;

  const rootBoard = boardFromFen(BASE.fen);
  assert.equal(resolveMoveDescriptor(rootBoard, reply.descriptor), reply.action);

  assert.ok(reply.predictedReply !== undefined, 'expected a predictedReply with pondering off');
  const afterEngineMove = gameFromHistory([...BASE.history, reply.san]).board;
  assert.equal(
    resolveMoveDescriptor(afterEngineMove, reply.predictedReply.descriptor),
    reply.predictedReply.action,
  );
});

test('descriptor is the last key of the reply when predictedReply is present, and predictedReply.descriptor is the last key of predictedReply', async () => {
  const engine = new WorkerEngine();
  engine.handlePonderSettings({ id: 's', type: 'ponderSettings', enabled: false });
  const reply = await engine.handleFast(baseRequest(1), fakeBatchEvaluator());
  assert.ok(isOk(reply), JSON.stringify(reply));
  if (!isOk(reply)) return;

  assert.ok(reply.predictedReply !== undefined);
  assert.equal(Object.keys(reply).at(-1), 'descriptor');
  assert.equal(Object.keys(reply.predictedReply).at(-1), 'descriptor');
});

test('ponder continuity with a descriptor history: the second request extends the first by reply.descriptor plus reply.predictedReply.descriptor', async () => {
  const engine = new WorkerEngine();
  const req1 = baseRequest(1);
  const reply1 = await engine.handleFast(req1, fakeBatchEvaluator());
  assert.ok(isOk(reply1), JSON.stringify(reply1));
  if (!isOk(reply1)) return;
  assert.ok(reply1.predictedReply !== undefined, 'expected a predicted reply to extend the history with');

  await engine.ponderingPromise();
  assert.ok(engine.ponderCacheSize > 0, 'expected at least one cached predicted reply');

  const nextHistory = [...BASE.history, reply1.descriptor, reply1.predictedReply.descriptor];
  const nextFen = boardToFen(gameFromHistory(nextHistory).board);
  const req2: WorkerRequest = { id: 2, fen: nextFen, history: nextHistory, sims: 16, batch: 4 };

  const reply2 = await engine.handleFast(req2, fakeBatchEvaluator());
  assert.ok(isOk(reply2), JSON.stringify(reply2));
  if (!isOk(reply2)) return;
  assert.equal(reply2.stats.pondered, true);
});

test('ponderNow with a descriptor history resumes the standing ponder', async () => {
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

  const standingHistory = [...BASE.history, reply1.descriptor];
  const standingFen = boardToFen(gameFromHistory(standingHistory).board);
  const ponderNowRequest: WorkerPonderNowRequest = {
    id: 'pn1',
    type: 'ponderNow',
    fen: standingFen,
    history: standingHistory,
  };
  const resumed = engine.handlePonderNow(ponderNowRequest);
  assert.equal(resumed.started, true, JSON.stringify(resumed));

  await engine.ponderingPromise();
  assert.ok(engine.ponderCacheSize > 0, 'expected the resumed ponder to have cached a prediction');
});
