// KOM-1102 item 1: the fast reply's own `predictedReply` field -- the top
// predicted human reply to the engine's move, read off the standing tree
// the same way pondering does (ponder.ts's `predictReplies`), independent of
// whether pondering itself is enabled or gated. Fake evaluator throughout,
// same convention as ponder.test.ts/worker_fast.test.ts.

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { decodeAction } from '../src/actions.ts';
import type { MlhBatchEvaluator } from '../src/batch_evaluator.ts';
import { ACTION_SIZE } from '../src/constants.ts';
import { makeFakeMlhEvaluator } from '../src/fake_evaluator.ts';
import { boardToFen } from '../src/fen.ts';
import { describeRecord } from '../src/move_io.ts';
import { predictReplies } from '../src/ponder.ts';
import type { WorkerReply, WorkerRequest } from '../src/protocol.ts';
import { gameFromHistory } from '../src/root.ts';
import { buildSan, sanRecords } from '../src/san.ts';
import { WorkerEngine } from '../src/engine.ts';

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

// A sparse hand-built position (false_mate.test.ts's own convention for
// minimal FENs) where White's only sensible try is a soldier capturing
// Black's undefended marshal -- `search.ts`'s own tactical `rootWin` shortcut
// (design: a proven win is "played without spending a playout on deciding
// whether to") returns that action with ZERO playouts and no descent, so the
// chosen action's own child is never created in `root.children` -- exactly
// the "chosen child has no visits" case this field must report as absent.
// Verified directly against WorkerEngine: `stats.playouts === 0` and
// `tactical.rootWitness` equals the chosen action.
const MARSHAL_CAPTURE_FEN = '8m/8D/9/9/9/9/9/9/4M4 -/- w 1 - 1';

test('predictedReply equals predictReplies\' own top entry under the chosen child, present with pondering off, and no extra evaluator calls', async () => {
  const engine = new WorkerEngine();
  engine.handlePonderSettings({ id: 's', type: 'ponderSettings', enabled: false });
  const reply = await engine.handleFast(baseRequest(1), fakeBatchEvaluator());
  assert.ok(isOk(reply), JSON.stringify(reply));
  if (!isOk(reply)) return;

  assert.ok(engine.ponderCacheSize === 0, 'pondering was disabled; nothing should have been cached');
  assert.ok(reply.predictedReply !== undefined, 'expected a predictedReply with pondering off');

  const root = engine.tree!.root!;
  const [expectedAction] = predictReplies(root, reply.action, 1);
  assert.ok(expectedAction !== undefined);
  const afterEngineMove = gameFromHistory([...BASE.history, reply.san]).board;
  const expectedStruct = decodeAction(afterEngineMove, expectedAction).struct;
  const expectedSan = buildSan(afterEngineMove, expectedStruct);
  const expectedDescriptor = describeRecord(afterEngineMove, expectedStruct);

  assert.deepEqual(reply.predictedReply, { action: expectedAction, san: expectedSan, descriptor: expectedDescriptor });

  // Zero extra evaluator calls: an engine that never computes a
  // predictedReply (stubbed out below) must reach the exact same evaluation
  // count, and every other field of the reply must be identical -- the only
  // difference between the two is the presence of `predictedReply` itself.
  const plain = new WorkerEngine();
  plain.handlePonderSettings({ id: 's', type: 'ponderSettings', enabled: false });
  (plain as unknown as { predictReply: () => null }).predictReply = () => null;
  const plainReply = await plain.handleFast(baseRequest(1), fakeBatchEvaluator());
  assert.ok(isOk(plainReply), JSON.stringify(plainReply));
  if (!isOk(plainReply)) return;

  assert.equal((plainReply as { predictedReply?: unknown }).predictedReply, undefined);
  // `stats.ms` is wall-clock time, different on every run: it is the one
  // field excluded from the comparison (both sides), never the evaluation
  // counts or anything the search computed.
  const withoutField = { ...(reply as unknown as Record<string, unknown>) };
  delete withoutField.predictedReply;
  const strip = (r: Record<string, unknown>): Record<string, unknown> => {
    const stats = { ...(r.stats as Record<string, unknown>) };
    delete stats.ms;
    return { ...r, stats };
  };
  assert.deepEqual(strip(withoutField), strip(plainReply as unknown as Record<string, unknown>));
});

test('predictedReply is absent when the chosen action is a zero-playout tactical win (the chosen child is never created)', async () => {
  const engine = new WorkerEngine();
  const request: WorkerRequest = { id: 1, fen: MARSHAL_CAPTURE_FEN, sims: 16, batch: 4 };
  const reply = await engine.handleFast(request, fakeBatchEvaluator());
  assert.ok(isOk(reply), JSON.stringify(reply));
  if (!isOk(reply)) return;

  assert.equal(reply.stats.playouts, 0, 'expected the tactical rootWin shortcut (zero playouts)');
  assert.equal(
    (reply.stats as { tactical: { rootWitness: number | null } }).tactical.rootWitness,
    reply.action,
  );
  assert.equal((reply as { predictedReply?: unknown }).predictedReply, undefined);
});
