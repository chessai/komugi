// KOM-1195 unit U9 (docs/design/kom1195-analysis-api.md section 9 U9,
// section 8 AC7, re-review N9): `KomugiClient` driven through a REAL
// `node:worker_threads` worker (via `test/support/node_worker_adapter.ts`'s
// `nodeWorkerFactory`, which runs `client_test_worker.ts`'s `installWorkerHost`
// on the fake MLH evaluator -- no engine/worker_host fake, unlike
// kom1195_stop.test.ts and kom1195_worker_host.test.ts). One client/worker is
// driven through all three steps below, so `spawned() === 1` throughout is
// itself the "the worker is reused for a following play" proof.
//
// A real thread means the search is never hand-paused: `sims` is large and
// `progressEveryPlayouts: 1` (with `batch: 1`) is used so the chunked loop's
// own macrotask yield (section 3) gives the already-queued `stop` message
// (posted back-to-back with the search message once `moveNow()` is called
// before the search message exists, per client.ts's `stopRequested` path) a
// great many chances to land well before `sims` playouts complete -- the same
// progress-driven arming kom1195_worker_host.test.ts's case (b) uses, just
// over a real thread instead of a fake scope.

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { ADVANCED_POSITION } from '../src/constants.ts';
import type { WorkerProgressEvent, WorkerReplyOk } from '../src/protocol.ts';
import type { KomugiClient as KomugiClientType, PlayResult } from '../src/client.ts';
import { nodeWorkerFactory } from './support/node_worker_adapter.ts';

// client.ts is a value import in the test (not just type) so `new
// KomugiClient(...)` below actually constructs one; re-imported with a
// value binding alongside the type above.
import { KomugiClient } from '../src/client.ts';

const MODEL = { url: 'model.onnx', sha256: 'sha-a' };
const ORT_ASSETS = { wasm: { mjs: 'ort.mjs', wasm: 'ort.wasm' } };

// Arms `onMoveNow` once `progress` has fired `after` times, mirroring
// kom1195_worker_host.test.ts's (b) arming convention but driven off the
// client's own `onProgress` option instead of raw posted messages.
function armAfterProgress(after: number, onMoveNow: () => void): (event: WorkerProgressEvent) => void {
  let count = 0;
  let armed = false;
  return (_event: WorkerProgressEvent): void => {
    count += 1;
    if (armed || count < after) return;
    armed = true;
    onMoveNow();
  };
}

test('AC7: moveNow() through a real worker stops a play, then an analyze(), then the same worker serves a following play', async (t) => {
  const { factory, spawned } = nodeWorkerFactory(['ok']);
  let onProgress: (event: WorkerProgressEvent) => void = () => undefined;
  const client: KomugiClientType = new KomugiClient({
    createWorker: factory,
    model: MODEL,
    ortAssets: ORT_ASSETS,
    onProgress: (event) => onProgress(event),
  });
  t.after(() => client.dispose());

  // --- step 1: a stoppable play, stopped by moveNow() -----------------
  onProgress = armAfterProgress(3, () => {
    assert.equal(client.moveNow(), true);
  });
  const playResult = await client.play({
    fen: ADVANCED_POSITION,
    sims: 50_000,
    batch: 1,
    stoppable: true,
    progressEveryPlayouts: 1,
  });
  assert.equal(playResult.status, 'move', `expected a move reply, got ${JSON.stringify(playResult)}`);
  const playReply = (playResult as { status: 'move'; reply: WorkerReplyOk }).reply;
  assert.equal(playReply.stats.stopReason, 'stop', JSON.stringify(playReply.stats));
  assert.ok(playReply.stats.playouts < 50_000, `expected the stop to cut the search short, got ${playReply.stats.playouts}`);

  // --- step 2: analyze(), stopped by moveNow() -------------------------
  onProgress = armAfterProgress(3, () => {
    assert.equal(client.moveNow(), true);
  });
  const analyzeResult: PlayResult = await client.analyze({
    fen: ADVANCED_POSITION,
    nodeCeiling: 20_001,
    batch: 1,
    progressEveryPlayouts: 1,
  });
  assert.equal(analyzeResult.status, 'move', `expected a move reply, got ${JSON.stringify(analyzeResult)}`);
  const analyzeReply = (analyzeResult as { status: 'move'; reply: WorkerReplyOk }).reply;
  assert.equal(analyzeReply.stats.stopReason, 'stop', JSON.stringify(analyzeReply.stats));
  assert.ok(analyzeReply.stats.playouts < 20_000, `expected the stop to cut the analysis short, got ${analyzeReply.stats.playouts}`);

  // --- step 3: the SAME worker serves a following, ordinary play ------
  onProgress = () => undefined;
  const followingResult = await client.play({ fen: ADVANCED_POSITION, sims: 4, batch: 1 });
  assert.equal(followingResult.status, 'move', `expected a move reply, got ${JSON.stringify(followingResult)}`);
  assert.equal(spawned(), 1, 'expected the worker from step 1 to be reused, not a fresh spawn');
});
