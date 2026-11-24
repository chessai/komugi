// KOM-1102 item 3: does a real search ever call session.run before a
// ponder batch already in flight on the SAME ORT session has resolved?
// ponder.ts:240's own doc says design 5.3's "an in-flight session.run
// cannot be cancelled and is awaited regardless" -- but nothing wired that
// await into worker.ts's real-request path; `abortPonder()` only flips the
// signal `runBatchedSearchAbortable` checks before its NEXT `gen.next()`,
// it never waits for the batch already in flight. This file instruments
// the one seam every real/pondered session.run call passes through
// (`MlhBatchEvaluator`, the shape `batch_evaluator.ts`'s `runOrtBatch`
// wraps around the real `OrtSession.run`) with a fake "session" that counts
// concurrent calls and only resolves a call when the test releases it.

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { BackendError } from '../src/batch_evaluator.ts';
import type { MlhBatchEvaluator } from '../src/batch_evaluator.ts';
import { ACTION_SIZE } from '../src/constants.ts';
import { makeFakeMlhEvaluator } from '../src/fake_evaluator.ts';
import { boardToFen } from '../src/fen.ts';
import type { MlhLeafEval } from '../src/search/mlh.ts';
import type { MatchState } from '../src/search/state.ts';
import { ADVANCED_POSITION } from '../src/constants.ts';
import { gameFromHistory } from '../src/root.ts';
import { sanRecords } from '../src/san.ts';
import type { WorkerReply, WorkerRequest } from '../src/protocol.ts';
import { WorkerEngine } from '../src/engine.ts';

function isOk(reply: WorkerReply): reply is WorkerReply & { action: number; san: string } {
  return !('err' in reply);
}

// An instrumented fake "session": every call is held (its promise never
// settles) until the test explicitly releases or rejects it, and
// `inFlight`/`maxConcurrent` are updated synchronously at call time -- the
// same place a real `OrtSession.run()` would be entered/left. `MlhBatchEvaluator`
// is exactly the seam both the real request's `tree.batchEvaluator` and the
// ponder session's `ctx.batchEvaluator` call (worker.ts's `attemptFast`/
// `runPonderSession`, both via `wrapMlhBatch`) -- in production that one
// function is backed by `batch_evaluator.ts`'s `runOrtBatch`, which does
// nothing but pack tensors around one `session.run()` call.
class FakeSession {
  calls = 0;
  inFlight = 0;
  maxConcurrent = 0;
  private readonly leaf = makeFakeMlhEvaluator(ACTION_SIZE);
  private pending: { rows: MlhLeafEval[]; resolve: (rows: MlhLeafEval[]) => void; reject: (err: unknown) => void }[] =
    [];

  run: MlhBatchEvaluator = (states: MatchState[]): Promise<MlhLeafEval[]> => {
    this.calls += 1;
    this.inFlight += 1;
    this.maxConcurrent = Math.max(this.maxConcurrent, this.inFlight);
    const rows = states.map((state) => this.leaf(state));
    const promise = new Promise<MlhLeafEval[]>((resolve, reject) => {
      this.pending.push({ rows, resolve, reject });
    });
    return promise.finally(() => {
      this.inFlight -= 1;
    });
  };

  get pendingCount(): number {
    return this.pending.length;
  }

  releaseOldest(): void {
    const next = this.pending.shift();
    if (next === undefined) throw new Error('releaseOldest: no pending call');
    next.resolve(next.rows);
  }

  rejectOldest(err: unknown): void {
    const next = this.pending.shift();
    if (next === undefined) throw new Error('rejectOldest: no pending call');
    next.reject(err);
  }
}

// Releases whatever is pending, one call at a time, until `until` settles --
// safe ONLY when nothing else could add a call this loop must not touch
// (the calibration run below, where pondering is disabled, and draining the
// tail end of a sequence once no more deliberately-held call remains to
// protect). Checking `settled` before touching `pending` means a call that
// appears in the SAME microtask turn `until` settles is left alone, never
// grabbed by this loop.
async function drainOneAtATime(session: FakeSession, until: Promise<unknown>): Promise<void> {
  let settled = false;
  until.then(
    () => {
      settled = true;
    },
    () => {
      settled = true;
    },
  );
  let iterations = 0;
  while (!settled) {
    iterations += 1;
    if (iterations > 20000) {
      throw new Error(`drainOneAtATime: stuck after ${iterations} iterations, pendingCount=${session.pendingCount}`);
    }
    if (session.pendingCount > 0) session.releaseOldest();
    await Promise.resolve();
  }
}

async function flushMicrotasks(n: number): Promise<void> {
  for (let i = 0; i < n; i++) await Promise.resolve();
}

async function waitForPending(session: FakeSession): Promise<void> {
  let iterations = 0;
  while (session.pendingCount === 0) {
    iterations += 1;
    if (iterations > 20000) throw new Error(`waitForPending: stuck after ${iterations} iterations`);
    await Promise.resolve();
  }
}

// A short real, legal history from the advanced start (same convention as
// ponder.test.ts's own `buildHistory`).
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

// Counts exactly how many `session.run` calls `request`'s OWN search makes,
// on a throwaway engine with pondering disabled -- disabled so this count
// can never include a ponder call, which is the one thing the real test
// below must NOT accidentally release while draining request's own search.
// Safe to drain generically (`drainOneAtATime`) because with pondering off,
// nothing but `request`'s own search will ever add a pending call.
async function countOwnCalls(request: WorkerRequest): Promise<number> {
  const calib = new WorkerEngine();
  calib.handlePonderSettings({ id: 'calib', type: 'ponderSettings', enabled: false });
  const session = new FakeSession();
  const reply = calib.handleFast(request, session.run);
  await drainOneAtATime(session, reply);
  await reply;
  return session.calls;
}

// Runs `request` on `engine`/`session` to completion, releasing EXACTLY as
// many calls as `request`'s own search needs (precomputed by
// `countOwnCalls` against an identical request on a pondering-disabled
// engine) -- never more, so a ponder call this same reply kicks off
// afterward (worker.ts's `attemptFast`, synchronously, before `handleFast`'s
// own promise settles) is left pending, untouched, for the caller to
// inspect.
async function runLeavingPonderPending(
  engine: WorkerEngine,
  session: FakeSession,
  request: WorkerRequest,
): Promise<WorkerReply> {
  const ownCalls = await countOwnCalls(request);
  const reply = engine.handleFast(request, session.run);
  for (let i = 0; i < ownCalls; i++) {
    await waitForPending(session);
    session.releaseOldest();
  }
  return reply;
}

test('case (a): a real request never overlaps the ponder session`s in-flight session.run', async () => {
  const engine = new WorkerEngine();
  const session = new FakeSession();
  const req1 = baseRequest(1);

  const reply1 = await runLeavingPonderPending(engine, session, req1);
  assert.ok(isOk(reply1), JSON.stringify(reply1));

  // The ponder session's own first session.run call must already be
  // pending (worker.ts spawns it synchronously, inside attemptFast, before
  // handleFast's promise settles) -- otherwise this position/budget never
  // actually exercises pondering and the test proves nothing.
  await waitForPending(session);
  assert.equal(session.pendingCount, 1, 'expected exactly the ponder session`s own call pending');

  // A real request: cache miss (different sims), so it genuinely searches.
  const req2 = baseRequest(2, { sims: 17 });
  const p2 = engine.handleFast(req2, session.run);

  // Flush generously. With the bug, req2's first session.run call joins the
  // ponder's still-unresolved one within a handful of microtasks regardless
  // of how many we flush here. With the fix, req2 is genuinely blocked on
  // the ponder task's own promise and NO amount of flushing frees it until
  // the ponder's pending call is released below.
  await flushMicrotasks(50);
  assert.equal(
    session.pendingCount,
    1,
    'req2 must not call session.run while the ponder session`s run is still unresolved',
  );

  // Now let the ponder's in-flight run resolve (it is aborted by now, so it
  // is discarded, never cached) and drain req2 to completion.
  session.releaseOldest();
  await drainOneAtATime(session, p2);
  const reply2 = await p2;
  assert.ok(isOk(reply2), JSON.stringify(reply2));

  assert.equal(session.maxConcurrent, 1, 'two session.run calls overlapped');

  // The real reply must equal the unpondered reply -- a fresh engine,
  // pondering disabled, answering the same two requests for real.
  const plain = new WorkerEngine();
  plain.handlePonderSettings({ id: 's', type: 'ponderSettings', enabled: false });
  const plainSession = new FakeSession();
  const pr1 = plain.handleFast(req1, plainSession.run);
  await drainOneAtATime(plainSession, pr1);
  await pr1;
  const pr2 = plain.handleFast(req2, plainSession.run);
  await drainOneAtATime(plainSession, pr2);
  const plainReply2 = await pr2;
  assert.ok(isOk(plainReply2), JSON.stringify(plainReply2));
  if (isOk(reply2) && isOk(plainReply2)) {
    assert.equal(reply2.action, plainReply2.action);
    assert.equal(reply2.san, plainReply2.san);
  }
});

test('case (b): a real request never overlaps the ponder session even when its in-flight run rejects', async () => {
  const engine = new WorkerEngine();
  const session = new FakeSession();
  const req1 = baseRequest(1);

  const reply1 = await runLeavingPonderPending(engine, session, req1);
  assert.ok(isOk(reply1), JSON.stringify(reply1));
  await waitForPending(session);
  assert.equal(session.pendingCount, 1);

  const req2 = baseRequest(2, { sims: 17 });
  const p2 = engine.handleFast(req2, session.run);
  await flushMicrotasks(50);
  assert.equal(session.pendingCount, 1, 'req2 must not call session.run before the ponder`s rejection is observed');

  // The ponder's in-flight run fails (a backend error), rather than
  // resolving -- design 5.3's "discarded" path, not `runFast`'s own
  // fallback ladder (that's for the REAL request's own failures).
  session.rejectOldest(new BackendError('fake backend failure'));
  await drainOneAtATime(session, p2);
  const reply2 = await p2;
  assert.ok(isOk(reply2), JSON.stringify(reply2));
  assert.equal(session.maxConcurrent, 1, 'two session.run calls overlapped across the ponder`s rejection');
});

test('case (c): several rapid real requests during pondering stay fully serialized', async () => {
  const engine = new WorkerEngine();
  const session = new FakeSession();
  const req1 = baseRequest(1);

  const reply1 = await runLeavingPonderPending(engine, session, req1);
  assert.ok(isOk(reply1), JSON.stringify(reply1));
  await waitForPending(session);
  assert.equal(session.pendingCount, 1);

  // Three real requests, fired back to back, all cache misses (distinct
  // sims) so each genuinely searches, while the ponder's own call is still
  // held.
  const req2 = baseRequest(2, { sims: 17 });
  const req3 = baseRequest(3, { sims: 18 });
  const req4 = baseRequest(4, { sims: 19 });
  const p2 = engine.handleFast(req2, session.run);
  const p3 = engine.handleFast(req3, session.run);
  const p4 = engine.handleFast(req4, session.run);

  await flushMicrotasks(50);
  assert.equal(session.pendingCount, 1, 'no queued request may call session.run before the ponder`s run settles');

  session.releaseOldest();
  const all = Promise.all([p2, p3, p4]);
  await drainOneAtATime(session, all);
  const [reply2, reply3, reply4] = await all;
  for (const reply of [reply2, reply3, reply4]) assert.ok(isOk(reply), JSON.stringify(reply));

  assert.equal(session.maxConcurrent, 1, 'two session.run calls overlapped across the rapid request burst');
});

test('case (d): the single-leaf path (handle) never shares a session with pondering, so it cannot overlap one', async () => {
  // Pondering (ponder.ts's `runPonderSession`) is started ONLY from the
  // batched path (`attemptFast`/`lookupPonder`, both reached only through
  // `handleFast`) -- `handle()`/`handleAsync()` run against
  // `this.leafEvaluator`, a wholly separate evaluator never installed as
  // `ctx.batchEvaluator`. So a single-leaf request has no shared "session"
  // with an in-flight ponder session to overlap in the first place; it also
  // does not abort pondering (the "interloper" case ponder.test.ts already
  // pins), which this test re-confirms end to end against the instrumented
  // session: the single-leaf call completes on its own, and the pondering
  // the batched request started is untouched by it.
  const engine = new WorkerEngine();
  const session = new FakeSession();
  const req1 = baseRequest(1);

  const reply1 = await runLeavingPonderPending(engine, session, req1);
  assert.ok(isOk(reply1), JSON.stringify(reply1));
  await waitForPending(session);
  assert.equal(session.pendingCount, 1, 'expected the ponder session`s own call pending');

  // The single-leaf path: synchronous, uses the engine's default fake leaf
  // evaluator, never touches `session` at all.
  const leafReply = engine.handle({ id: 'leaf', fen: ADVANCED_POSITION, sims: 4 });
  assert.ok(isOk(leafReply), JSON.stringify(leafReply));

  // The ponder session is unaffected: nothing aborted it (no real request
  // arrived via handleFast), so it runs to its own natural completion --
  // draining however many further session.run calls its own search needs,
  // exactly as it would have without the single-leaf call in between.
  assert.equal(session.pendingCount, 1);
  const pondering = engine.ponderingPromise();
  await drainOneAtATime(session, pondering);
  await pondering;
  assert.ok(engine.ponderCacheSize > 0, 'the single-leaf call must not have disturbed pondering');
  assert.equal(session.maxConcurrent, 1);
});
