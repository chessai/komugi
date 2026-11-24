// U4: batched evaluator seam against a fake ORT module (design doc
// clientside-fast-mode.md section 2, section 8 unit U4). No real model,
// no network -- the real-model agreement test lives in
// batch_evaluator_ort.test.ts (re-review F4: this file must not also try
// to run the real-model case, since without KOM_MODEL that would have to
// either skip -- hiding the fake-ORT checks behind a false green -- or
// vacuously no-op).

import assert from 'node:assert/strict';
import { test } from 'node:test';

import {
  BackendError,
  chooseBucket,
  FAST_MODE_BUCKETS,
  runOrtBatch,
  wrapMlhBatch,
  type MlhBatchEvaluator,
} from '../src/batch_evaluator.ts';
import { softmax3f32 } from '../src/ort_core.ts';
import type { OrtModule, OrtSession } from '../src/ort_core.ts';
import { MlhContext, wrapMlhEvaluator, type MlhConfig, type MlhLeafEval } from '../src/search/mlh.ts';
import { WDL } from '../src/search/state.ts';
import type { MatchState, WdlEval } from '../src/search/state.ts';

const NUM_ACTIONS = 2;

class FakeTensor {
  readonly data: Float32Array;
  readonly dims: readonly number[];
  constructor(_kind: string, data: Float32Array, dims: readonly number[]) {
    this.data = data;
    this.dims = dims;
  }
}

function makeFakeOrt(): OrtModule {
  return {
    env: { wasm: { numThreads: 1 } },
    InferenceSession: {
      create: async (): Promise<OrtSession> => {
        throw new Error('fake ORT: InferenceSession.create is not used by runOrtBatch');
      },
    },
    Tensor: FakeTensor,
  };
}

function sumRow(data: Float32Array, offset: number, rowSize: number): number {
  let s = 0;
  for (let i = 0; i < rowSize; i++) s += data[offset + i];
  return s;
}

type RecordedCall = { dims: readonly number[]; data: Float32Array };
type Corruption = { row: number; field: 'policy' | 'wdl' | 'moves_left' };

// Deterministic echo session: row r's outputs are a function of that row's
// own input floats only (sum), so a test can hand-compute every expected
// output without depending on runOrtBatch's own arithmetic.
function makeFakeSession(rowSize: number, calls: RecordedCall[], corrupt?: Corruption): OrtSession {
  return {
    run: async (feeds: Record<string, unknown>) => {
      const tensor = feeds.obs as FakeTensor;
      calls.push({ dims: tensor.dims, data: Float32Array.from(tensor.data) });
      const bucket = tensor.dims[0];
      const policy = new Float32Array(bucket * NUM_ACTIONS);
      const wdl = new Float32Array(bucket * WDL);
      const movesLeft = new Float32Array(bucket);
      for (let r = 0; r < bucket; r++) {
        const s = sumRow(tensor.data, r * rowSize, rowSize);
        policy[r * NUM_ACTIONS] = s;
        policy[r * NUM_ACTIONS + 1] = -s;
        wdl[r * WDL] = s;
        wdl[r * WDL + 1] = 0;
        wdl[r * WDL + 2] = -s;
        movesLeft[r] = s / 10;
      }
      if (corrupt && corrupt.row < bucket) {
        if (corrupt.field === 'policy') policy[corrupt.row * NUM_ACTIONS] = Infinity;
        if (corrupt.field === 'wdl') wdl[corrupt.row * WDL] = NaN;
        if (corrupt.field === 'moves_left') movesLeft[corrupt.row] = NaN;
      }
      return { policy: { data: policy }, wdl: { data: wdl }, moves_left: { data: movesLeft } };
    },
  };
}

test('chooseBucket: smallest bucket >= rows, over the full bucket set', () => {
  assert.deepEqual(Array.from(FAST_MODE_BUCKETS), [1, 2, 4, 8, 16]);
  assert.equal(chooseBucket(1), 1);
  assert.equal(chooseBucket(2), 2);
  assert.equal(chooseBucket(3), 4);
  assert.equal(chooseBucket(4), 4);
  assert.equal(chooseBucket(5), 8);
  assert.equal(chooseBucket(8), 8);
  assert.equal(chooseBucket(9), 16);
  assert.equal(chooseBucket(16), 16);
  assert.throws(() => chooseBucket(17), RangeError);
  assert.throws(() => chooseBucket(0), RangeError);
});

test('runOrtBatch: bucket 4 for 3 rows, row 0 repeated in the pad slot, padding dropped from the result', async () => {
  const rowSize = 4;
  const rows = [
    Float32Array.from([1, 2, 3, 4]),
    Float32Array.from([5, 6, 7, 8]),
    Float32Array.from([9, 10, 11, 12]),
  ];
  const obsRows = new Float32Array(rows.length * rowSize);
  rows.forEach((row, r) => obsRows.set(row, r * rowSize));

  const calls: RecordedCall[] = [];
  const session = makeFakeSession(rowSize, calls);
  const ort = makeFakeOrt();

  const result = await runOrtBatch(session, ort, obsRows, rows.length, [rowSize]);

  assert.equal(calls.length, 1, 'one session.run per runOrtBatch call');
  assert.deepEqual(Array.from(calls[0].dims), [4, rowSize], 'bucket chosen for 3 rows is 4');
  assert.deepEqual(
    Array.from(calls[0].data.slice(3 * rowSize, 4 * rowSize)),
    Array.from(rows[0]),
    'the padding row sent to the backend is row 0 repeated, not zeros',
  );

  assert.equal(result.length, rows.length, 'padding never reaches the caller');
  for (let r = 0; r < rows.length; r++) {
    const s = sumRow(rows[r], 0, rowSize);
    assert.deepEqual(Array.from(result[r].policyLogits), [s, -s], `row ${r} policy`);
    const expectedWdl = softmax3f32(Float64Array.from([s, 0, -s]));
    assert.deepEqual(Array.from(result[r].wdl), Array.from(expectedWdl), `row ${r} wdl`);
    assert.equal(result[r].movesLeft, Math.fround(s / 10), `row ${r} movesLeft`);
  }
});

test('runOrtBatch: a NaN/Inf output row throws BackendError', async () => {
  const rowSize = 4;
  const twoRows = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]);
  const threeRows = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
  const ort = makeFakeOrt();

  await assert.rejects(
    runOrtBatch(makeFakeSession(rowSize, [], { row: 1, field: 'wdl' }), ort, twoRows, 2, [rowSize]),
    (err: unknown) => err instanceof BackendError,
  );
  await assert.rejects(
    runOrtBatch(makeFakeSession(rowSize, [], { row: 0, field: 'policy' }), ort, twoRows, 2, [rowSize]),
    (err: unknown) => err instanceof BackendError,
  );
  // rows=3 -> bucket 4, so row 3 is the padding slot: a NaN there (a GPU
  // driver computing garbage on a row the tree never sees) must still fail
  // the whole batch, not be silently discarded along with the padding.
  await assert.rejects(
    runOrtBatch(makeFakeSession(rowSize, [], { row: 3, field: 'moves_left' }), ort, threeRows, 3, [rowSize]),
    (err: unknown) => err instanceof BackendError,
    'a NaN in a padding row (row 3, bucket 4) must also fail the batch',
  );
});

test('runOrtBatch: rejects a row/obsRows length mismatch rather than silently misreading rows', async () => {
  const ort = makeFakeOrt();
  await assert.rejects(runOrtBatch(makeFakeSession(4, []), ort, new Float32Array(7), 2, [4]), RangeError);
});

const DUMMY_STATE = {} as unknown as MatchState;

function mlhLeaf(policy: [number, number], wdl: [number, number, number], movesLeft: number): MlhLeafEval {
  return { policyLogits: Float32Array.from(policy), wdl: Float64Array.from(wdl), movesLeft };
}

test('wrapMlhBatch: rows equal sequential wrapMlhEvaluator results, same ctx.mRoot/ctx.gated', async () => {
  const config: MlhConfig = { lam: 0.05, tau0: 0.1, tau1: 0.3, slope: 24 };
  const leaves: MlhLeafEval[] = [
    mlhLeaf([1, 0], [0.6, 0.1, 0.3], 0.9), // first row after a reset: sets ctx.mRoot, never gated
    mlhLeaf([0, 1], [0.9, 0.05, 0.05], 0.1),
    mlhLeaf([1, 1], [0.4, 0.2, 0.4], 0.5),
    mlhLeaf([2, 0], [0.05, 0.05, 0.9], 0.95),
  ];
  const states: MatchState[] = leaves.map(() => DUMMY_STATE);

  const batchCtx = new MlhContext();
  const batchInner: MlhBatchEvaluator = async () => leaves;
  const batched = wrapMlhBatch(batchInner, config, batchCtx);
  const batchResult: WdlEval[] = await batched(states);

  const seqCtx = new MlhContext();
  const seqResult: WdlEval[] = leaves.map((leaf) => wrapMlhEvaluator(() => leaf, config, seqCtx)(DUMMY_STATE));

  assert.equal(batchResult.length, seqResult.length);
  for (let i = 0; i < leaves.length; i++) {
    // Row i's own distinguishing policyLogits pins the positional contract
    // (N-5's fallback, re-review F13): if a row were permuted this would
    // fail at the first mismatched index rather than only in aggregate.
    assert.deepEqual(Array.from(batchResult[i].policyLogits), Array.from(seqResult[i].policyLogits), `row ${i} policy`);
    assert.deepEqual(Array.from(batchResult[i].wdl), Array.from(seqResult[i].wdl), `row ${i} wdl`);
  }
  assert.equal(batchCtx.mRoot, seqCtx.mRoot);
  assert.equal(batchCtx.gated, seqCtx.gated);
  assert.ok(seqCtx.gated > 0, 'the fixture config/leaves above must actually exercise a gate, or this test proves nothing');
});

test('wrapMlhBatch: a row-count mismatch from the inner evaluator is rejected', async () => {
  const config: MlhConfig = { lam: 0.05, tau0: 0.1, tau1: 0.3, slope: 24 };
  const leaves: MlhLeafEval[] = [mlhLeaf([1, 0], [0.5, 0.3, 0.2], 0.5), mlhLeaf([0, 1], [0.5, 0.3, 0.2], 0.5)];
  const states: MatchState[] = [DUMMY_STATE, DUMMY_STATE, DUMMY_STATE];
  const batched = wrapMlhBatch(async () => leaves, config, new MlhContext());
  await assert.rejects(batched(states), RangeError);
});
