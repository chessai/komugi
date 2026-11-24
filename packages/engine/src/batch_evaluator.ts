// Fast-mode batched evaluator seam (docs/design/clientside-fast-mode.md
// section 2, section 8 unit U4; re-review docs/reviews/2026-11-12-clientside-fast-mode-re-review.md
// F4/F11/F12/N-5). U4 runs in G1, before U2 adds `BatchWdlEvaluator`/
// `playoutBatchGen` to search/tree.ts, so `BatchWdlEvaluator` below is a
// LOCAL structural type (F12), not imported from tree.ts -- a value import
// of a name tree.ts does not define yet would fail to load under Node's
// type stripping, and even an `import type` would be importing a type that
// does not exist there yet. Its shape matches design 2.1's frozen seam
// (`(states: MatchState[]) => WdlEval[] | Promise<WdlEval[]>`) exactly, so
// U6 can assign this module's exports to the real `tree.batchEvaluator`
// property without a cast once U2 lands.

import type { OrtModule, OrtSession } from './ort_core.ts';
import { softmax3f32 } from './ort_core.ts';
import type { MlhConfig, MlhContext, MlhLeafEval, MlhLeafEvaluator } from './search/mlh.ts';
import { wrapMlhEvaluator } from './search/mlh.ts';
import type { MatchState, WdlEval } from './search/state.ts';
import { WDL } from './search/state.ts';

// Design 2.2: one dynamic-batch model, padded to these buckets only (the
// WebGPU EP's compiled shapes, section 3). U5's probe warms every one of
// them; U4 only ever picks among them.
export const FAST_MODE_BUCKETS: readonly number[] = [1, 2, 4, 8, 16];

// Smallest bucket that holds `rows` real rows (design 2.3).
export function chooseBucket(rows: number): number {
  if (rows < 1) throw new RangeError(`chooseBucket: rows must be >= 1, got ${rows}`);
  for (const bucket of FAST_MODE_BUCKETS) {
    if (bucket >= rows) return bucket;
  }
  throw new RangeError(
    `chooseBucket: ${rows} rows exceeds the largest bucket (${FAST_MODE_BUCKETS[FAST_MODE_BUCKETS.length - 1]})`,
  );
}

// S-3/F11: a NaN/Inf row from a GPU driver bug must surface as a backend
// failure here, not reach tree.ts's WDL-distribution validation, which has
// no idea a NaN came from the backend. Defined in U4 (F11), since the
// finite-row check lives in `runOrtBatch`; U6 owns the fallback this feeds.
export class BackendError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'BackendError';
  }
}

function assertFinite(values: Float32Array, label: string): void {
  for (let i = 0; i < values.length; i++) {
    if (!Number.isFinite(values[i])) {
      throw new BackendError(`runOrtBatch: non-finite ${label}[${i}] = ${values[i]}`);
    }
  }
}

// Design 2.2: "The staging buffer is allocated once per bucket and reused."
// Keyed by bucket only; a buffer whose length no longer matches the
// current row size (a different model's obsShape) is reallocated rather
// than reused stale.
const stagingBuffers = new Map<number, Float32Array>();

function stagingBuffer(bucket: number, rowSize: number): Float32Array {
  const wanted = bucket * rowSize;
  const existing = stagingBuffers.get(bucket);
  if (existing && existing.length === wanted) return existing;
  const fresh = new Float32Array(wanted);
  stagingBuffers.set(bucket, fresh);
  return fresh;
}

// One batched forward pass (design 2.2): `obsRows` is `rows` observations
// concatenated contiguously (row r at `[r*rowSize, (r+1)*rowSize)`). Runs
// at the smallest bucket `>= rows`, padding rows `rows..bucket-1` with row
// 0's observation (design 2.3: realistic activations, no special-cased
// zero input). Returns exactly `rows` rows, in input order -- padding
// outputs are discarded here and never reach the caller (2.3: "they never
// reach the tree or `tree.evaluations`").
export async function runOrtBatch(
  session: OrtSession,
  ort: OrtModule,
  obsRows: Float32Array,
  rows: number,
  obsShape: readonly number[],
): Promise<MlhLeafEval[]> {
  if (rows < 1) throw new RangeError(`runOrtBatch: rows must be >= 1, got ${rows}`);
  const rowSize = obsShape.reduce((a, b) => a * b, 1);
  if (obsRows.length !== rows * rowSize) {
    throw new RangeError(
      `runOrtBatch: obsRows has ${obsRows.length} floats, expected ${rows} * ${rowSize} = ${rows * rowSize}`,
    );
  }

  const bucket = chooseBucket(rows);
  const staged = stagingBuffer(bucket, rowSize);
  staged.set(obsRows.subarray(0, rows * rowSize), 0);
  for (let r = rows; r < bucket; r++) {
    staged.set(staged.subarray(0, rowSize), r * rowSize);
  }

  const tensor = new ort.Tensor('float32', staged, [bucket, ...obsShape]);
  const result = await session.run({ obs: tensor });
  const policy = Float32Array.from(result.policy!.data);
  const wdl = Float32Array.from(result.wdl!.data);
  const movesLeft = Float32Array.from(result.moves_left!.data);

  // S-3/F11: every output row finite, including padding rows -- a NaN
  // anywhere in the batch is a backend failure regardless of which row it
  // lands on.
  assertFinite(policy, 'policy');
  assertFinite(wdl, 'wdl');
  assertFinite(movesLeft, 'moves_left');

  const numActions = policy.length / bucket;
  const out: MlhLeafEval[] = [];
  for (let r = 0; r < rows; r++) {
    out.push({
      policyLogits: policy.slice(r * numActions, (r + 1) * numActions),
      wdl: softmax3f32(wdl.subarray(r * WDL, (r + 1) * WDL)),
      movesLeft: movesLeft[r]!,
    });
  }
  return out;
}

// F12: local structural type for the seam design 2.1 pins on `tree.ts`
// (U2, not yet written when U4 runs). Identical in shape to the type U2
// will export as `BatchWdlEvaluator`.
export type BatchWdlEvaluator = (states: MatchState[]) => WdlEval[] | Promise<WdlEval[]>;

// The worker's leaf evaluator shape (design 2.1): one batch call returning
// one MLH-flavoured row per input state, in input order.
export type MlhBatchEvaluator = (states: MatchState[]) => MlhLeafEval[] | Promise<MlhLeafEval[]>;

// Design 1.6: "the batched MLH wrapper applies the per-row arithmetic in
// row order (= collection order), so `ctx.gated` is deterministic ...
// reuses the exported `wrapMlhEvaluator` with a per-row synchronous inner
// returning that row's already-computed `MlhLeafEval`: no edit to mlh.ts."
// Row i of the output is always leaf i of `inner`'s result (design 2.1's
// frozen positional seam); F13's fallback for N-5 since the seam's type
// carries no index: a wrong row count is rejected rather than silently
// mismatched, and that positional contract is what batch_evaluator.test.ts
// pins instead of a permutation test.
export function wrapMlhBatch(inner: MlhBatchEvaluator, config: MlhConfig, ctx: MlhContext): BatchWdlEvaluator {
  return async (states: MatchState[]): Promise<WdlEval[]> => {
    const leaves = await inner(states);
    if (leaves.length !== states.length) {
      throw new RangeError(`wrapMlhBatch: inner returned ${leaves.length} rows for ${states.length} states`);
    }
    const out: WdlEval[] = [];
    for (let i = 0; i < states.length; i++) {
      const leaf = leaves[i]!;
      const rowLeaf: MlhLeafEvaluator = (): MlhLeafEval => leaf;
      out.push(wrapMlhEvaluator(rowLeaf, config, ctx)(states[i]!));
    }
    return out;
  };
}
