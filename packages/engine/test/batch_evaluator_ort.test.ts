// U4: runOrtBatch against a real dynamic-batch ONNX model (design doc
// clientside-fast-mode.md section 2.2/2.3, section 8 unit U4; re-review
// F4). A SEPARATE file from batch_evaluator.test.ts on purpose (F4): it
// must FAIL -- not skip -- when KOM_MODEL/KOM_ONNX_FIXTURE are unset, so
// an env-skip can never hide this unit's real-model checks the way a
// quiet `t.skip` would (the same discipline ort.test.ts already uses).
// KOM-1196: that holds under KOM_FULL_REGRESSION=1; without the flag the
// case skips with that reason.

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';

import { runOrtBatch } from '../src/batch_evaluator.ts';
import { softmax3f32 } from '../src/ort_core.ts';
import { loadOrtSession } from './support/ort_node.ts';
import { FULL_REGRESSION, assertNoStrayKomVars } from './fixture.ts';

const ATOL = 1e-4; // design 2.3's S-8 tolerance: a batch-B run agrees with batch 1 within 1e-4.

function requiredEnv(name: string): string {
  const value = process.env[name];
  if (!value) {
    throw new Error(`${name} is not set: batch_evaluator_ort.test.ts must fail, not skip, without it`);
  }
  return value;
}

type FixturePosition = { obs: string; wdl: string };
type Fixture = { shapes: { obs: number[]; wdl: number[] }; positions: FixturePosition[] };

function decodeFloat32(base64: string, expectedLength: number): Float32Array {
  const buf = Buffer.from(base64, 'base64');
  const data = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
  assert.equal(data.length, expectedLength, `decodeFloat32: expected ${expectedLength} floats, got ${data.length}`);
  return data;
}

function maxAbsDelta(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let max = 0;
  for (let i = 0; i < a.length; i++) max = Math.max(max, Math.abs(a[i] - b[i]));
  return max;
}

function concatRows(rows: Float32Array[]): Float32Array {
  const rowSize = rows[0].length;
  const out = new Float32Array(rows.length * rowSize);
  rows.forEach((row, r) => out.set(row, r * rowSize));
  return out;
}

test('runOrtBatch: per-row agreement with the fixture, and with a batch-1 run, within 1e-4 at buckets 1/4/8', async (t) => {
  // KOM-1196 (golden-fixtures design section 3.4): full-regression only; a
  // stray KOM_* without the flag still fails rather than skips (N3).
  if (!FULL_REGRESSION) {
    assertNoStrayKomVars();
    t.skip('requires KOM_FULL_REGRESSION=1');
    return;
  }
  const modelPath = requiredEnv('KOM_MODEL');
  const fixturePath = requiredEnv('KOM_ONNX_FIXTURE');
  const ortwebDir = process.env.ORTWEB_DIR;

  const fixture = JSON.parse(readFileSync(fixturePath, 'utf8')) as Fixture;
  assert.ok(fixture.positions.length >= 8, `${fixturePath} needs >= 8 positions, has ${fixture.positions.length}`);
  const obsShape = fixture.shapes.obs;
  const rowSize = obsShape.reduce((a, b) => a * b, 1);

  const { ort, session } = await loadOrtSession(modelPath, { ortwebDir, threads: 1 });

  const positions = fixture.positions.slice(0, 8);
  const obsRows = positions.map((p) => decodeFloat32(p.obs, rowSize));

  // Reference: one row at a time, bucket 1 (what the design calls "batch
  // 1" -- every batch run must agree with this within 1e-4).
  const reference = [];
  for (const obs of obsRows) {
    reference.push(...(await runOrtBatch(session, ort, obs, 1, obsShape)));
  }
  assert.equal(reference.length, positions.length);

  let worstVsFixture = 0;
  for (let i = 0; i < reference.length; i++) {
    const wantWdlLogits = decodeFloat32(positions[i].wdl, fixture.shapes.wdl.reduce((a, b) => a * b, 1));
    const wantWdl = softmax3f32(wantWdlLogits);
    worstVsFixture = Math.max(worstVsFixture, maxAbsDelta(reference[i].wdl, wantWdl));
  }
  console.log(`batch_evaluator_ort: bucket=1 max|wdl delta| vs fixture = ${worstVsFixture}`);
  assert.ok(worstVsFixture <= ATOL, `bucket-1 wdl vs fixture delta ${worstVsFixture} exceeds ${ATOL}`);

  for (const rows of [4, 8]) {
    const batch = await runOrtBatch(session, ort, concatRows(obsRows.slice(0, rows)), rows, obsShape);
    assert.equal(batch.length, rows, `bucket for ${rows} rows must drop padding`);
    let worstPolicy = 0;
    let worstWdl = 0;
    let worstMovesLeft = 0;
    for (let r = 0; r < rows; r++) {
      worstPolicy = Math.max(worstPolicy, maxAbsDelta(batch[r].policyLogits, reference[r].policyLogits));
      worstWdl = Math.max(worstWdl, maxAbsDelta(batch[r].wdl, reference[r].wdl));
      worstMovesLeft = Math.max(worstMovesLeft, Math.abs(batch[r].movesLeft - reference[r].movesLeft));
    }
    console.log(
      `batch_evaluator_ort: bucket for rows=${rows} max|delta| vs bucket-1 policy=${worstPolicy} ` +
        `wdl=${worstWdl} moves_left=${worstMovesLeft}`,
    );
    assert.ok(worstPolicy <= ATOL, `rows=${rows} policy delta ${worstPolicy} exceeds ${ATOL}`);
    assert.ok(worstWdl <= ATOL, `rows=${rows} wdl delta ${worstWdl} exceeds ${ATOL}`);
    assert.ok(worstMovesLeft <= ATOL, `rows=${rows} moves_left delta ${worstMovesLeft} exceeds ${ATOL}`);
  }
});
