// Fast-mode unit U8: the Node bench (docs/design/clientside-fast-mode.md
// section 8 unit U8; ticket KOM-1094). Needs no fixture (design 8's fan-out
// note): every check runs against the fake evaluator over a fixed position.
// The real-ORT bench case runs only under KOM_FULL_REGRESSION=1 (KOM-1196),
// and skips with that reason otherwise; every other case always runs.

import assert from 'node:assert/strict';
import { mkdtempSync, readFileSync, readdirSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';

import {
  EVAL_CEILING_SLACK,
  runFastBenchFake,
  runFastBenchOrt,
  writeReport,
  type FastBenchResult,
} from '../bench/fast.ts';
import { loadOrtSession } from './support/ort_node.ts';
import { FULL_REGRESSION, assertNoStrayKomVars } from './fixture.ts';

const BATCHES = [1, 4, 8, 16];
const SIMS = 200;

// One untimed warm-up at every batch before any assertion reads a timing,
// the same discipline benchReplay (bench/replay.ts) uses -- JIT warm-up
// jitter would otherwise dominate a single short run's phase split.
function warm(): void {
  for (const batch of BATCHES) runFastBenchFake(batch, 32);
}
warm();

function checkSchema(label: string, r: FastBenchResult, batch: number, sims: number): void {
  assert.equal(r.batch, batch, `${label}: batch`);
  assert.equal(r.sims, sims, `${label}: sims`);
  assert.ok(r.backend === 'fake' || r.backend === 'ort-wasm', `${label}: backend`);
  assert.ok(Number.isFinite(r.totalMs) && r.totalMs >= 0, `${label}: totalMs`);
  assert.ok(Number.isFinite(r.simsPerSec) && r.simsPerSec >= 0, `${label}: simsPerSec`);
  assert.ok(Number.isInteger(r.playoutsUsed) && r.playoutsUsed >= 0, `${label}: playoutsUsed`);
  assert.ok(Number.isInteger(r.evaluationsUsed) && r.evaluationsUsed >= 1, `${label}: evaluationsUsed`);
  for (const key of ['netMs', 'encodeMs', 'treeMs', 'surveyMs'] as const) {
    assert.ok(Number.isFinite(r.phases[key]) && r.phases[key] >= 0, `${label}: phases.${key}`);
  }
}

test('schema: every field present and well-typed at batch 1/4/8/16', () => {
  for (const batch of BATCHES) {
    const r = runFastBenchFake(batch, SIMS);
    checkSchema(`B${batch}`, r, batch, SIMS);
  }
});

test('batch 1 and batch 8 agree on playout/evaluation accounting', () => {
  const a = runFastBenchFake(1, SIMS);
  const b = runFastBenchFake(8, SIMS);
  // design 6.2 item 2 (visit conservation): every search spends exactly the
  // requested budget, regardless of batch.
  assert.equal(a.playoutsUsed, SIMS, 'batch 1: playoutsUsed != sims');
  assert.equal(b.playoutsUsed, SIMS, 'batch 8: playoutsUsed != sims');
  // design 6.2 item 4 (evaluation accounting): the same ceiling holds at
  // every batch size, even though the two trees are not the same shape
  // (virtual loss distortion, design 1.9) and so need not share a count.
  assert.ok(a.evaluationsUsed <= SIMS + EVAL_CEILING_SLACK, `batch 1: evaluationsUsed ${a.evaluationsUsed}`);
  assert.ok(b.evaluationsUsed <= SIMS + EVAL_CEILING_SLACK, `batch 8: evaluationsUsed ${b.evaluationsUsed}`);
  assert.ok(a.evaluationsUsed >= 1, 'batch 1: at least the root evaluation');
  assert.ok(b.evaluationsUsed >= 1, 'batch 8: at least the root evaluation');
});

test('the phase split sums to within 10% of the total, at every batch', () => {
  for (const batch of BATCHES) {
    const r = runFastBenchFake(batch, SIMS);
    const sum = r.phases.netMs + r.phases.encodeMs + r.phases.treeMs + r.phases.surveyMs;
    const tol = Math.max(1, r.totalMs * 0.1); // a floor for near-zero totals
    console.log(
      `fast_bench B${batch}: total=${r.totalMs.toFixed(2)}ms sum=${sum.toFixed(2)}ms ` +
        `net=${r.phases.netMs.toFixed(2)} encode=${r.phases.encodeMs.toFixed(2)} ` +
        `tree=${r.phases.treeMs.toFixed(2)} survey=${r.phases.surveyMs.toFixed(2)} ` +
        `rate=${Math.round(r.simsPerSec)}/s`,
    );
    assert.ok(
      Math.abs(sum - r.totalMs) <= tol,
      `B${batch}: phase split ${sum.toFixed(2)}ms vs total ${r.totalMs.toFixed(2)}ms exceeds 10%`,
    );
  }
});

test('writes nothing outside a temp dir', () => {
  const dir = mkdtempSync(join(tmpdir(), 'kom-1094-fast-bench-'));
  try {
    const before = readdirSync(dir);
    assert.equal(before.length, 0, 'temp dir not empty before writeReport');
    const results = BATCHES.map((batch) => runFastBenchFake(batch, 32));
    const reportPath = join(dir, 'fast_bench.json');
    writeReport(results, reportPath);
    const after = readdirSync(dir);
    assert.deepEqual(after, ['fast_bench.json'], 'writeReport touched more than its own file');
    const loaded = JSON.parse(readFileSync(reportPath, 'utf8')) as { results: FastBenchResult[] };
    assert.equal(loaded.results.length, BATCHES.length);
    for (let i = 0; i < BATCHES.length; i++) {
      assert.equal(loaded.results[i].batch, BATCHES[i]);
    }
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

// A real `--batch dynamic` ONNX model on ORT wasm (ticket SCOPE's second
// bullet). KOM-1196 (golden-fixtures design section 3.4): full-regression
// only. Without the flag it skips with the named reason (a stray KOM_* still
// fails, N3); with the flag KOM_MODEL is required.
const threads = Number(process.env.KOM_FAST_BENCH_THREADS ?? '1');

test('real model (KOM_MODEL): batch 1/4/8/16 on ORT wasm, when a model is given', async (t) => {
  if (!FULL_REGRESSION) {
    assertNoStrayKomVars();
    t.skip('requires KOM_FULL_REGRESSION=1');
    return;
  }
  const KOM_MODEL = process.env.KOM_MODEL;
  if (!KOM_MODEL) throw new Error('KOM_FULL_REGRESSION=1 but KOM_MODEL is not set');
  const { ort, session } = await loadOrtSession(KOM_MODEL, { threads });
  const results: FastBenchResult[] = [];
  for (const batch of BATCHES) {
    const r = await runFastBenchOrt(session, ort, batch, SIMS, threads);
    checkSchema(`ort B${batch}`, r, batch, SIMS);
    assert.equal(r.backend, 'ort-wasm');
    assert.equal(r.threads, threads);
    const sum = r.phases.netMs + r.phases.encodeMs + r.phases.treeMs + r.phases.surveyMs;
    const tol = Math.max(1, r.totalMs * 0.1);
    assert.ok(Math.abs(sum - r.totalMs) <= tol, `ort B${batch}: phase split exceeds 10% of total`);
    console.log(
      `fast_bench ort B${batch} threads=${threads}: total=${r.totalMs.toFixed(2)}ms ` +
        `net=${r.phases.netMs.toFixed(2)} encode=${r.phases.encodeMs.toFixed(2)} ` +
        `tree=${r.phases.treeMs.toFixed(2)} survey=${r.phases.surveyMs.toFixed(2)} ` +
        `rate=${Math.round(r.simsPerSec)}/s`,
    );
    results.push(r);
  }
  assert.equal(results.length, BATCHES.length);
});
