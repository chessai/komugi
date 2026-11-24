// KOM-1196 unit 6 (docs/design/kom1196-golden-fixtures.md sections 7-8,
// AC1/F6/F7/N4/N5): the committed-fixture suite (every top-level
// test/*.test.ts file except this one, fixture_flag.test.ts -- which spawn
// children of their own -- and bench.test.ts, a machine-dependent timing
// floor already excluded from the "test" script) runs to green with no
// Python interpreter reachable at all, and the only skips in that run are
// the pinned ONNX-model cases. Folds in what used to be a separate
// allowed_skips.test.ts (N4): both spawned the identical child, so one
// child run now carries both assertions.

import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { existsSync, mkdtempSync, readdirSync, rmSync, symlinkSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';

import { cleanEnv } from './fixture.ts';

const ENGINE_DIR = fileURLToPath(new URL('..', import.meta.url));
const TEST_DIR = fileURLToPath(new URL('.', import.meta.url));

// The child spawns of their own (N2/N4); excluded so this test does not
// recursively spawn them, and asserted to exist below so a rename cannot
// silently widen the run.
const EXCLUDED = ['no_python_regression.test.ts', 'fixture_flag.test.ts', 'bench.test.ts'];

// Section 3.4: the group E/F model cases, each gated behind
// `assertNoStrayKomVars()` then `t.skip('requires KOM_FULL_REGRESSION=1')`.
// Pinned by exact test name so any other skip fails this test.
const PINNED_SKIPS = new Map<string, string>([
  ['real model (KOM_MODEL): batch 1/4/8/16 on ORT wasm, when a model is given', 'requires KOM_FULL_REGRESSION=1'],
  [
    'runOrtBatch: per-row agreement with the fixture, and with a batch-1 run, within 1e-4 at buckets 1/4/8',
    'requires KOM_FULL_REGRESSION=1',
  ],
  [
    'ORT evaluator: TS observation matches the fixture bit for bit, ORT outputs agree within 1e-4',
    'requires KOM_FULL_REGRESSION=1',
  ],
  [
    'opt-in real install: KOM_ORT_WEB_DIR copies both glue pairs',
    'KOM_ORT_WEB_DIR is not set (point it at an installed onnxruntime-web package)',
  ],
]);

// Top level only (test/pack/* is a directory, not a `*.test.ts` name, so it
// is out of this glob by construction).
function globFiles(): string[] {
  return readdirSync(TEST_DIR)
    .filter((name) => name.endsWith('.test.ts') && !EXCLUDED.includes(name))
    .sort();
}

const SKIP_LINE = /^ok \d+ - (.*) # SKIP (.*)$/gm;

function skippedNamesAndReasons(output: string): Map<string, string> {
  const found = new Map<string, string>();
  for (const m of output.matchAll(SKIP_LINE)) {
    found.set(m[1], m[2]);
  }
  return found;
}

test('every file excluded from the AC1 glob still exists', () => {
  for (const name of EXCLUDED) {
    assert.ok(existsSync(join(TEST_DIR, name)), `excluded file is gone: ${name}`);
  }
});

test('AC1: the committed-fixture suite is python-free, and its only skips are the pinned ONNX cases', () => {
  // A fresh PATH containing only a `node` symlink to this very process's
  // binary -- the child can run node, nothing else, including no python.
  const pathDir = mkdtempSync(join(tmpdir(), 'kom1196-no-python-path-'));
  try {
    symlinkSync(process.execPath, join(pathDir, 'node'));
    const env = cleanEnv({ PATH: pathDir, NODE_TEST_CONTEXT: undefined });

    // Vacuity guard: in the exact env the child below runs under, neither
    // python3 nor python can even be found, let alone run.
    for (const bin of ['python3', 'python']) {
      const probe = spawnSync(bin, [], { env, timeout: 5_000 });
      assert.ok(probe.error, `${bin}: expected ENOENT, got no spawn error (status=${probe.status})`);
      assert.equal(probe.error?.code, 'ENOENT', `${bin}: expected ENOENT, got ${probe.error?.code}`);
    }

    const files = globFiles();
    assert.ok(files.length > 0, 'no files matched the AC1 glob');

    const r = spawnSync(
      process.execPath,
      ['--test', '--test-reporter=tap', ...files.map((f) => `test/${f}`)],
      { cwd: ENGINE_DIR, env, encoding: 'utf8', timeout: 600_000, maxBuffer: 64 * 1024 * 1024 },
    );
    if (r.error) throw r.error;
    const output = `${r.stdout}\n${r.stderr}`;
    assert.equal(r.status, 0, `child suite exited ${r.status}\n${output}`);
    assert.match(output, /^# fail 0$/m, `child suite did not report "# fail 0"\n${output}`);

    const skips = skippedNamesAndReasons(output);
    const actualNames = [...skips.keys()].sort();
    const expectedNames = [...PINNED_SKIPS.keys()].sort();
    assert.deepEqual(actualNames, expectedNames, `skip set diverged from the pin\n${output}`);
    for (const [name, reason] of skips) {
      assert.equal(reason, PINNED_SKIPS.get(name), `${name}: unexpected skip reason "${reason}"`);
    }
    assert.match(
      output,
      new RegExp(`^# skipped ${PINNED_SKIPS.size}$`, 'm'),
      `child suite's skip count did not match the pin (${PINNED_SKIPS.size})\n${output}`,
    );
  } finally {
    rmSync(pathDir, { recursive: true, force: true });
  }
});
