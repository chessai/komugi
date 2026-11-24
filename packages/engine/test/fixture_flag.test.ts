// KOM-1196: the KOM_FULL_REGRESSION flag contract of test/fixture.ts
// (docs/design/kom1196-golden-fixtures.md sections 5 and 7). Each case runs a
// child `node --test` whose environment comes from cleanEnv() (N4), so this
// file's own KOM_*/KOM_FULL_REGRESSION never leaks into a child.

import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { test } from 'node:test';

import { cleanEnv } from './fixture.ts';

const ENGINE_DIR = fileURLToPath(new URL('..', import.meta.url));

type ChildRun = { status: number | null; output: string };

function runChild(file: string, overrides: Record<string, string | undefined>): ChildRun {
  // NODE_TEST_CONTEXT is set by our own runner; a nested `node --test` that
  // inherits it would report to this process instead of running standalone.
  const env = cleanEnv({ NODE_TEST_CONTEXT: undefined, ...overrides });
  const r = spawnSync(process.execPath, ['--test', '--test-reporter=tap', file], {
    cwd: ENGINE_DIR,
    env,
    encoding: 'utf8',
    timeout: 300_000,
  });
  if (r.error) throw r.error;
  return { status: r.status, output: `${r.stdout}\n${r.stderr}` };
}

function assertFailed(r: ChildRun, needle: string, label: string): void {
  assert.notEqual(r.status, 0, `${label}: child exited 0\n${r.output}`);
  assert.ok(r.output.includes(needle), `${label}: child output lacks ${JSON.stringify(needle)}\n${r.output}`);
}

test('KOM_FULL_REGRESSION=1 with KOM_FIXTURE unset: fen.test.ts fails', () => {
  const r = runChild('test/fen.test.ts', { KOM_FULL_REGRESSION: '1' });
  assertFailed(r, 'KOM_FULL_REGRESSION=1 but KOM_FIXTURE is not set', 'flag, no KOM_FIXTURE');
});

test('KOM_FULL_REGRESSION=1 KOM_FIXTURE=/nonexistent: the override is honoured (ENOENT)', () => {
  const r = runChild('test/fen.test.ts', { KOM_FULL_REGRESSION: '1', KOM_FIXTURE: '/nonexistent' });
  assertFailed(r, 'ENOENT', 'flag, KOM_FIXTURE=/nonexistent');
});

test('KOM_FIXTURE set without the flag: fen.test.ts fails', () => {
  const r = runChild('test/fen.test.ts', { KOM_FIXTURE: '/nonexistent' });
  assertFailed(r, 'KOM_FIXTURE set without KOM_FULL_REGRESSION=1', 'KOM_FIXTURE, no flag');
});

test('KOM_MODEL set without the flag: ort.test.ts fails rather than skipping (N3)', () => {
  const r = runChild('test/ort.test.ts', { KOM_MODEL: '/nonexistent' });
  assertFailed(r, 'KOM_MODEL set without KOM_FULL_REGRESSION=1', 'KOM_MODEL, no flag');
  assert.match(r.output, /^# skipped 0$/m, `KOM_MODEL, no flag: the case skipped\n${r.output}`);
});

test('KOM_FULL_REGRESSION=true: fen.test.ts fails at module load (N11)', () => {
  const r = runChild('test/fen.test.ts', { KOM_FULL_REGRESSION: 'true' });
  assertFailed(r, 'KOM_FULL_REGRESSION must be "1" if set', 'flag=true');
});
