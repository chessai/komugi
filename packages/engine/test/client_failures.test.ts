// KOM-1199 unit U1 (design docs/design/kom1199-client.md section 4/7):
// classifyErrorReply's mapping table, the stamp:'mismatch' review-F2 case,
// and the two module invariants (KOMUGI_FAILURE_CODES membership, import
// type-only).

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';

import { WORKER_ERROR_CODES } from '../src/protocol.ts';
import type { WorkerReplyErr } from '../src/protocol.ts';
import {
  CLIENT_ERROR_CODES,
  KOMUGI_FAILURE_CODES,
  classifyErrorReply,
  validateInitReply,
  validateSearchReply,
} from '../src/client_failures.ts';

const HERE = dirname(fileURLToPath(import.meta.url));
const SRC_FILE = join(HERE, '..', 'src', 'client_failures.ts');

function errReply(err: string, code?: string): WorkerReplyErr {
  return code === undefined ? { id: 'req-1', err } : { id: 'req-1', err, code };
}

test('classifyErrorReply: every WORKER_ERROR_CODES member maps to itself at init stage', () => {
  for (const code of WORKER_ERROR_CODES) {
    const failure = classifyErrorReply('init', errReply('boom', code));
    assert.equal(failure.code, code);
    assert.equal(failure.stage, 'init');
    assert.equal(failure.workerCode, code);
  }
});

test('classifyErrorReply: every WORKER_ERROR_CODES member maps to itself at search stage (first row regardless of stage)', () => {
  for (const code of WORKER_ERROR_CODES) {
    const failure = classifyErrorReply('search', errReply('boom', code));
    assert.equal(failure.code, code);
    assert.equal(failure.stage, 'search');
    assert.equal(failure.workerCode, code);
  }
});

test('classifyErrorReply: init stage, code absent -> init-failed, no workerCode', () => {
  const failure = classifyErrorReply('init', errReply('boom'));
  assert.equal(failure.code, 'init-failed');
  assert.equal(failure.stage, 'init');
  assert.equal(failure.workerCode, undefined);
});

test('classifyErrorReply: init stage, unknown string code -> init-failed, workerCode kept', () => {
  const failure = classifyErrorReply('init', errReply('boom', 'some-unknown-code'));
  assert.equal(failure.code, 'init-failed');
  assert.equal(failure.stage, 'init');
  assert.equal(failure.workerCode, 'some-unknown-code');
});

test('classifyErrorReply: search stage, code absent -> backend-unavailable, no workerCode', () => {
  const failure = classifyErrorReply('search', errReply('boom'));
  assert.equal(failure.code, 'backend-unavailable');
  assert.equal(failure.stage, 'search');
  assert.equal(failure.workerCode, undefined);
});

test('classifyErrorReply: search stage, unknown string code -> backend-unavailable, workerCode kept', () => {
  const failure = classifyErrorReply('search', errReply('boom', 'some-unknown-code'));
  assert.equal(failure.code, 'backend-unavailable');
  assert.equal(failure.stage, 'search');
  assert.equal(failure.workerCode, 'some-unknown-code');
});

test('validateSearchReply: stamp === "mismatch" is ok, not a validation failure (review F2)', () => {
  const result = validateSearchReply(
    { id: 'req-1', san: 'P-11-12', action: 3, stamp: 'mismatch', stats: {} },
    'req-1',
    64,
  );
  assert.equal(result.ok, true);
  assert.ok(result.ok && result.reply.stamp === 'mismatch');
});

test('validateSearchReply: accepts a well-formed reply', () => {
  const result = validateSearchReply({ id: 'req-1', san: 'P-11-12', action: 3, stamp: 'present', stats: {} }, 'req-1', 64);
  assert.equal(result.ok, true);
});

test('validateSearchReply: rejects a non-matching id', () => {
  const result = validateSearchReply({ id: 'other', san: 'P-11-12', action: 3, stamp: 'present', stats: {} }, 'req-1', 64);
  assert.equal(result.ok, false);
  assert.ok(!result.ok && result.failure.code === 'illegal-reply' && result.failure.stage === 'search');
});

test('validateSearchReply: rejects an empty san', () => {
  const result = validateSearchReply({ id: 'req-1', san: '', action: 3, stamp: 'present', stats: {} }, 'req-1', 64);
  assert.equal(result.ok, false);
});

test('validateSearchReply: rejects a non-string san', () => {
  const result = validateSearchReply({ id: 'req-1', san: 42, action: 3, stamp: 'present', stats: {} }, 'req-1', 64);
  assert.equal(result.ok, false);
});

test('validateSearchReply: rejects an out-of-range action', () => {
  const result = validateSearchReply({ id: 'req-1', san: 'P-11-12', action: 64, stamp: 'present', stats: {} }, 'req-1', 64);
  assert.equal(result.ok, false);
});

test('validateSearchReply: rejects a non-integer action', () => {
  const result = validateSearchReply({ id: 'req-1', san: 'P-11-12', action: 1.5, stamp: 'present', stats: {} }, 'req-1', 64);
  assert.equal(result.ok, false);
});

test('validateSearchReply: rejects a missing stats object', () => {
  const result = validateSearchReply({ id: 'req-1', san: 'P-11-12', action: 3, stamp: 'present' }, 'req-1', 64);
  assert.equal(result.ok, false);
});

test('validateInitReply: accepts a well-formed reply', () => {
  const result = validateInitReply({ id: 'req-1', backend: 'wasm', threads: 1, batch: 1, probeRecord: {} }, 'req-1');
  assert.equal(result.ok, true);
});

test('validateInitReply: rejects a non-matching id', () => {
  const result = validateInitReply({ id: 'other', backend: 'wasm', threads: 1, batch: 1, probeRecord: {} }, 'req-1');
  assert.equal(result.ok, false);
  assert.ok(!result.ok && result.failure.code === 'illegal-reply' && result.failure.stage === 'init');
});

test('validateInitReply: rejects a bad backend', () => {
  const result = validateInitReply({ id: 'req-1', backend: 'cpu', threads: 1, batch: 1, probeRecord: {} }, 'req-1');
  assert.equal(result.ok, false);
});

test('validateInitReply: rejects non-positive threads', () => {
  const result = validateInitReply({ id: 'req-1', backend: 'wasm', threads: 0, batch: 1, probeRecord: {} }, 'req-1');
  assert.equal(result.ok, false);
});

test('validateInitReply: rejects non-positive batch', () => {
  const result = validateInitReply({ id: 'req-1', backend: 'wasm', threads: 1, batch: -1, probeRecord: {} }, 'req-1');
  assert.equal(result.ok, false);
});

test('validateInitReply: rejects a missing probeRecord', () => {
  const result = validateInitReply({ id: 'req-1', backend: 'wasm', threads: 1, batch: 1 }, 'req-1');
  assert.equal(result.ok, false);
});

// Design section 7 U1 states this invariant as "exactly 8 distinct members
// equal to WORKER_ERROR_CODES plus CLIENT_ERROR_CODES". protocol.ts's
// WORKER_ERROR_CODES has grown a 5th member ('threads-timeout', KOM-1214)
// since section 4's mapping table was written, so the real union is 9, not
// 8 -- see this task's final report. The invariant that still holds, and
// the one this test checks, is the equality to the union itself.
test('KOMUGI_FAILURE_CODES is exactly WORKER_ERROR_CODES plus CLIENT_ERROR_CODES, with no duplicates', () => {
  const expected = new Set<string>([...WORKER_ERROR_CODES, ...CLIENT_ERROR_CODES]);
  const actual = new Set(KOMUGI_FAILURE_CODES);
  assert.equal(KOMUGI_FAILURE_CODES.length, actual.size, 'KOMUGI_FAILURE_CODES has duplicate members');
  assert.deepEqual(actual, expected);
  assert.equal(actual.size, WORKER_ERROR_CODES.length + CLIENT_ERROR_CODES.length);
});

test('src/client_failures.ts: every import line is import type', () => {
  const src = readFileSync(SRC_FILE, 'utf8');
  const importLines = src.split('\n').filter((line) => /^\s*import\b/.test(line));
  assert.ok(importLines.length > 0, 'expected at least one import line');
  for (const line of importLines) {
    assert.match(line, /^\s*import type\b/, `not a type-only import: ${line}`);
  }
});
