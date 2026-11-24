// AC1: pure LEVELS/resolveLevel tests, no WorkerEngine.

import assert from 'node:assert/strict';
import { test } from 'node:test';

import {
  DEFAULT_LEVEL,
  LEVEL_NAMES,
  LEVELS,
  type LevelName,
  parseLevel,
  resolveLevel,
  type ResolvedLevel,
} from '../src/levels.ts';

interface BackendLike {
  backend: 'webgpu' | 'wasm';
  threads: number;
}

const ROWS: [LevelName, BackendLike, ResolvedLevel][] = [
  ['casual', { backend: 'webgpu', threads: 4 }, { sims: 100, batch: 2, maxMs: 2_000 }],
  ['casual', { backend: 'wasm', threads: 2 }, { sims: 100, batch: 2, maxMs: 2_000 }],
  ['casual', { backend: 'wasm', threads: 1 }, { sims: 100, batch: 1, maxMs: 2_000 }],
  ['focused', { backend: 'webgpu', threads: 4 }, { sims: 400, batch: 8, maxMs: 8_000 }],
  ['focused', { backend: 'wasm', threads: 2 }, { sims: 400, batch: 4, maxMs: 8_000 }],
  ['focused', { backend: 'wasm', threads: 1 }, { sims: 400, batch: 1, maxMs: 8_000 }],
  ['sharp', { backend: 'webgpu', threads: 4 }, { sims: 800, batch: 8, maxMs: 15_000 }],
  ['sharp', { backend: 'wasm', threads: 2 }, { sims: 800, batch: 4, maxMs: 15_000 }],
  ['sharp', { backend: 'wasm', threads: 1 }, { sims: 800, batch: 1, maxMs: 15_000 }],
  ['awakened', { backend: 'webgpu', threads: 4 }, { sims: 1600, batch: 8, maxMs: 30_000 }],
  ['awakened', { backend: 'wasm', threads: 2 }, { sims: 1600, batch: 4, maxMs: 30_000 }],
  ['awakened', { backend: 'wasm', threads: 1 }, { sims: 1600, batch: 1, maxMs: 30_000 }],
  // Two common real-world records (webgpu without cross-origin isolation;
  // wasm at a thread count above the measured 2): threads beyond "mt vs 1t"
  // must not change which batch is picked.
  ['focused', { backend: 'webgpu', threads: 1 }, { sims: 400, batch: 8, maxMs: 8_000 }],
  ['focused', { backend: 'wasm', threads: 8 }, { sims: 400, batch: 4, maxMs: 8_000 }],
];

test('resolveLevel pins the exact {sims, batch, maxMs} for 14 level/backend combinations (AC1)', () => {
  assert.equal(ROWS.length, 14);
  for (const [name, record, expected] of ROWS) {
    assert.deepEqual(resolveLevel(name, record), expected, `${name} / ${JSON.stringify(record)}`);
  }
});

test('LEVEL_NAMES and LEVELS agree on membership', () => {
  for (const name of LEVEL_NAMES) assert.ok(name in LEVELS, name);
  assert.deepEqual(Object.keys(LEVELS).sort(), [...LEVEL_NAMES].sort());
});

test('LEVELS, each entry and each entry\'s batch are frozen', () => {
  assert.ok(Object.isFrozen(LEVELS));
  for (const name of LEVEL_NAMES) {
    assert.ok(Object.isFrozen(LEVELS[name]), name);
    assert.ok(Object.isFrozen(LEVELS[name].batch), `${name}.batch`);
  }
});

test('DEFAULT_LEVEL is focused', () => {
  assert.equal(DEFAULT_LEVEL, 'focused');
});

test('parseLevel round-trips every LEVEL_NAMES entry', () => {
  for (const name of LEVEL_NAMES) assert.equal(parseLevel(name), name);
});

test('parseLevel rejects unknown values with /unknown level/', () => {
  const bad: unknown[] = ['Focused', 'champion', 7, '__proto__', 'constructor', undefined, 10n];
  for (const value of bad) {
    assert.throws(() => parseLevel(value), /unknown level/, String(value));
  }
});
