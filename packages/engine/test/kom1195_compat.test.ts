// KOM-1195 AC1 (docs/design/kom1195-analysis-api.md sections 2, 8, 9 U0):
// every golden in `test/fixtures/kom1195_compat_goldens.json.gz`, recorded
// on the pre-change base by `test/kom1195_record_goldens.ts`, is replayed
// through `handleFast` with no new field and must be byte-identical
// (`JSON.stringify`, `ms` zeroed, `descriptor` kept). Committed green on the
// base, so a later unit that changes an un-opted request's path goes red;
// the `g0p88` check below proves the fixture sits where that can happen
// (re-review N6).

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import { gunzipSync } from 'node:zlib';

import { loadFixture } from './fixture.ts';
import {
  GOLDEN_BATCHES,
  GOLDEN_SIMS,
  GOLDENS_PATH,
  NEVER_MS,
  parityGoldenName,
  runGolden,
} from './kom1195_record_goldens.ts';
import type { GoldenRecord } from './kom1195_record_goldens.ts';

const RAW = readFileSync(GOLDENS_PATH);
const GOLDENS = JSON.parse(gunzipSync(RAW).toString('utf8')) as GoldenRecord[];
const BY_NAME = new Map(GOLDENS.map((rec) => [rec.name, rec]));

// The request fields KOM-1195 adds (section 2); no golden may carry one.
const NEW_FIELDS = ['stoppable', 'progressEveryPlayouts', 'progressEveryMs', 'topN', 'analyze', 'nodeCeiling'];

test('kom1195 goldens: gzip with a zero header mtime, unique names, no new field on any request', () => {
  assert.equal(RAW[0], 0x1f);
  assert.equal(RAW[1], 0x8b);
  assert.equal(RAW.readUInt32LE(4), 0, 'gzip header mtime');
  assert.ok(GOLDENS.length > 0, 'empty golden fixture');
  assert.equal(BY_NAME.size, GOLDENS.length, 'duplicate golden names');
  for (const rec of GOLDENS) {
    for (const field of NEW_FIELDS) {
      assert.ok(!(field in rec.request), `${rec.name}: request carries ${field}`);
    }
    assert.ok(!('err' in rec.reply), `${rec.name}: golden reply is an error`);
  }
  const kinds = new Set(GOLDENS.map((rec) => rec.kind));
  for (const kind of ['parity', 'level', 'history', 'kom1198']) assert.ok(kinds.has(kind as GoldenRecord['kind']), kind);
});

test('kom1195 goldens: >= 4 records per committed non-terminal parity position, all at sims 96', () => {
  const positions = loadFixture().filter((rec) => !rec.terminal);
  assert.ok(positions.length > 0, 'no non-terminal parity positions');
  for (const pos of positions) {
    const records = GOLDENS.filter((rec) => rec.kind === 'parity' && rec.position === pos.id);
    assert.ok(records.length >= 4, `${pos.id}: ${records.length} records`);
    for (const rec of records) {
      assert.equal((rec.request as { sims?: number }).sims, GOLDEN_SIMS, `${rec.name}: sims`);
      assert.equal(rec.request.fen, pos.fen, `${rec.name}: fen`);
    }
    for (const batch of GOLDEN_BATCHES) {
      const plain = BY_NAME.get(parityGoldenName(pos.id, batch, false));
      const capped = BY_NAME.get(parityGoldenName(pos.id, batch, true));
      assert.ok(plain !== undefined && capped !== undefined, `${pos.id} b${batch}: missing a maxMs variant`);
      assert.equal((plain.request as { batch?: number }).batch, batch);
      assert.equal((capped.request as { batch?: number }).batch, batch);
      assert.ok(!('maxMs' in plain.request), `${plain.name}: maxMs`);
      assert.equal((capped.request as { maxMs?: number }).maxMs, NEVER_MS);
    }
  }
});

test('kom1195 goldens: g0p88 batch-8 goldens with and without maxMs differ (N6)', () => {
  const plain = BY_NAME.get(parityGoldenName('g0p88', 8, false));
  const capped = BY_NAME.get(parityGoldenName('g0p88', 8, true));
  assert.ok(plain !== undefined && capped !== undefined, 'g0p88 batch-8 goldens missing');
  assert.notEqual(JSON.stringify(plain.reply), JSON.stringify(capped.reply));
});

test('kom1195 AC1: every golden reply equals a fresh run byte-for-byte (ms zeroed)', async () => {
  for (const golden of GOLDENS) {
    const reply = await runGolden(golden.engine, golden.request);
    assert.equal(JSON.stringify(reply), JSON.stringify(golden.reply), golden.name);
  }
});
