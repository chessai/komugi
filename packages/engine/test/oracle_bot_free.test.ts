// KOM-1191: ligungi bot game records must not appear in any public
// artifact, including the ligungi oracle fixture or its derived fixtures
// (tickets/KOM-1191.md). This file asserts the committed fixtures stay
// clean and that the stripping tool is idempotent on what is committed.

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';

import { stripBotRecords } from '../tools/strip_bot_records.mjs';
import type { OracleRecord } from './oracle_helpers.ts';
import { loadOracleFixture } from './oracle_helpers.ts';

const ORACLE_FIXTURE_PATH = fileURLToPath(new URL('./fixtures/ligungi_oracle.json', import.meta.url));
const EXTRA_FIXTURE_PATH = fileURLToPath(new URL('./fixtures/ligungi_oracle_extra.json', import.meta.url));
const SEEDED_GAMES_FIXTURE_PATH = fileURLToPath(new URL('./fixtures/ligungi_seeded_games.json', import.meta.url));

test('ligungi_oracle.json: no record comes from a bot-match game', () => {
  const fixture = loadOracleFixture();
  for (const record of fixture.records) {
    const id = record.id;
    const source = (record as unknown as { source: string }).source;
    assert.equal(id.startsWith('match-'), false, `${id}: id starts with match-`);
    assert.equal(id.includes('/match-'), false, `${id}: id contains /match-`);
    assert.equal(source.startsWith('match-'), false, `${id}: source starts with match-`);
    assert.equal(source.includes('/match-'), false, `${id}: source contains /match-`);
  }
});

test("ligungi_oracle_extra.json: 26 targeted marshal-reveal records are loaded", () => {
  const fixture = loadOracleFixture();
  const extraFileRecords = (
    JSON.parse(readFileSync(EXTRA_FIXTURE_PATH, 'utf8')) as { records: OracleRecord[] }
  ).records;
  const byId = new Map(extraFileRecords.map((r) => [r.id, r]));

  const loaded = fixture.records.filter((r) => (r as unknown as { source: string }).source.startsWith('targeted-marshal-reveal/'));
  assert.equal(loaded.length, 26);
  assert.equal(extraFileRecords.length, 26);

  for (const record of loaded) {
    assert.equal(record.kind, 'position');
    assert.ok(record.classes.includes('self_exposure'), `${record.id}: expected class self_exposure`);
    const want = byId.get(record.id);
    assert.ok(want, `${record.id}: no matching entry in ligungi_oracle_extra.json`);
    assert.deepEqual(record, want);
  }
});

test('loaded oracle fixture: record ids are unique and sorted', () => {
  const fixture = loadOracleFixture();
  const ids = fixture.records.map((r) => r.id);
  const seen = new Set<string>();
  for (const id of ids) {
    assert.equal(seen.has(id), false, `duplicate record id: ${id}`);
    seen.add(id);
  }
  const sorted = [...ids].sort((a, b) => (a < b ? -1 : a > b ? 1 : 0));
  assert.deepEqual(ids, sorted);
});

test('ligungi_oracle_extra.json: no record comes from a bot-match game', () => {
  const extraFileRecords = (
    JSON.parse(readFileSync(EXTRA_FIXTURE_PATH, 'utf8')) as { records: OracleRecord[] }
  ).records;
  for (const record of extraFileRecords) {
    const id = record.id;
    const source = (record as unknown as { source: string }).source;
    assert.equal(id.startsWith('match-'), false, `${id}: id starts with match-`);
    assert.equal(id.includes('/match-'), false, `${id}: id contains /match-`);
    assert.equal(source.startsWith('match-'), false, `${id}: source starts with match-`);
    assert.equal(source.includes('/match-'), false, `${id}: source contains /match-`);
  }
});

test('strip_bot_records is idempotent on the committed fixture', () => {
  const text = readFileSync(ORACLE_FIXTURE_PATH, 'utf8');
  assert.equal(stripBotRecords(text), text);
});

test('ligungi_seeded_games.json: seeded no-bot games', () => {
  const fixture = JSON.parse(readFileSync(SEEDED_GAMES_FIXTURE_PATH, 'utf8')) as {
    schema: string;
    method: string;
    games: Record<string, { plies: string[]; ligungiTerminal: { winner: string | null } }>;
  };

  assert.equal(fixture.schema, 'komugi-ligungi-seeded-games/1');
  assert.ok(fixture.method.includes('no bot'), `method should mention 'no bot': ${fixture.method}`);

  const games = Object.values(fixture.games);
  assert.equal(games.length, 2);
  for (const g of games) {
    assert.ok(g.plies.length > 0, 'expected a non-empty plies array');
    assert.notEqual(g.ligungiTerminal.winner, null);
  }
});
