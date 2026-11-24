// KOM-1196 section 7 (F9): invariants over the five committed golden
// fixtures under test/fixtures/ -- the manifest's file set and hashes, the
// .gz size budget, no \r in any uncompressed fixture, coverage invariants
// strong enough that a parity set with no terminal record would fail, and
// that no committed fixture carries a recorded-game id (F2).

import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { gunzipSync, gzipSync } from 'node:zlib';

const FIXTURES_DIR = fileURLToPath(new URL('./fixtures/', import.meta.url));

const EXPECTED_NAMES = [
  'parity.jsonl',
  'move_records.jsonl',
  'search.json',
  'pairwise.json',
  'rules_cases.json',
] as const;

//: section 3.3's budget: the NEW .gz set alone, not combined with the
// pre-existing committed fixtures (open item for the advisor, section 9).
const NEW_FIXTURE_SIZE_BUDGET = 2_097_152;
//: ligungi_oracle.json + observation_g3_modes.json + ligungi_oracle_extra.json
// + ligungi_seeded_games.json, measured once (section 3.3) -- printed
// alongside the new total, not asserted against.
const EXISTING_FIXTURE_BYTES = 4_079_034;

const FORBIDDEN_RECORDED_IDS = ['kom239_b0_g4', 'kom239_b0_g6', 'kom239_b0_g10'];

function sha256(data: Buffer): string {
  return createHash('sha256').update(data).digest('hex');
}

function parseManifest(text: string): Map<string, string> {
  const out = new Map<string, string>();
  for (const line of text.split('\n')) {
    if (!line || line.startsWith('#')) continue;
    const idx = line.indexOf('  ');
    if (idx === -1) continue;
    out.set(line.slice(idx + 2), line.slice(0, idx));
  }
  return out;
}

function readUncompressed(dir: string, name: string): Buffer {
  return gunzipSync(readFileSync(join(dir, `${name}.gz`)));
}

function checkManifestFileSet(manifest: Map<string, string>): string[] {
  const violations: string[] = [];
  const names = new Set(manifest.keys());
  for (const n of EXPECTED_NAMES) {
    if (!names.has(n)) violations.push(`manifest is missing ${n}`);
  }
  for (const n of names) {
    if (!(EXPECTED_NAMES as readonly string[]).includes(n)) violations.push(`manifest names unexpected ${n}`);
  }
  return violations;
}

function checkManifestHashes(dir: string, manifest: Map<string, string>): string[] {
  const violations: string[] = [];
  for (const [name, digest] of manifest) {
    const gzPath = join(dir, `${name}.gz`);
    if (!existsSync(gzPath)) {
      violations.push(`missing ${gzPath}`);
      continue;
    }
    const actual = sha256(readUncompressed(dir, name));
    if (actual !== digest) violations.push(`${name}: manifest sha ${digest} != actual ${actual}`);
  }
  return violations;
}

function checkSizeBudget(dir: string): { violations: string[]; totalNew: number } {
  let totalNew = 0;
  for (const name of EXPECTED_NAMES) {
    const gzPath = join(dir, `${name}.gz`);
    if (existsSync(gzPath)) totalNew += readFileSync(gzPath).length;
  }
  const violations =
    totalNew > NEW_FIXTURE_SIZE_BUDGET ? [`new .gz total ${totalNew} exceeds ${NEW_FIXTURE_SIZE_BUDGET}`] : [];
  return { violations, totalNew };
}

function checkNoCR(dir: string): string[] {
  const violations: string[] = [];
  for (const name of EXPECTED_NAMES) {
    const gzPath = join(dir, `${name}.gz`);
    if (!existsSync(gzPath)) continue;
    if (readUncompressed(dir, name).includes(0x0d)) violations.push(`${name}: uncompressed content contains \\r`);
  }
  return violations;
}

function checkNoRecordedIds(dir: string): string[] {
  const violations: string[] = [];
  for (const name of EXPECTED_NAMES) {
    const gzPath = join(dir, `${name}.gz`);
    if (!existsSync(gzPath)) continue;
    const text = readUncompressed(dir, name).toString('utf8');
    for (const id of FORBIDDEN_RECORDED_IDS) {
      if (text.includes(id)) violations.push(`${name} contains recorded-game id ${id}`);
    }
  }
  return violations;
}

type RulesCases = {
  kom239: { name: string; replays: Record<string, { repetition: number }[]> }[];
};

function parseJsonl(text: string): unknown[] {
  return text
    .split('\n')
    .filter((line) => line.trim().length > 0)
    .map((line) => JSON.parse(line));
}

function checkCoverage(dir: string): string[] {
  const violations: string[] = [];
  if (!existsSync(join(dir, 'parity.jsonl.gz'))) return violations;

  const parityRecords = parseJsonl(readUncompressed(dir, 'parity.jsonl').toString('utf8')) as Array<{
    terminal: boolean;
    phase: string;
  }>;
  if (!parityRecords.some((r) => r.terminal)) violations.push('parity: no terminal record');
  if (!parityRecords.some((r) => r.phase === 'draft')) violations.push('parity: no draft-phase record');
  if (parityRecords.filter((r) => r.phase === 'play').length < 5) {
    violations.push('parity: fewer than 5 play-phase records');
  }

  const moveRecords = parseJsonl(readUncompressed(dir, 'move_records.jsonl').toString('utf8')) as Array<{
    draft_dense: boolean;
    moves: Array<{ captured: unknown[]; draft_finished: boolean }>;
  }>;
  const draftDenseCount = moveRecords.filter((r) => r.draft_dense).length;
  if (draftDenseCount < 40) violations.push(`moves: only ${draftDenseCount} draft_dense positions (need >= 40)`);
  const allMoves = moveRecords.flatMap((r) => r.moves ?? []);
  if (!allMoves.some((m) => Array.isArray(m.captured) && m.captured.length > 0)) {
    violations.push('moves: no move with non-empty captured');
  }
  if (!allMoves.some((m) => m.draft_finished)) violations.push('moves: no move with draft_finished');

  const rules = JSON.parse(readUncompressed(dir, 'rules_cases.json').toString('utf8')) as RulesCases;
  let placementOnlyFourfold = false;
  let fullFourfold = false;
  for (const entry of rules.kom239) {
    const placement = entry.replays.placement ?? [];
    const full = entry.replays.full ?? [];
    for (let i = 0; i < placement.length; i++) {
      if (placement[i].repetition >= 4 && (full[i]?.repetition ?? 0) < 4) placementOnlyFourfold = true;
    }
    for (const p of full) {
      if (p.repetition >= 4) fullFourfold = true;
    }
  }
  if (!placementOnlyFourfold) violations.push('rules: no ply with PLACEMENT repetition >= 4 while FULL < 4');
  if (!fullFourfold) violations.push('rules: no ply with FULL repetition >= 4');

  return violations;
}

function withScratchCopy(mutate: (dir: string) => void): string {
  const dir = mkdtempSync(join(tmpdir(), 'kom1196-fixtures-'));
  for (const name of EXPECTED_NAMES) {
    writeFileSync(join(dir, `${name}.gz`), readFileSync(join(FIXTURES_DIR, `${name}.gz`)));
  }
  writeFileSync(join(dir, 'manifest.sha256'), readFileSync(join(FIXTURES_DIR, 'manifest.sha256')));
  mutate(dir);
  return dir;
}

test('fixtures_manifest: the committed set passes every invariant', () => {
  const manifestText = readFileSync(join(FIXTURES_DIR, 'manifest.sha256'), 'utf8');
  const manifest = parseManifest(manifestText);

  assert.deepEqual(checkManifestFileSet(manifest), []);
  assert.deepEqual(checkManifestHashes(FIXTURES_DIR, manifest), []);

  const { violations: sizeViolations, totalNew } = checkSizeBudget(FIXTURES_DIR);
  console.log(
    `fixtures_manifest: new .gz total=${totalNew} existing=${EXISTING_FIXTURE_BYTES} ` +
      `combined=${totalNew + EXISTING_FIXTURE_BYTES}`,
  );
  assert.deepEqual(sizeViolations, []);

  assert.deepEqual(checkNoCR(FIXTURES_DIR), []);
  assert.deepEqual(checkCoverage(FIXTURES_DIR), []);
  assert.deepEqual(checkNoRecordedIds(FIXTURES_DIR), []);
});

test('fixtures_manifest: dropping every terminal parity record fails coverage', () => {
  const dir = withScratchCopy((d) => {
    const records = (parseJsonl(readUncompressed(FIXTURES_DIR, 'parity.jsonl').toString('utf8')) as Array<{
      terminal: boolean;
    }>).filter((r) => !r.terminal);
    const text = `${records.map((r) => JSON.stringify(r)).join('\n')}\n`;
    writeFileSync(join(d, 'parity.jsonl.gz'), gzipSync(Buffer.from(text, 'utf8'), { level: 9 }));
  });
  try {
    assert.notEqual(checkCoverage(dir).length, 0, 'expected a coverage violation once every terminal record is gone');
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test('fixtures_manifest: planting a recorded-game id fails the denylist check', () => {
  const dir = withScratchCopy((d) => {
    const text = `${readUncompressed(FIXTURES_DIR, 'rules_cases.json').toString('utf8')}/* kom239_b0_g4 */`;
    writeFileSync(join(d, 'rules_cases.json.gz'), gzipSync(Buffer.from(text, 'utf8'), { level: 9 }));
  });
  try {
    assert.notEqual(checkNoRecordedIds(dir).length, 0, 'expected a denylist violation once the id is planted');
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test('fixtures_manifest: dropping a manifest line fails the file-set check', () => {
  const manifestText = readFileSync(join(FIXTURES_DIR, 'manifest.sha256'), 'utf8');
  const dropped = manifestText
    .split('\n')
    .filter((line) => !line.includes('search.json'))
    .join('\n');
  const manifest = parseManifest(dropped);
  assert.notEqual(checkManifestFileSet(manifest).length, 0, 'expected a file-set violation once a line is dropped');
});
