// Fixture loading for the parity tests (docs/design/clientside-engine-port.md section 4.1).
// docs/design/kom1196-golden-fixtures.md section 5: by default every loader
// reads the committed gzip sample under test/fixtures/; with
// KOM_FULL_REGRESSION=1 it reads the per-run (uncompressed) fixture named by
// its KOM_* variable instead, and setting any KOM_* without the flag throws.

import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { gunzipSync } from 'node:zlib';

const FIXTURES_DIR = fileURLToPath(new URL('./fixtures/', import.meta.url));

const _rawFlag = process.env.KOM_FULL_REGRESSION;
if (_rawFlag !== undefined && _rawFlag !== '1') {
  // N11: a misspelled value (e.g. "true") must not silently fall back to
  // the committed sample.
  throw new Error(`KOM_FULL_REGRESSION must be "1" if set, got ${JSON.stringify(_rawFlag)}`);
}
export const FULL_REGRESSION = _rawFlag === '1';
export const KOM_FIXTURE_VARS = [
  'KOM_FIXTURE', 'KOM_MOVES', 'KOM_SEARCH', 'KOM_PAIRWISE', 'KOM_RULES',
  'KOM_MODEL', 'KOM_ONNX_FIXTURE',
] as const;

// N3/N4: shared by every group E/F skip guard and every child-spawning test,
// so none of them can bypass the stray-variable check or leak KOM_* into a
// child's environment.
export function assertNoStrayKomVars(): void {
  if (FULL_REGRESSION) return;
  const stray = KOM_FIXTURE_VARS.filter((v) => process.env[v] !== undefined);
  if (stray.length > 0) {
    throw new Error(`${stray.join(',')} set without KOM_FULL_REGRESSION=1`);
  }
}

export function cleanEnv(overrides: Record<string, string | undefined> = {}): Record<string, string | undefined> {
  const env: Record<string, string | undefined> = { ...process.env };
  delete env.KOM_FULL_REGRESSION;
  for (const v of KOM_FIXTURE_VARS) delete env[v];
  // KOM_ORT_WEB_DIR is read standalone by cli_assets.test.ts, outside
  // KOM_FIXTURE_VARS/assertNoStrayKomVars, but must still be stripped here so
  // a no-python child run's skip of that case stays deterministic.
  delete env.KOM_ORT_WEB_DIR;
  return { ...env, ...overrides };
}

const loggedSources = new Set<string>();

// Flag unset: committed `fixtures/<committedName>`; any KOM_* set throws.
// Flag set: the variable's path; unset/empty throws.
function resolveFixturePath(envName: string, committedName: string): string {
  const p = process.env[envName];
  let path: string;
  if (FULL_REGRESSION) {
    if (!p) throw new Error(`KOM_FULL_REGRESSION=1 but ${envName} is not set`);
    path = p;
  } else {
    assertNoStrayKomVars();
    path = join(FIXTURES_DIR, committedName);
  }
  if (!loggedSources.has(envName)) {
    loggedSources.add(envName);
    console.log(`fixture ${envName}: source=${FULL_REGRESSION ? 'env' : 'committed'} ${path}`);
  }
  return path;
}

function readText(path: string): string {
  const bytes = readFileSync(path);
  return (path.endsWith('.gz') ? gunzipSync(bytes) : bytes).toString('utf8');
}

export type FixtureRecord = {
  id: string;
  fen: string;
  history: string[];
  phase: string;
  to_move: string;
  legal: number[];
  legal_san: string[];
  obs_sha256: string;
  obs_shape: number[];
  terminal: boolean;
  win_state: number[] | null;
  repetition: number;
};

export type MoveRecord = {
  id: number;
  move_type: number;
  piece: number;
  color: number;
  from_r: number;
  from_f: number;
  from_tier: number;
  to_r: number;
  to_f: number;
  to_tier: number;
  captured: number[][];
  draft_finished: boolean;
  san: string;
};

export type MovePosition = {
  id: string;
  game: number;
  ply: number;
  fen: string;
  sampled: boolean;
  draft_dense: boolean;
  terminal: boolean;
  in_check_w: boolean;
  in_check_b: boolean;
  moves: MoveRecord[];
};

export function parseJsonl<T>(text: string): T[] {
  const out: T[] = [];
  for (const line of text.split('\n')) {
    if (line.trim() !== '') out.push(JSON.parse(line) as T);
  }
  return out;
}

export function loadJsonl<T>(path: string): T[] {
  return parseJsonl<T>(readText(path));
}

// KOM_FIXTURE: the recorded parity fixture's output.
export function loadFixture(): FixtureRecord[] {
  return parseJsonl<FixtureRecord>(readText(resolveFixturePath('KOM_FIXTURE', 'parity.jsonl.gz')));
}

// KOM_MOVES: the recorded move-record fixture's output.
export function loadMoveRecords(): MovePosition[] {
  return parseJsonl<MovePosition>(readText(resolveFixturePath('KOM_MOVES', 'move_records.jsonl.gz')));
}

// KOM_SEARCH: the recorded search fixture's --out output. Generic: each suite
// passes its own record type (fixture.ts defines none).
export function loadSearchFixture<T>(): T {
  return JSON.parse(readText(resolveFixturePath('KOM_SEARCH', 'search.json.gz'))) as T;
}

// KOM_PAIRWISE: the recorded search fixture's --pairwise-vectors output.
export function loadPairwiseFixture<T>(): T {
  return JSON.parse(readText(resolveFixturePath('KOM_PAIRWISE', 'pairwise.json.gz'))) as T;
}

// KOM_RULES: the recorded rules-case fixture's --out output.
export function loadRulesCases<T>(): T {
  return JSON.parse(readText(resolveFixturePath('KOM_RULES', 'rules_cases.json.gz'))) as T;
}

export function sha256Hex(bytes: Uint8Array | ArrayBufferView): string {
  const view = new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  return createHash('sha256').update(view).digest('hex');
}
