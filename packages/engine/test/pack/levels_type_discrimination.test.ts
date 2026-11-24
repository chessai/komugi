// KOM-1202 unit 2a (design docs/design/kom1202-levels.md section 3, review
// V1): proves at the TYPE level that `WorkerRequest | WorkerLevelRequest`
// discriminates `level` from `sims`/`batch`/`maxMs` on BOTH sides of the
// union -- a positive fixture (a clean pure-`level` literal and a clean
// pure-`sims`/`batch`/`maxMs` literal) must typecheck, and a negative
// fixture (the mixed `{ level, sims }` literal) must not. Modeled on
// test/pack/consumer_fixture.test.ts's positive/negative tsc-spawn pair.
// Lives under test/pack/ (not test/) because it spawns the vendored
// compiler via KOM_TSC, which only the pack gate sets -- the full
// regression globs test/*.test.ts only. `tsconfig.strict.json`'s own
// `include` is `src/**/*.ts` and never widens to cover `test/**` (design
// section 7), so this fixture brings its own self-contained tsconfig
// instead.

import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, relative, resolve } from 'node:path';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const PROTOCOL_PATH = resolve(HERE, '..', '..', 'src', 'protocol.ts');

// `rewriteRelativeImportExtensions` (on below, matching tsconfig.json)
// requires a genuinely RELATIVE specifier for a `.ts` extension to be
// accepted (TS2877 otherwise) -- an absolute path does not qualify even
// though `noEmit` means nothing is ever rewritten.
function protocolSpecifierFrom(fixtureDir: string): string {
  const rel = relative(fixtureDir, PROTOCOL_PATH).replace(/\\/g, '/');
  return rel.startsWith('.') ? rel : `./${rel}`;
}

// `tsconfig.json`'s base compiler options plus `tsconfig.strict.json`'s own
// overrides, inlined (same reason test/pack/consumer_fixture.test.ts inlines
// a consumer tsconfig: this fixture needs its own self-contained project,
// pointed at tmp files only).
const FIXTURE_COMPILER_OPTIONS = {
  target: 'ES2022',
  module: 'ESNext',
  lib: ['ES2022', 'DOM', 'DOM.Iterable'],
  moduleResolution: 'bundler',
  rewriteRelativeImportExtensions: true,
  declaration: true,
  strict: true,
  skipLibCheck: true,
  verbatimModuleSyntax: true,
  noUncheckedSideEffectImports: true,
  noEmit: true,
  allowImportingTsExtensions: true,
  exactOptionalPropertyTypes: true,
  noUncheckedIndexedAccess: true,
  noUnusedLocals: true,
  noUnusedParameters: true,
  noFallthroughCasesInSwitch: true,
  noImplicitOverride: true,
  isolatedModules: true,
  types: [] as string[],
};

function goodFixtureSource(fixtureDir: string): string {
  return `import type { WorkerRequest, WorkerLevelRequest } from '${protocolSpecifierFrom(fixtureDir)}';

type Req = WorkerRequest | WorkerLevelRequest;

export const byLevel: Req = { id: 1, fen: 'fen', level: 'casual' };
export const byRaw: Req = { id: 2, fen: 'fen', sims: 100, batch: 4, maxMs: 1000 };
`;
}

function badFixtureSource(fixtureDir: string): string {
  return `import type { WorkerRequest, WorkerLevelRequest } from '${protocolSpecifierFrom(fixtureDir)}';

type Req = WorkerRequest | WorkerLevelRequest;

export const mixed: Req = { id: 1, fen: 'fen', level: 'casual', sims: 100 };
`;
}

function writeJson(path: string, value: unknown): void {
  writeFileSync(path, JSON.stringify(value, null, 2) + '\n');
}

test('WorkerRequest | WorkerLevelRequest discriminates level from sims/batch/maxMs at the type level (review V1)', () => {
  const tscPath = process.env.KOM_TSC;
  if (!tscPath) {
    throw new Error(
      'KOM_TSC env var is not set -- it must point at the TypeScript compiler\'s JS entry ' +
        '(tsc/bin/tsc) for the levels type-discrimination test to run the compiler on the fixture project',
    );
  }

  const fixtureDir = mkdtempSync(join(tmpdir(), 'komugi-levels-type-discrimination-'));
  try {
    writeFileSync(join(fixtureDir, 'good.ts'), goodFixtureSource(fixtureDir));
    writeFileSync(join(fixtureDir, 'bad.ts'), badFixtureSource(fixtureDir));
    writeJson(join(fixtureDir, 'tsconfig.good.json'), {
      compilerOptions: FIXTURE_COMPILER_OPTIONS,
      include: ['good.ts'],
    });
    writeJson(join(fixtureDir, 'tsconfig.bad.json'), {
      compilerOptions: FIXTURE_COMPILER_OPTIONS,
      include: ['bad.ts'],
    });

    const positive = spawnSync(process.execPath, [tscPath, '-p', join(fixtureDir, 'tsconfig.good.json')], {
      cwd: fixtureDir,
      encoding: 'utf8',
    });
    assert.equal(
      positive.status,
      0,
      `tsc rejected the pure-level/pure-raw fixture:\n${positive.stdout}${positive.stderr}`,
    );

    const negative = spawnSync(process.execPath, [tscPath, '-p', join(fixtureDir, 'tsconfig.bad.json')], {
      cwd: fixtureDir,
      encoding: 'utf8',
    });
    assert.notEqual(
      negative.status,
      0,
      'tsc accepted a mixed { level, sims } literal -- WorkerRequest | WorkerLevelRequest is not ' +
        'discriminated on both sides',
    );
  } finally {
    rmSync(fixtureDir, { recursive: true, force: true });
  }
});
