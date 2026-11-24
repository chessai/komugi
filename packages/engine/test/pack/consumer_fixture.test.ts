// F5 (design doc docs/design/ts-engine-library.md section 6.3/9 unit 6):
// a real `npm pack` installed into a throwaway fixture project under a
// consumer's own strict tsconfig (ligungi's flags, but with skipLibCheck
// false so every shipped .d.ts is actually checked) must typecheck and
// import cleanly through package specifiers only -- never a dist/ path.

import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { npmPack } from './support.ts';

const HERE = dirname(fileURLToPath(import.meta.url));
const ENGINE_DIR = resolve(HERE, '..', '..');

// Ligungi's tsconfig.base.json consumer flags, inlined (design 6.3), except
// skipLibCheck is flipped to false -- with it true every .d.ts the package
// ships is skipped and this whole test would be vacuous.
const FIXTURE_TSCONFIG_BASE = {
  target: 'ES2022',
  module: 'ESNext',
  moduleResolution: 'Bundler',
  strict: true,
  noUncheckedIndexedAccess: true,
  exactOptionalPropertyTypes: true,
  noImplicitOverride: true,
  isolatedModules: true,
  verbatimModuleSyntax: true,
  declaration: true,
  skipLibCheck: false,
  lib: ['ES2022', 'DOM'],
  noEmit: true,
  types: [],
};

const FIXTURE_INDEX_TS = `import { WorkerEngine } from '@komugi-ai/engine';
import type { WorkerRequest } from '@komugi-ai/engine';
import { ADVANCED_POSITION } from '@komugi-ai/engine/rules';
import type * as WorkerEntry from '@komugi-ai/engine/worker';

const engine = new WorkerEngine();
const request: WorkerRequest = { id: 1, fen: ADVANCED_POSITION, sims: 1 };
engine.handle(request);

// the worker entry exports no values -- used only in a type position, to
// prove \`@komugi-ai/engine/worker\`'s types resolve through the package
// specifier (importing it at runtime installs nothing: see consumer import
// check below).
type WorkerEntryModule = typeof WorkerEntry;
const workerEntryCheck: WorkerEntryModule = {};
void workerEntryCheck;
`;

// Negative self-test (proves skipLibCheck:false + real resolution are live,
// not a vacuous green): importing a name the package does not export must
// make tsc exit non-zero.
const FIXTURE_BAD_TS = `import { ThisNameDoesNotExistInTheEngine } from '@komugi-ai/engine';

export const unused = ThisNameDoesNotExistInTheEngine;
`;

function writeJson(path: string, value: unknown): void {
  writeFileSync(path, JSON.stringify(value, null, 2) + '\n');
}

test('a packed packages/engine typechecks and imports cleanly under a strict consumer tsconfig (F5)', () => {
  const tscPath = process.env.KOM_TSC;
  if (!tscPath) {
    throw new Error(
      'KOM_TSC env var is not set -- it must point at the TypeScript compiler\'s JS entry ' +
        '(tsc/bin/tsc) for the consumer-fixture test to run the compiler on the fixture project',
    );
  }

  const pack = npmPack(ENGINE_DIR);
  const fixtureDir = mkdtempSync(join(tmpdir(), 'komugi-consumer-fixture-'));
  const installCacheDir = mkdtempSync(join(tmpdir(), 'komugi-consumer-fixture-npm-cache-'));
  try {
    writeJson(join(fixtureDir, 'package.json'), { name: 'fixture', private: true, type: 'module' });
    writeJson(join(fixtureDir, 'tsconfig.json'), {
      compilerOptions: FIXTURE_TSCONFIG_BASE,
      include: ['index.ts'],
    });
    writeJson(join(fixtureDir, 'tsconfig.bad.json'), {
      extends: './tsconfig.json',
      include: ['bad.ts'],
    });
    writeFileSync(join(fixtureDir, 'index.ts'), FIXTURE_INDEX_TS);
    writeFileSync(join(fixtureDir, 'bad.ts'), FIXTURE_BAD_TS);

    const install = spawnSync(
      'npm',
      ['install', '--offline', '--no-package-lock', '--no-audit', '--no-fund', pack.tarballPath],
      {
        cwd: fixtureDir,
        env: { ...process.env, npm_config_cache: installCacheDir },
        encoding: 'utf8',
      },
    );
    assert.equal(
      install.status,
      0,
      `npm install of the packed tarball failed:\n${install.stdout}${install.stderr}`,
    );

    const positive = spawnSync(process.execPath, [tscPath, '-p', join(fixtureDir, 'tsconfig.json')], {
      cwd: fixtureDir,
      encoding: 'utf8',
    });
    assert.equal(
      positive.status,
      0,
      `tsc failed on the consumer fixture (ligungi flags, skipLibCheck:false):\n${positive.stdout}${positive.stderr}`,
    );

    const negative = spawnSync(process.execPath, [tscPath, '-p', join(fixtureDir, 'tsconfig.bad.json')], {
      cwd: fixtureDir,
      encoding: 'utf8',
    });
    assert.notEqual(
      negative.status,
      0,
      'tsc did not reject an import of a name @komugi-ai/engine does not export -- ' +
        'skipLibCheck/resolution is not actually live in this fixture',
    );

    const importScript = [
      "await import('@komugi-ai/engine');",
      "await import('@komugi-ai/engine/rules');",
      "await import('@komugi-ai/engine/worker');",
    ].join('\n');
    const runtimeCheck = spawnSync(process.execPath, ['--input-type=module', '-e', importScript], {
      cwd: fixtureDir,
      encoding: 'utf8',
    });
    assert.equal(
      runtimeCheck.status,
      0,
      `Node failed to import the installed package at runtime:\n${runtimeCheck.stdout}${runtimeCheck.stderr}`,
    );
  } finally {
    rmSync(fixtureDir, { recursive: true, force: true });
    rmSync(installCacheDir, { recursive: true, force: true });
    pack.cleanup();
  }
});
