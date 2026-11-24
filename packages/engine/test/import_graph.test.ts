// AC7 (design doc docs/design/ts-engine-library.md section 7.1/3.3): the
// rules-only entry's import graph is exactly the stated 19-module closure --
// an exact-set assertion, not a subset or a denylist, so both an accidental
// new import and a module quietly dropping out of the closure fail this test.
// (KOM-1203 section 3/F6: rules_config.ts joined as the 17th module. KOM-1198
// unit 5 (docs/design/kom1198-move-io.md section 7): move_io.ts and
// position_builder.ts join as the 18th and 19th.)

import assert from 'node:assert/strict';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { scanText, walkImports } from './support/import_graph.ts';

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = resolve(HERE, '..', '..', '..');
const RULES_ENTRY = resolve(HERE, '..', 'src', 'rules.ts');
const INDEX_ENTRY = resolve(HERE, '..', 'src', 'index.ts');
const WORKER_ENTRY = resolve(HERE, '..', 'src', 'worker.ts');

const EXPECTED_MODULES = [
  'rules.ts',
  'board.ts',
  'constants.ts',
  'fen.ts',
  'movegen.ts',
  'movegen_board.ts',
  'movegen_arata.ts',
  'ligungi_filter.ts',
  'rules_ligungi.ts',
  'repetition.ts',
  'rules_config.ts',
  'apply.ts',
  'actions.ts',
  'san.ts',
  'game.ts',
  'terminal.ts',
  'root.ts',
  'move_io.ts',
  'position_builder.ts',
].map((name) => `packages/engine/src/${name}`).sort();

test('src/rules.ts reaches exactly the 19-module rules closure', () => {
  const result = walkImports(RULES_ENTRY);
  assert.deepEqual(result.modules, EXPECTED_MODULES);
});

test('src/rules.ts closure imports no node builtins', () => {
  const result = walkImports(RULES_ENTRY);
  assert.deepEqual(result.nodeBuiltins, []);
});

test('src/rules.ts closure has no non-literal dynamic import()', () => {
  const result = walkImports(RULES_ENTRY);
  assert.deepEqual(result.nonLiteralImports, []);
});

// AC6 (design doc section 7.1): the browser-reachable import graph from both
// src/index.ts and src/worker.ts has no node:/absolute-path/computed-
// import() escape, and its text (comments included) carries none of the
// Node-vendor-path patterns a specifier-only walk would miss.
const ALLOWED_PACKAGE_SPECIFIERS = ['onnxruntime-web/wasm', 'onnxruntime-web/webgpu'];

for (const [label, entry] of [
  ['src/index.ts', INDEX_ENTRY],
  ['src/worker.ts', WORKER_ENTRY],
] as const) {
  test(`${label} import graph has no node builtins`, () => {
    const result = walkImports(entry);
    assert.deepEqual(result.nodeBuiltins, []);
  });

  test(`${label} import graph has no non-literal dynamic import()`, () => {
    const result = walkImports(entry);
    assert.deepEqual(result.nonLiteralImports, []);
  });

  test(`${label} import graph has no absolute specifiers`, () => {
    const result = walkImports(entry);
    assert.deepEqual(result.absoluteSpecifiers, []);
  });

  test(`${label} import graph's package specifiers are a subset of the allowed ORT specifiers`, () => {
    const result = walkImports(entry);
    for (const spec of result.packageSpecifiers) {
      assert.ok(ALLOWED_PACKAGE_SPECIFIERS.includes(spec), `unexpected package specifier: ${spec}`);
    }
  });

  test(`${label} import graph's text scan reports nothing`, () => {
    const result = walkImports(entry);
    const findings = scanText(result.modules, REPO_ROOT);
    assert.deepEqual(findings, []);
  });
}

test('src/worker.ts reaches src/worker_host.ts and src/engine.ts (walk is not vacuous)', () => {
  const result = walkImports(WORKER_ENTRY);
  assert.ok(result.modules.includes('packages/engine/src/worker_host.ts'));
  assert.ok(result.modules.includes('packages/engine/src/engine.ts'));
});

test('src/index.ts reaches src/rules.ts (walk is not vacuous)', () => {
  const result = walkImports(INDEX_ENTRY);
  assert.ok(result.modules.includes('packages/engine/src/rules.ts'));
});

// Planted-fixture self-test (design doc section 7.1 fix 11): proves the text
// scan and the specifier classification themselves go red against a
// throwaway module tree, not just happen to pass against today's src/.
test('scanText and walkImports go red against a planted fixture', () => {
  const dir = mkdtempSync(join(tmpdir(), 'kom-1185-import-graph-'));
  try {
    const entryPath = join(dir, 'entry.ts');
    const bPath = join(dir, 'b.ts');
    writeFileSync(entryPath, "import './b.ts';\n", 'utf8');
    writeFileSync(
      bPath,
      [
        "// a comment mentioning import('ignored') must never be reported",
        "const variableName: string = './ignored.ts';",
        "import('node:fs');",
        'import(variableName);',
        "import('/abs/path.ts');",
        '',
        "export const HOME_PATH = '/home/someone/x';",
        'export function readEnv(): string | undefined {',
        '  return process.env.FOO;',
        '}',
        'export const DIR = ORTWEB_DIR;',
        'export const legacyProcess = (globalThis as any).process;',
        '',
      ].join('\n'),
      'utf8',
    );

    const result = walkImports(entryPath);
    assert.equal(result.nonLiteralImports.length, 1);
    assert.ok(result.nonLiteralImports[0]?.includes('variableName'));
    assert.ok(result.nodeBuiltins.includes('node:fs'));
    assert.ok(result.absoluteSpecifiers.includes('/abs/path.ts'));

    const findings = scanText(result.modules, dir);
    const joined = findings.join('\n');
    assert.match(joined, /\/home\//);
    assert.match(joined, /process\.env/);
    assert.match(joined, /ORTWEB_DIR/);
    assert.match(joined, /\.process/);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test('await import of src/index.ts exposes the worker-host/engine/backend surface and a rules export', async () => {
  const mod = (await import('../src/index.ts')) as Record<string, unknown>;
  assert.equal(typeof mod.installWorkerHost, 'function');
  assert.equal(typeof mod.WorkerEngine, 'function');
  assert.equal(typeof mod.makeOrtBackend, 'function');
  assert.equal(typeof mod.probeBackend, 'function');
  assert.equal(typeof mod.BackendError, 'function');
  assert.equal(typeof mod.applyMove, 'function');
});
