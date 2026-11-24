// AC2 (design doc docs/design/ts-engine-library.md section 7.1/9 unit 6):
// `npm pack`'s file list is exactly the public surface -- every required
// entry is present, nothing unexpected is, and every `exports` target
// package.json promises actually ships.

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { npmPack } from './support.ts';

const HERE = dirname(fileURLToPath(import.meta.url));
const ENGINE_DIR = resolve(HERE, '..', '..');

const REQUIRED_ENTRIES = [
  'dist/index.js',
  'dist/index.d.ts',
  'dist/rules.js',
  'dist/rules.d.ts',
  'dist/worker.js',
  'dist/worker.d.ts',
  'dist/client.js',
  'dist/client.d.ts',
  'README.md',
  'LICENSE',
  'NOTICE',
  'package.json',
  'docs/integration.md',
];

const TOP_LEVEL_ALLOWED = new Set(['package.json', 'README.md', 'LICENSE', 'NOTICE']);
// The assets CLI ships uncompiled under bin/ (package.json "bin").
const BIN_ALLOWED = new Set(['bin/komugi-assets.mjs']);
// The integration guide ships uncompiled under docs/ (package.json "files").
const DOCS_ALLOWED = new Set(['docs/integration.md']);

function isAllowedEntry(path: string): boolean {
  if (TOP_LEVEL_ALLOWED.has(path)) return true;
  if (BIN_ALLOWED.has(path)) return true;
  if (DOCS_ALLOWED.has(path)) return true;
  return path.startsWith('dist/');
}

function isDisallowedExtra(path: string): boolean {
  if (path.endsWith('.d.ts')) return false;
  if (path.endsWith('.map')) return true;
  if (path.endsWith('.ts')) return true;
  return false;
}

test('npm pack ships exactly the public surface (AC2)', () => {
  const pack = npmPack(ENGINE_DIR);
  try {
    const fileList = pack.result.files.map((f) => f.path);

    for (const required of REQUIRED_ENTRIES) {
      assert.ok(fileList.includes(required), `npm pack is missing required entry ${required}`);
    }

    for (const path of fileList) {
      assert.ok(isAllowedEntry(path), `npm pack ships unexpected entry outside dist/: ${path}`);
      assert.ok(!isDisallowedExtra(path), `npm pack ships a .map or non-.d.ts .ts file: ${path}`);
    }

    const pkg = JSON.parse(readFileSync(join(ENGINE_DIR, 'package.json'), 'utf8')) as {
      exports: Record<string, string | { types?: string; default?: string }>;
    };
    for (const [subpath, target] of Object.entries(pkg.exports)) {
      if (typeof target === 'string') {
        const relPath = target.replace(/^\.\//, '');
        assert.ok(fileList.includes(relPath), `exports[${subpath}] target ${relPath} is not in npm pack's file list`);
        continue;
      }
      for (const key of ['types', 'default'] as const) {
        const value = target[key];
        if (!value) continue;
        const relPath = value.replace(/^\.\//, '');
        assert.ok(
          fileList.includes(relPath),
          `exports[${subpath}].${key} target ${relPath} is not in npm pack's file list`,
        );
      }
    }
  } finally {
    pack.cleanup();
  }
});
