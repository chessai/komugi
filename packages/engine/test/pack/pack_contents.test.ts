// F9 (design doc docs/design/ts-engine-library.md section 7.1/9 unit 6):
// a real `npm pack`, untarred, with every file's path and contents checked
// against packages/engine/test/pack_denylist.json -- plus a self-test
// proving the denylist config is actually live (a planted `KOM-123`
// reference must turn the scan red).

import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import {
  cpSync,
  mkdtempSync,
  readdirSync,
  readFileSync,
  rmSync,
  statSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, relative, resolve } from 'node:path';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { npmPack } from './support.ts';

const HERE = dirname(fileURLToPath(import.meta.url));
const ENGINE_DIR = resolve(HERE, '..', '..');
const DENYLIST_PATH = resolve(ENGINE_DIR, 'test', 'pack_denylist.json');

interface Denylist {
  terms: string[];
  path_patterns: string[];
  content_patterns: string[];
}

interface Violation {
  relPath: string;
  rule: 'term' | 'path-pattern' | 'content-pattern';
  matched: string;
}

function listFiles(dir: string): string[] {
  const out: string[] = [];
  for (const name of readdirSync(dir)) {
    const full = join(dir, name);
    if (statSync(full).isDirectory()) {
      out.push(...listFiles(full));
    } else {
      out.push(full);
    }
  }
  return out;
}

function scanTree(dir: string, denylist: Denylist): Violation[] {
  const violations: Violation[] = [];
  const pathPatterns = denylist.path_patterns.map((p) => new RegExp(p));
  const contentPatterns = denylist.content_patterns.map((p) => new RegExp(p));

  for (const absPath of listFiles(dir)) {
    const relPath = relative(dir, absPath).split('\\').join('/');
    const content = readFileSync(absPath, 'utf8');

    for (const term of denylist.terms) {
      if (relPath.includes(term)) violations.push({ relPath, rule: 'term', matched: term });
      if (content.includes(term)) violations.push({ relPath, rule: 'term', matched: term });
    }
    for (const pattern of pathPatterns) {
      if (pattern.test(relPath)) violations.push({ relPath, rule: 'path-pattern', matched: pattern.source });
    }
    for (const pattern of contentPatterns) {
      if (pattern.test(content)) violations.push({ relPath, rule: 'content-pattern', matched: pattern.source });
    }
  }
  return violations;
}

test('a packed packages/engine has no file whose path or contents match pack_denylist.json (F9)', () => {
  const denylist = JSON.parse(readFileSync(DENYLIST_PATH, 'utf8')) as Denylist;
  const pack = npmPack(ENGINE_DIR);
  const untarDir = mkdtempSync(join(tmpdir(), 'komugi-pack-untar-'));
  try {
    execFileSync('tar', ['-xzf', pack.tarballPath, '-C', untarDir]);
    const packageDir = join(untarDir, 'package');

    const violations = scanTree(packageDir, denylist);
    assert.deepEqual(
      violations,
      [],
      `pack_denylist.json found violations:\n${violations.map((v) => `${v.relPath}: ${v.rule} ${v.matched}`).join('\n')}`,
    );

    // Self-test (proves the denylist config is actually live, not vacuous):
    // copy the unpacked tree, plant a file containing KOM-123, and assert the
    // scan now fails.
    const plantedDir = mkdtempSync(join(tmpdir(), 'komugi-pack-planted-'));
    try {
      cpSync(packageDir, plantedDir, { recursive: true });
      writeFileSync(join(plantedDir, 'planted-ticket-ref.txt'), 'see KOM-123 for context\n');

      const plantedViolations = scanTree(plantedDir, denylist);
      assert.ok(
        plantedViolations.some((v) => v.relPath === 'planted-ticket-ref.txt' && v.rule === 'content-pattern'),
        'pack_denylist.json did not flag a planted KOM-123 reference -- the scan is not live',
      );
    } finally {
      rmSync(plantedDir, { recursive: true, force: true });
    }
  } finally {
    rmSync(untarDir, { recursive: true, force: true });
    pack.cleanup();
  }
});
