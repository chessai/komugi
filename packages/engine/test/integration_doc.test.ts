// KOM-1201 unit B (design docs/design/kom1201-asset-kit.md section 6 AC4):
// the integration guide mirrors AC3's header expectations, links the
// README's failure-codes section instead of copying it, stays
// self-contained (no docs/design/ link), states the bundler requirement,
// and carries the N1 path-resolution snippet verbatim. No network, no
// browser: this is a text-content check over the committed files.

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { DOCUMENTED_HEADERS, parseGuideHeaders } from './support/documented_headers.ts';

const HERE = dirname(fileURLToPath(import.meta.url));
const ENGINE_DIR = resolve(HERE, '..');
const GUIDE_PATH = join(ENGINE_DIR, 'docs', 'integration.md');
const README_PATH = join(ENGINE_DIR, 'README.md');

const guide = readFileSync(GUIDE_PATH, 'utf8');
const readme = readFileSync(README_PATH, 'utf8');

test('guide contains every documented header value, one assertion per value', () => {
  const required = [
    'Content-Security-Policy',
    'script-src',
    "'wasm-unsafe-eval'",
    'connect-src',
    'blob:',
    'application/wasm',
    'text/javascript',
    'application/octet-stream',
    'application/json',
    'Cache-Control',
    'no-cache',
    'Cross-Origin-Opener-Policy',
    'same-origin',
    'Cross-Origin-Embedder-Policy',
    'require-corp',
  ];
  for (const value of required) {
    assert.ok(guide.includes(value), `guide is missing documented value ${JSON.stringify(value)}`);
  }
});

test('guide links the README failure-codes section, not a copy of the list', () => {
  assert.ok(
    guide.includes('https://github.com/chessai/komugi/blob/main/packages/engine/README.md#failure-codes'),
    'guide does not link the README failure-codes anchor',
  );
});

test('guide is self-contained: no docs/design/ link', () => {
  assert.ok(!guide.includes('docs/design/'), 'guide links a design doc that is not part of the package');
});

test('guide states the bundler requirement', () => {
  assert.ok(guide.includes('a bundler is required'), 'guide is missing the bundler-required phrase');
});

test('guide carries the N1 path-resolution snippet verbatim', () => {
  assert.ok(guide.includes('new URL(p, ortAssetsUrl).href'), 'guide is missing the ortAssets path-resolution line');
});

test("guide's two header tables parse to exactly DOCUMENTED_HEADERS (N8: drift fails here)", () => {
  const parsed = parseGuideHeaders(guide);
  assert.deepEqual(parsed, DOCUMENTED_HEADERS);
});

test('neither the guide nor the README carries a ticket ID or a home-directory path', () => {
  for (const [name, text] of [
    ['docs/integration.md', guide],
    ['README.md', readme],
  ] as const) {
    assert.ok(!/KOM-[0-9]/.test(text), `${name} contains a ticket ID`);
    assert.ok(!text.includes('/home/'), `${name} contains a home-directory path`);
  }
});
