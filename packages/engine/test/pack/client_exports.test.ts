// KOM-1199 unit U4 (design docs/design/kom1199-client.md section 7): the
// `./client` subpath export exists in package.json and the built
// dist/client.js module is importable and starts nothing on its own --
// mirrors test/client.test.ts's "import starts nothing" (AC4) check, but
// against the BUILT file (the thing a real consumer actually imports), which
// is why this lives under test/pack/ alongside the other build-gated checks
// rather than test/ (only the pack gate builds dist/ first).

import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { test } from 'node:test';
import { fileURLToPath, pathToFileURL } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const ENGINE_DIR = resolve(HERE, '..', '..');
const DIST_CLIENT_JS = join(ENGINE_DIR, 'dist', 'client.js');

function requireClientBuilt(): void {
  if (!existsSync(DIST_CLIENT_JS)) {
    throw new Error(
      `${DIST_CLIENT_JS} is missing -- build packages/engine first ` +
        `(tsc -p packages/engine/tsconfig.json) before running the client exports test`,
    );
  }
}

test("package.json#exports['./client'] points at dist/client.js and dist/client.d.ts", () => {
  const pkg = JSON.parse(readFileSync(join(ENGINE_DIR, 'package.json'), 'utf8')) as {
    exports: Record<string, string | { types?: string; default?: string }>;
  };
  const target = pkg.exports['./client'];
  assert.ok(target && typeof target === 'object', "exports['./client'] is missing or not a conditions object");
  assert.equal((target as { default?: string }).default, './dist/client.js');
  assert.equal((target as { types?: string }).types, './dist/client.d.ts');
});

test('importing dist/client.js starts nothing (no stub factory call, no fetch)', async () => {
  requireClientBuilt();

  const originalFetch = globalThis.fetch;
  let fetchCalls = 0;
  let factoryCalls = 0;
  globalThis.fetch = (() => {
    fetchCalls += 1;
    return Promise.reject(new Error('no fetch in tests'));
  }) as typeof fetch;
  try {
    const mod = (await import(pathToFileURL(DIST_CLIENT_JS).href)) as {
      KomugiClient: new (options: unknown) => { state: string; dispose: () => void };
    };
    const client = new mod.KomugiClient({
      createWorker: () => {
        factoryCalls += 1;
        throw new Error('must not be called');
      },
      model: { url: 'model.onnx', sha256: 'sha-a' },
      ortAssets: { wasm: { mjs: 'ort.mjs', wasm: 'ort.wasm' } },
    });
    assert.equal(client.state, 'idle');
    assert.equal(factoryCalls, 0);
    assert.equal(fetchCalls, 0);
    client.dispose();
  } finally {
    globalThis.fetch = originalFetch;
  }
});
