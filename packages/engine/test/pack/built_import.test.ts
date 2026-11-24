// AC3 (design doc docs/design/ts-engine-library.md section 7.1/9 unit 6):
// the build (tsc -p tsconfig.json) must not drop or mis-rewrite anything on
// the default handle() path, and the public export surface of the built
// dist/ files must match the TS sources it was built from.

import assert from 'node:assert/strict';
import { dirname, join, resolve } from 'node:path';
import { test } from 'node:test';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { requireDistBuilt } from './support.ts';

const HERE = dirname(fileURLToPath(import.meta.url));
const ENGINE_DIR = resolve(HERE, '..', '..');
const DIST_DIR = join(ENGINE_DIR, 'dist');
const SRC_DIR = join(ENGINE_DIR, 'src');

interface WorkerEngineLike {
  handle: (request: { id: number; fen: string; sims: number }) => unknown;
}
type WorkerEngineCtor = new () => WorkerEngineLike;

// `ms` is the only wall-clock field a default handle() reply carries
// (protocol.ts's WorkerStats.ms, from Date.now() -- see engine.ts); every
// other field in a fake-evaluator, timing:false reply is deterministic.
function stripTiming(reply: unknown): unknown {
  const clone = JSON.parse(JSON.stringify(reply)) as { stats?: { ms?: unknown } };
  if (clone.stats && 'ms' in clone.stats) delete clone.stats.ms;
  return clone;
}

test('built dist/index.js and src/index.ts produce identical replies on the default handle() path (AC3)', async () => {
  requireDistBuilt(ENGINE_DIR);

  const distIndex = (await import(pathToFileURL(join(DIST_DIR, 'index.js')).href)) as Record<string, unknown>;
  const srcIndex = (await import(pathToFileURL(join(SRC_DIR, 'index.ts')).href)) as Record<string, unknown>;

  const distFen = distIndex.ADVANCED_POSITION;
  const srcFen = srcIndex.ADVANCED_POSITION;
  assert.equal(typeof distFen, 'string', 'dist/index.js does not export ADVANCED_POSITION');
  assert.equal(distFen, srcFen, 'ADVANCED_POSITION differs between dist/index.js and src/index.ts');

  const DistWorkerEngine = distIndex.WorkerEngine as WorkerEngineCtor;
  const SrcWorkerEngine = srcIndex.WorkerEngine as WorkerEngineCtor;

  const distReply = new DistWorkerEngine().handle({ id: 1, fen: distFen as string, sims: 8 });
  const srcReply = new SrcWorkerEngine().handle({ id: 1, fen: srcFen as string, sims: 8 });

  assert.deepEqual(
    stripTiming(distReply),
    stripTiming(srcReply),
    `dist/index.js and src/index.ts replies differ once timing fields are removed:\n` +
      `dist=${JSON.stringify(distReply)}\nsrc=${JSON.stringify(srcReply)}`,
  );

  const distRules = (await import(pathToFileURL(join(DIST_DIR, 'rules.js')).href)) as Record<string, unknown>;
  const srcRules = (await import(pathToFileURL(join(SRC_DIR, 'rules.ts')).href)) as Record<string, unknown>;
  assert.deepEqual(
    Object.keys(distRules).sort(),
    Object.keys(srcRules).sort(),
    'dist/rules.js exports different names than src/rules.ts',
  );

  assert.deepEqual(
    Object.keys(distIndex).sort(),
    Object.keys(srcIndex).sort(),
    'dist/index.js exports different names than src/index.ts',
  );
});
