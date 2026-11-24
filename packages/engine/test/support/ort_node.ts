// The Node-only onnxruntime-web vendor-path loader (design doc section 2.7,
// section 6 U8; moved out of src/ entirely per
// docs/design/ts-engine-library.md section 4.1a, review F1): it exists only
// so the package's own Node-side tests and dev tools can load the vendored
// onnxruntime-web Node build without a real npm install -- no browser
// entry has any legitimate reason to reach it, so it lives under test/
// rather than src/.
//
// Loads onnxruntime-web the way the ORT benchmark harness does (KOM-1056
// SCOPE): a dynamic `import()` of the vendored Node ESM
// entry point under ORTWEB_DIR -- no npm, no network, no package.json here.

import type { OrtModule, OrtSession } from '../../src/ort_core.ts';

// KOM-1103: no node types here (this module is reused by src/-adjacent
// tooling that also ships to Node only, but keeps the same globalThis-based
// read as the browser-safe modules it loads sessions for), so the Node
// global is read through `globalThis` with a typed guard -- `undefined` in
// a browser, the real env var under the `$NODE --test` harness, exactly as
// `process.env.ORTWEB_DIR` read before.
function nodeEnv(name: string): string | undefined {
  const proc = (globalThis as { process?: { env?: Record<string, string | undefined> } }).process;
  return proc?.env?.[name];
}

// The package root (packages/engine), resolved from this module's own URL:
// a plain checkout that has run `npm install onnxruntime-web` there holds
// node_modules/onnxruntime-web under it.
function packageRootDir(): string {
  return decodeURIComponent(new URL('../../', import.meta.url).pathname).replace(/\/$/, '');
}

// ORTWEB_DIR names a directory holding node_modules/onnxruntime-web (the
// vendor layout of KOM-1056); unset, the package root's own node_modules.
export function defaultOrtwebDir(): string {
  return nodeEnv('ORTWEB_DIR') || packageRootDir();
}

// ortwebEntry (used by the ORT benchmark script): the Node ESM entry point
// onnxruntime-web ships under its own vendored node_modules.
export function ortwebEntryPath(ortwebDir: string): string {
  return `${ortwebDir}/node_modules/onnxruntime-web/dist/ort.node.min.mjs`;
}

// Loads onnxruntime-web and builds one InferenceSession on the wasm EP,
// numThreads set BEFORE creation (design 2.7, KOM-1056's own order).
export async function loadOrtSession(
  modelPath: string,
  opts: { ortwebDir?: string; threads?: number } = {},
): Promise<{ ort: OrtModule; session: OrtSession }> {
  const ortwebDir = opts.ortwebDir ?? defaultOrtwebDir();
  const entryPath = ortwebEntryPath(ortwebDir);
  let ort: OrtModule;
  try {
    ort = (await import(entryPath)) as OrtModule;
  } catch (err) {
    // Never a silent skip: a missing onnxruntime-web is a named failure.
    throw new Error(
      `cannot load onnxruntime-web from ${entryPath}: set ORTWEB_DIR to a directory ` +
        `holding node_modules/onnxruntime-web, or run \`npm install onnxruntime-web\` in ` +
        `packages/engine (${String(err)})`,
    );
  }
  ort.env.wasm.numThreads = opts.threads ?? 1;
  const session = await ort.InferenceSession.create(modelPath, { executionProviders: ['wasm'] });
  return { ort, session };
}
