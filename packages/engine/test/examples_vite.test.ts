// examples/vite is a source example: it is type-checked against the built
// package types (a gate command, since it needs dist/), never built here.
// These checks pin it to the README and the package's exports map.
import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';

const ENGINE_DIR = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const EXAMPLE = join(ENGINE_DIR, 'examples', 'vite');
const read = (p: string): string => readFileSync(p, 'utf8');

function bundlerSnippet(): string {
  const readme = read(join(ENGINE_DIR, 'README.md'));
  const section = readme.split(/^## Bundler configuration$/m)[1];
  assert.ok(section !== undefined, 'README has a "Bundler configuration" section');
  const body = section.split(/^## /m)[0] as string;
  const match = /```ts\n([\s\S]*?)```/.exec(body);
  assert.ok(match, 'the Bundler configuration section has a ts code block');
  return match[1] as string;
}

test('vite.config.ts equals the README Bundler-configuration snippet character for character', () => {
  const snippet = bundlerSnippet();
  assert.ok(snippet.includes('onnxruntime-web-use-extern-wasm'));
  assert.equal(read(join(EXAMPLE, 'vite.config.ts')), snippet);
});

test('tsconfig paths map exactly the type-bearing exports, each to its own dist .d.ts, no wildcard', () => {
  const pkg = JSON.parse(read(join(ENGINE_DIR, 'package.json'))) as { exports: Record<string, { types?: string }> };
  const tsconfig = JSON.parse(read(join(EXAMPLE, 'tsconfig.json'))) as {
    compilerOptions: { paths: Record<string, string[]> };
    include: string[];
  };
  const paths = tsconfig.compilerOptions.paths;
  const exportKeys = Object.keys(pkg.exports).filter((k) => k !== './package.json');
  const expected = exportKeys.map((k) => (k === '.' ? '@komugi-ai/engine' : `@komugi-ai/engine/${k.slice(2)}`));
  assert.deepEqual(Object.keys(paths).sort(), expected.sort());
  for (const key of Object.keys(paths)) assert.ok(!key.includes('*'), `wildcard key ${key}`);
  for (const k of exportKeys) {
    const spec = k === '.' ? '@komugi-ai/engine' : `@komugi-ai/engine/${k.slice(2)}`;
    const types = pkg.exports[k]?.types;
    assert.ok(types !== undefined && types.startsWith('./'), `${k} has types`);
    assert.deepEqual(paths[spec], [`../../${types.slice(2)}`]);
  }
  assert.deepEqual(tsconfig.include, ['src/**/*.ts']);
});

test('README installs with --install-links and the example ships no .npmrc', () => {
  const readme = read(join(EXAMPLE, 'README.md'));
  assert.ok(readme.includes('npm install --install-links'));
  assert.doesNotMatch(readme, /\.npmrc/);
  assert.ok(!existsSync(join(EXAMPLE, '.npmrc')));
});

test('.gitignore lists node_modules/, public/ and dist/', () => {
  const lines = read(join(EXAMPLE, '.gitignore')).split('\n').map((l) => l.trim());
  for (const entry of ['node_modules/', 'public/', 'dist/']) assert.ok(lines.includes(entry), `missing ${entry}`);
});

test('package.json depends on the engine via file:../.. and pins vite and onnxruntime-web', () => {
  const pkg = JSON.parse(read(join(EXAMPLE, 'package.json'))) as {
    dependencies: Record<string, string>;
    devDependencies: Record<string, string>;
  };
  const all = { ...pkg.dependencies, ...pkg.devDependencies };
  assert.equal(all['@komugi-ai/engine'], 'file:../..');
  for (const name of ['vite', 'onnxruntime-web']) assert.match(all[name] ?? '', /^\d+\.\d+\.\d+$/, `${name} pinned`);
});

test('src/main.ts resolves ort-assets.json paths and the card URL against the served URL and calls loadModel', () => {
  const main = read(join(EXAMPLE, 'src', 'main.ts'));
  assert.ok(main.includes("const ortAssetsUrl = new URL('/ort-assets.json', location.href).href;"));
  assert.ok(main.includes('const abs = (p: string) => new URL(p, ortAssetsUrl).href;'));
  assert.ok(main.includes('loadModel(\n    { manifestUrl: cardUrl }'));
  assert.ok(existsSync(join(EXAMPLE, 'index.html')));
});

test('README orders build, npm install, the assets CLI, then npm run dev, and carries no ticket ID', () => {
  const readme = read(join(EXAMPLE, 'README.md'));
  const steps = ['npm run build', 'npm install --install-links', 'npx @komugi-ai/engine assets public --release-dir', 'npm run dev'];
  const at = steps.map((s) => readme.indexOf(s));
  for (const [i, pos] of at.entries()) assert.ok(pos >= 0, `missing ${steps[i]}`);
  for (let i = 1; i < at.length; i++) assert.ok((at[i] as number) > (at[i - 1] as number), `${steps[i]} out of order`);
  assert.match(readme, /`file:` dependency resolves through a symlink/);
  assert.match(readme, /realpath/);
  assert.match(readme, /--install-links/);
  assert.doesNotMatch(readme, /KOM-[0-9]/);
});
