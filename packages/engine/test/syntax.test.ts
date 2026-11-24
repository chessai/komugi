// U1: enforce the section-1 syntax ban across web/engine/src (Node type stripping
// cannot generate code), then load every module to prove it strips and links.

import assert from 'node:assert/strict';
import { readdirSync, readFileSync } from 'node:fs';
import { dirname, join, relative, resolve } from 'node:path';
import { test } from 'node:test';
import { fileURLToPath, pathToFileURL } from 'node:url';

const SRC = resolve(dirname(fileURLToPath(import.meta.url)), '..', 'src');

function walk(dir: string): string[] {
  const out: string[] = [];
  for (const ent of readdirSync(dir, { withFileTypes: true })) {
    const p = join(dir, ent.name);
    if (ent.isDirectory()) out.push(...walk(p));
    else out.push(p);
  }
  return out.sort();
}

// Blank out comments and string/template contents, keeping offsets and newlines,
// so the patterns only see code. Import specifiers are read from the raw text.
function codeOnly(src: string): string {
  let out = '';
  let i = 0;
  while (i < src.length) {
    const c = src[i];
    const n = src[i + 1];
    if (c === '/' && n === '/') {
      while (i < src.length && src[i] !== '\n') { out += ' '; i++; }
    } else if (c === '/' && n === '*') {
      while (i < src.length && !(src[i] === '*' && src[i + 1] === '/')) { out += src[i] === '\n' ? '\n' : ' '; i++; }
      out += '  ';
      i += 2;
    } else if (c === '"' || c === "'" || c === '`') {
      out += c;
      i++;
      while (i < src.length && src[i] !== c) {
        if (src[i] === '\\') { out += ' '; i++; }
        out += src[i] === '\n' ? '\n' : ' ';
        i++;
      }
      out += c;
      i++;
    } else {
      out += c;
      i++;
    }
  }
  return out;
}

const BANNED: [string, RegExp][] = [
  ['enum', /\benum\s+[A-Za-z_$]/],
  ['namespace/module block', /(^|[^.\w$])(?<!declare\s+)(namespace|module)\s+[A-Za-z_$][\w$.]*\s*\{/m],
  ['import = require', /\bimport\s+[\w$]+\s*=\s*require\s*\(/],
  ['export =', /\bexport\s*=(?!=)/],
  ['decorator', /(^|[\s(;{])@[A-Za-z_$]/m],
];
const PARAM_PROPERTY = /\b(public|private|protected|readonly|override)\s+[\w$]/;

export function violations(src: string): string[] {
  const code = codeOnly(src);
  const found: string[] = [];
  for (const [name, re] of BANNED) if (re.test(code)) found.push(name);
  for (const m of code.matchAll(/\bconstructor\s*\(([^)]*)\)/g)) {
    if (PARAM_PROPERTY.test(m[1])) found.push('constructor parameter property');
  }
  return found;
}

type Specifier = { names: string[]; spec: string; typeOnly: boolean };

function imports(src: string): Specifier[] {
  const out: Specifier[] = [];
  const re = /\b(import|export)\s+(type\s+)?(?:([\w$]+\s*,?\s*)?\{([^}]*)\}|\*\s+as\s+[\w$]+|[\w$]+|\*)?\s*(?:from\s*)?['"]([^'"]+)['"]/g;
  for (const m of src.matchAll(re)) {
    const names = (m[4] ?? '').split(',').map((s) => s.trim()).filter(Boolean)
      .filter((s) => !s.startsWith('type ')).map((s) => s.split(/\s+as\s+/)[0]);
    out.push({ names, spec: m[5], typeOnly: Boolean(m[2]) });
  }
  for (const m of src.matchAll(/\bimport\s*\(\s*['"]([^'"]+)['"]\s*\)/g)) {
    out.push({ names: [], spec: m[1], typeOnly: false });
  }
  return out;
}

function valueExports(src: string): Set<string> {
  const code = codeOnly(src);
  const out = new Set<string>();
  for (const m of code.matchAll(/\bexport\s+(?:async\s+)?(?:const|let|var|function\*?|class|abstract\s+class)\s+([\w$]+)/g)) out.add(m[1]);
  for (const m of code.matchAll(/\bexport\s*\{([^}]*)\}/g)) {
    for (const part of m[1].split(',')) {
      const s = part.trim();
      if (s && !s.startsWith('type ')) out.add(s.split(/\s+as\s+/).pop() as string);
    }
  }
  return out;
}

export function importProblems(file: string, src: string, read: (p: string) => string): string[] {
  const found: string[] = [];
  for (const imp of imports(src)) {
    if (!imp.spec.startsWith('.')) continue;
    if (!imp.spec.endsWith('.ts')) {
      found.push(`relative import without .ts: ${imp.spec}`);
      continue;
    }
    if (imp.typeOnly) continue;
    const exported = valueExports(read(resolve(dirname(file), imp.spec)));
    for (const name of imp.names) {
      if (!exported.has(name)) found.push(`'${name}' from ${imp.spec} is not a value export; use import type`);
    }
  }
  return found;
}

test('the detectors catch every banned form (no vacuous pass)', () => {
  const bad: [string, string][] = [
    ['enum Color { W, B }', 'enum'],
    ['export const enum Mode { A }', 'enum'],
    ['namespace Foo { export const x = 1; }', 'namespace/module block'],
    ['module Bar { const y = 2 }', 'namespace/module block'],
    ["import fs = require('fs');", 'import = require'],
    ['export = Foo;', 'export ='],
    ['@sealed\nclass A {}', 'decorator'],
    ['class A { constructor(private x: number) {} }', 'constructor parameter property'],
    ['class A { constructor(public readonly x: number) {} }', 'constructor parameter property'],
  ];
  for (const [snippet, kind] of bad) assert.ok(violations(snippet).includes(kind), snippet);
  const fine = [
    "const enumValue = 1; // enum Foo {}\nconst s = 'namespace X {';",
    'declare module Ext { }',
    'class A { constructor(x: number) { this.x = x; } x: number; }',
    'const M = { a: 1 } as const; const e = a === b;',
  ];
  for (const snippet of fine) assert.deepEqual(violations(snippet), [], snippet);
  const read = (p: string) => (p.endsWith('/t.ts') ? 'export type T = number;\nexport interface I {}\nexport const v = 1;\nexport function g() {}' : '');
  assert.deepEqual(importProblems('/x/a.ts', "import { v, g } from './t.ts';\nimport type { T } from './t.ts';\nimport { type I, v as w } from './t.ts';", read), []);
  assert.equal(importProblems('/x/a.ts', "import { T } from './t.ts';", read).length, 1);
  assert.equal(importProblems('/x/a.ts', "import { v } from './t';", read).length, 1);
});

test('web/engine/src uses no syntax that type stripping rejects', () => {
  const files = walk(SRC);
  const problems: string[] = [];
  for (const f of files) {
    const rel = relative(SRC, f);
    if (f.endsWith('.tsx')) problems.push(`${rel}: JSX/.tsx is banned`);
    if (!f.endsWith('.ts')) continue;
    const src = readFileSync(f, 'utf8');
    for (const v of violations(src)) problems.push(`${rel}: ${v}`);
    for (const v of importProblems(f, src, (p) => readFileSync(p, 'utf8'))) problems.push(`${rel}: ${v}`);
  }
  console.log(`syntax: files=${files.length} problems=${problems.length}${problems.length ? ' first=' + problems[0] : ''}`);
  assert.ok(files.some((f) => f.endsWith('fen.ts')));
  assert.deepEqual(problems, []);
});

test('every web/engine/src module loads under type stripping', async () => {
  for (const f of walk(SRC).filter((p) => p.endsWith('.ts'))) {
    await import(pathToFileURL(f).href);
  }
});
