#!/usr/bin/env node
// packages/engine/tools/strip.mjs -- browser delivery without a bundler (design
// doc section 1 "Browser delivery without a bundler", section 6 U8).
//
// Uses Node's built-in `module.stripTypeScriptTypes` (type stripping, no
// code generation, no `tsc`) to write a plain-JS copy of every packages/engine/src
// module under `--out` (default `packages/engine/dist`), rewriting each relative
// `./x.ts` / `../x.ts` import specifier to `./x.js` / `../x.js` so the
// written tree is a set of ES modules loadable by a Web Worker with
// `type: 'module'`, with no `.ts` specifier anywhere in the output.
// `dist/` is generated and never committed (design section 1).
//
// No npm, no network, no package.json: only Node's own module.

import { stripTypeScriptTypes } from 'node:module';
import { mkdirSync, readFileSync, readdirSync, statSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import process from 'node:process';

const USAGE = 'usage: strip.mjs [--src packages/engine/src] [--out packages/engine/dist]';

function parseArgs(argv) {
  const args = { src: null, out: null };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    const next = () => {
      i += 1;
      if (i >= argv.length) throw new Error(`strip.mjs: ${arg} needs a value`);
      return argv[i];
    };
    if (arg === '--src') args.src = next();
    else if (arg === '--out') args.out = next();
    else throw new Error(`strip.mjs: unknown argument ${arg}\n${USAGE}`);
  }
  const here = path.dirname(new URL(import.meta.url).pathname);
  args.src ??= path.resolve(here, '..', 'src');
  args.out ??= path.resolve(here, '..', 'dist');
  return args;
}

function walkTsFiles(dir) {
  const out = [];
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) out.push(...walkTsFiles(full));
    else if (entry.name.endsWith('.ts')) out.push(full);
  }
  return out.sort();
}

// Rewrites every relative `'./x.ts'` / `"../x.ts"` import/export specifier
// (static or dynamic `import(...)`) to its `.js` sibling. Only touches
// specifiers that are already relative (start with `.`) and end in `.ts` --
// a bare package specifier (none exist in packages/engine/src today, design
// section 1) would be left untouched.
function rewriteSpecifiers(code) {
  return code.replace(
    /((?:\bimport\b[^'"()]*?|\bexport\b[^'"()]*?)from\s*|\bimport\s*\()\s*(['"])(\.[^'"]+?)\.ts\2/g,
    (_match, prefix, quote, specifier) => `${prefix}${quote}${specifier}.js${quote}`,
  );
}

function main(argv) {
  const args = parseArgs(argv);
  mkdirSync(args.out, { recursive: true });

  const files = walkTsFiles(args.src);
  let written = 0;
  for (const file of files) {
    const rel = path.relative(args.src, file);
    const outRel = rel.replace(/\.ts$/, '.js');
    const outPath = path.join(args.out, outRel);
    mkdirSync(path.dirname(outPath), { recursive: true });

    const source = readFileSync(file, 'utf8');
    const stripped = stripTypeScriptTypes(source, { mode: 'strip' });
    const rewritten = rewriteSpecifiers(stripped);

    if (/(^|[^.\w])\.{1,2}\/[^'"]*\.ts['"]/.test(rewritten)) {
      throw new Error(`strip.mjs: ${outRel} still has a .ts import specifier after rewriting`);
    }

    writeFileSync(outPath, rewritten);
    written += 1;
  }

  const sizeOf = (p) => statSync(p).size;
  console.log(
    JSON.stringify({ src: args.src, out: args.out, files: written, bytes: files.reduce((a, f) => a + sizeOf(f), 0) }),
  );
  return 0;
}

process.exitCode = main(process.argv.slice(2));
