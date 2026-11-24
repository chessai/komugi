// Reusable static import-graph walker, shared by AC6 (test/import_graph.ts
// walking index.ts/worker.ts, unit 2) and AC7 (this unit, walking rules.ts).
// Text-based, not a real parser: the repo's own syntax rules (every relative
// import/export names its own `.ts` extension, no bundler-only shorthand)
// make a regex walk sufficient without pulling in a TS parser dependency.

import { readFileSync, existsSync } from 'node:fs';
import { builtinModules } from 'node:module';
import { dirname, join, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

// This module lives at packages/engine/test/support/, so four directories up
// from its own file is the repo root -- usable even when there is no `.git`
// above the tree (e.g. a source download of the repository).
const REPO_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..', '..', '..');

export type ImportGraphResult = {
  modules: string[];
  nodeBuiltins: string[];
  nonLiteralImports: string[];
  packageSpecifiers: string[];
  absoluteSpecifiers: string[];
};

const BUILTIN_NAMES = new Set(builtinModules);

function isRelativeSpecifier(spec: string): boolean {
  return spec.startsWith('./') || spec.startsWith('../');
}

function isNodeBuiltin(spec: string): boolean {
  if (spec.startsWith('node:')) return true;
  const bare = spec.split('/')[0] ?? spec;
  return BUILTIN_NAMES.has(bare);
}

// Absolute specifiers (design doc section 7.1 AC6): a leading '/', a Windows
// drive letter ('C:\' or 'C:/'), a 'file:' specifier, or any other URL
// scheme ('https://...', 'data:...').
function isAbsoluteSpecifier(spec: string): boolean {
  if (spec.startsWith('/')) return true;
  if (/^[a-zA-Z]:[\\/]/.test(spec)) return true;
  if (/^[a-zA-Z][a-zA-Z0-9+.-]*:/.test(spec)) return true;
  return false;
}

// Strips `//` line comments and `/* */` block comments before specifier
// collection, skipping over string and template literals so a comment
// marker inside a literal is left alone (design doc section 7.1 AC6, fix
// for src/worker_host.ts's `// each import() is lazy` being misread as a
// non-literal dynamic import).
function stripComments(text: string): string {
  let out = '';
  let i = 0;
  const n = text.length;
  while (i < n) {
    const ch = text[i] as string;
    const next = text[i + 1];
    if (ch === '/' && next === '/') {
      const end = text.indexOf('\n', i);
      if (end === -1) break;
      i = end;
      continue;
    }
    if (ch === '/' && next === '*') {
      const end = text.indexOf('*/', i + 2);
      if (end === -1) {
        i = n;
        break;
      }
      i = end + 2;
      continue;
    }
    if (ch === "'" || ch === '"' || ch === '`') {
      const quote = ch;
      out += ch;
      i += 1;
      while (i < n) {
        const c = text[i] as string;
        out += c;
        if (c === '\\') {
          i += 1;
          if (i < n) out += text[i];
          i += 1;
          continue;
        }
        if (c === quote) {
          i += 1;
          break;
        }
        i += 1;
      }
      continue;
    }
    out += ch;
    i += 1;
  }
  return out;
}

function findRepoRoot(startDir: string): string {
  // An entry inside this repo's own tree always roots at REPO_ROOT, with or
  // without a `.git` above it. An entry outside the tree (e.g. a fixture
  // planted under os.tmpdir()) keeps the old walk-up-for-.git behaviour, so
  // its module paths stay relative to wherever it actually lives.
  const rel = relative(REPO_ROOT, startDir);
  if (rel === '' || !rel.startsWith('..')) return REPO_ROOT;

  let dir = startDir;
  for (;;) {
    if (existsSync(join(dir, '.git'))) return dir;
    const parent = dirname(dir);
    if (parent === dir) return startDir;
    dir = parent;
  }
}

function toRepoRelative(repoRoot: string, absPath: string): string {
  return relative(repoRoot, absPath).split('\\').join('/');
}

function collectFromClauseSpecifiers(text: string): string[] {
  const specifiers: string[] = [];
  const fromClauseRe = /\b(?:import|export)\b[^'"(;]*?\bfrom\s*(['"])((?:(?!\1).)*)\1/g;
  let m: RegExpExecArray | null;
  while ((m = fromClauseRe.exec(text)) !== null) {
    specifiers.push(m[2] as string);
  }
  return specifiers;
}

function collectSideEffectSpecifiers(text: string): string[] {
  const specifiers: string[] = [];
  const sideEffectRe = /\bimport\s*(['"])((?:(?!\1).)*)\1/g;
  let m: RegExpExecArray | null;
  while ((m = sideEffectRe.exec(text)) !== null) {
    specifiers.push(m[2] as string);
  }
  return specifiers;
}

function collectDynamicImports(text: string): { literal: string[]; nonLiteral: string[] } {
  const literal: string[] = [];
  const nonLiteral: string[] = [];
  const callRe = /\bimport\s*\(/g;
  let m: RegExpExecArray | null;
  while ((m = callRe.exec(text)) !== null) {
    const afterIdx = m.index + m[0].length;
    const rest = text.slice(afterIdx);
    const trimmed = rest.replace(/^\s+/, '');
    if (trimmed.startsWith("'") || trimmed.startsWith('"')) {
      const quote = trimmed[0] as string;
      const end = trimmed.indexOf(quote, 1);
      if (end !== -1) {
        literal.push(trimmed.slice(1, end));
        continue;
      }
    }
    const snippetStart = Math.max(0, m.index - 20);
    const snippet = text.slice(snippetStart, m.index + 60).replace(/\s+/g, ' ').trim();
    nonLiteral.push(snippet);
  }
  return { literal, nonLiteral };
}

export function walkImports(entry: string): ImportGraphResult {
  const entryAbs = resolve(entry);
  const repoRoot = findRepoRoot(dirname(entryAbs));

  const visited = new Set<string>([entryAbs]);
  const queue: string[] = [entryAbs];
  const nodeBuiltins = new Set<string>();
  const nonLiteralImports: string[] = [];
  const packageSpecifiers = new Set<string>();
  const absoluteSpecifiers = new Set<string>();

  while (queue.length > 0) {
    const current = queue.shift() as string;
    const rawText = readFileSync(current, 'utf8');
    const text = stripComments(rawText);
    const dir = dirname(current);

    const specifiers = [
      ...collectFromClauseSpecifiers(text),
      ...collectSideEffectSpecifiers(text),
    ];
    const dynamic = collectDynamicImports(text);
    specifiers.push(...dynamic.literal);
    for (const snippet of dynamic.nonLiteral) {
      nonLiteralImports.push(`${toRepoRelative(repoRoot, current)}: ${snippet}`);
    }

    for (const spec of specifiers) {
      if (isNodeBuiltin(spec)) {
        nodeBuiltins.add(spec);
      } else if (isAbsoluteSpecifier(spec)) {
        absoluteSpecifiers.add(spec);
      } else if (isRelativeSpecifier(spec)) {
        const resolved = resolve(dir, spec);
        if (!visited.has(resolved)) {
          visited.add(resolved);
          queue.push(resolved);
        }
      } else {
        packageSpecifiers.add(spec);
      }
    }
  }

  const modules = [...visited].map((abs) => toRepoRelative(repoRoot, abs)).sort();
  return {
    modules,
    nodeBuiltins: [...nodeBuiltins].sort(),
    nonLiteralImports,
    packageSpecifiers: [...packageSpecifiers].sort(),
    absoluteSpecifiers: [...absoluteSpecifiers].sort(),
  };
}

// AC6 (design doc section 7.1): reads each reached module's RAW text --
// comments included, unlike the specifier walk above -- and reports
// `<path>: <pattern>` for every disallowed pattern found. `modules` is the
// repo-relative list `walkImports` already produced (or an absolute-path
// list, resolved via `repoRoot`).
const TEXT_SCAN_PATTERNS: ReadonlyArray<{ label: string; re: RegExp }> = [
  { label: '/home/', re: /\/home\// },
  { label: 'process.env', re: /process\.env/ },
  { label: 'ORTWEB_DIR', re: /ORTWEB_DIR/ },
  { label: '.process via globalThis cast', re: /globalThis[^;\n]*\)\s*\.process\b/ },
];

export function scanText(modules: string[], repoRoot?: string): string[] {
  const findings: string[] = [];
  for (const mod of modules) {
    const abs = mod.startsWith('/') ? mod : resolve(repoRoot ?? '.', mod);
    const text = readFileSync(abs, 'utf8');
    for (const { label, re } of TEXT_SCAN_PATTERNS) {
      if (re.test(text)) {
        findings.push(`${mod}: ${label}`);
      }
    }
  }
  return findings;
}
