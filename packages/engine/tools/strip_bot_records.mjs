#!/usr/bin/env node
// web/engine/tools/strip_bot_records.mjs -- KOM-1191: drops every ligungi
// oracle fixture record whose `source` is a bot-match game (and the
// after-claim positions derived from them), recomputing `counts`/`modes`
// and re-serializing with recursively sorted keys. The algorithm is FIXED
// (KOM-1191 spec): restructured into stripBotRecords() + a CLI guard only,
// the output bytes must not change.

import { readFileSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

function sortKeys(v) {
  return Array.isArray(v)
    ? v.map(sortKeys)
    : v && typeof v === 'object'
      ? Object.fromEntries(Object.keys(v).sort().map((k) => [k, sortKeys(v[k])]))
      : v;
}

function strip(doc) {
  const records = doc.records.filter((r) => !r.source.startsWith('match-'));
  const counts = Object.fromEntries(doc.classes.map((c) => [c, 0]));
  for (const r of records) for (const c of r.classes.length ? r.classes : ['none']) counts[c] = (counts[c] ?? 0) + 1;
  const modes = {};
  for (const r of records) modes[r.mode] = (modes[r.mode] ?? 0) + 1;
  const text = `${JSON.stringify(sortKeys({ ...doc, counts, modes, records }))}\n`;
  return { records, counts, modes, text };
}

export function stripBotRecords(text) {
  return strip(JSON.parse(text)).text;
}

function main(argv) {
  const [inp, out] = argv;
  const doc = JSON.parse(readFileSync(inp, 'utf8'));
  const { records, counts, modes, text } = strip(doc);
  writeFileSync(out, text);
  console.log(
    JSON.stringify({
      before: doc.records.length,
      after: records.length,
      counts,
      modes,
      bytes: text.length,
      sha256: createHash('sha256').update(text).digest('hex'),
    }),
  );
}

if (fileURLToPath(import.meta.url) === path.resolve(process.argv[1])) {
  main(process.argv.slice(2));
}
