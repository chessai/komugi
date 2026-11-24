// KOM-1198 unit 0 (design doc docs/design/kom1198-move-io.md section 7):
// records base-commit WorkerEngine replies so later units' AC3
// (test/protocol_compat.test.ts) can diff their patched replies against
// byte-identical pre-change behavior. Runs on the pre-change base commit,
// before any engine.ts/protocol.ts change.
//
// Determinism: a fresh WorkerEngine per request (the deterministic fake-leaf
// evaluator worker.test.ts relies on, engine.ts's own default), stats.ms
// zeroed (worker.test.ts:157's own convention), and no wall-clock or random
// value left anywhere else in a WorkerStats reply (no `timing`/`maxMs` on
// any recorded request, so `phases`/`deadlineHit` never carry wall-clock
// noise). Object literals below are always built in the same key order, so
// JSON.stringify's output is stable across runs.

import { writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { ADVANCED_POSITION } from '../../src/constants.ts';
import { WorkerEngine } from '../../src/engine.ts';
import { boardToFen } from '../../src/fen.ts';
import type { WorkerReply, WorkerRequest } from '../../src/protocol.ts';
import { gameFromHistory } from '../../src/root.ts';
import { sanRecords } from '../../src/san.ts';

const DEFAULT_OUT = resolve(dirname(fileURLToPath(import.meta.url)), '..', 'fixtures', 'kom1198_golden_replies.json');

interface GoldenRecord {
  name: string;
  request: WorkerRequest;
  reply: WorkerReply;
}

// observation_modes.test.ts:278's own walk: at each ply, take the first
// legal SAN move enumerated at the position reached so far.
function walk(n: number): string[] {
  const history: string[] = [];
  for (let i = 0; i < n; i++) history.push(sanRecords(gameFromHistory(history).board)[0].san);
  return history;
}

function zeroMs(reply: WorkerReply): WorkerReply {
  if ('err' in reply) return reply;
  return { ...reply, stats: { ...reply.stats, ms: 0 } };
}

function record(name: string, request: WorkerRequest): GoldenRecord {
  const engine = new WorkerEngine();
  return { name, request, reply: zeroMs(engine.handle(request)) };
}

async function recordFast(name: string, request: WorkerRequest): Promise<GoldenRecord> {
  const engine = new WorkerEngine();
  // No pondering left running past this request: unit 0 wants exactly this
  // reply, not a speculative search this script never awaits.
  engine.handlePonderSettings({ id: 'golden', type: 'ponderSettings', enabled: false });
  const reply = await engine.handleFast(request, engine.defaultBatchEvaluator());
  return { name, request, reply: zeroMs(reply) };
}

async function buildRecords(): Promise<GoldenRecord[]> {
  const records: GoldenRecord[] = [];

  records.push(record('advanced-start', { id: 'advanced-start', fen: ADVANCED_POSITION, sims: 16 }));

  for (const plies of [10, 40, 70]) {
    const history = walk(plies);
    const fen = boardToFen(gameFromHistory(history).board);
    records.push(
      record(`walk-${plies}`, { id: `walk-${plies}`, fen, history, sims: 16 }),
    );
  }

  records.push(
    record('rules-ligungi', { id: 'rules-ligungi', fen: ADVANCED_POSITION, sims: 16, rules: 'ligungi' }),
  );

  const fastHistory = walk(2);
  const fastFen = boardToFen(gameFromHistory(fastHistory).board);
  records.push(
    await recordFast('fast-predicted-reply', {
      id: 'fast-predicted-reply',
      fen: fastFen,
      history: fastHistory,
      sims: 16,
      batch: 4,
    }),
  );

  return records;
}

function parseOut(argv: string[]): string {
  const i = argv.indexOf('--out');
  if (i === -1) return DEFAULT_OUT;
  const value = argv[i + 1];
  if (value === undefined) throw new Error('--out requires a path argument');
  return resolve(value);
}

async function main(): Promise<void> {
  const out = parseOut(process.argv.slice(2));
  const records = await buildRecords();
  writeFileSync(out, JSON.stringify(records, null, 2) + '\n');
}

await main();
