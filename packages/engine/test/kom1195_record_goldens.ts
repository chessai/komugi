// KOM-1195 unit U0 (docs/design/kom1195-analysis-api.md sections 8/9, AC1):
// records base-commit `handleFast` replies so every later KOM-1195 `src/`
// unit's `test/kom1195_compat.test.ts` can check that a request carrying no
// new field still gets a byte-identical reply. Runs on the pre-change base
// (KOM-1203 unit 6's merge), before any `src/` change:
//
//   node test/kom1195_record_goldens.ts [--out <path>]
//
// Records, all through `handleFast` on a fresh engine per request:
// - every committed non-terminal parity position (fixture.ts's
//   `loadFixture`, the committed sample; no KOM_* variable) at `sims: 96`
//   x batch {1, 8} x {no `maxMs`, `maxMs: 1e12`} (re-review N6: the chunked
//   and continuous paths only differ at sims this high);
// - one `level` request and one `history` request per position class
//   (the parity record's `phase`);
// - every `test/fixtures/kom1198_golden_replies.json` request, verbatim.
//
// Determinism: each engine is built with a constant clock (`clock: () => 0`)
// so neither a `maxMs: 1e12` nor a level's own `capMs` deadline can ever
// fire, ponder is disabled before the request (no speculative search left
// running), and `stats.ms` is zeroed. `descriptor` is kept (unlike
// protocol_compat.test.ts's `normalize()`). The output is gzipped with a
// zero header mtime, so re-recording on the same base is byte-identical.

import { readFileSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { gzipSync } from 'node:zlib';

import type { ProbeSession, SessionFactory } from '../src/backend.ts';
import type { MlhBatchEvaluator } from '../src/batch_evaluator.ts';
import { ACTION_SIZE } from '../src/constants.ts';
import { WorkerEngine } from '../src/engine.ts';
import type { WorkerInitEnv } from '../src/engine.ts';
import { makeFakeMlhEvaluator } from '../src/fake_evaluator.ts';
import type { LevelName } from '../src/levels.ts';
import type { WorkerLevelRequest, WorkerReply, WorkerRequest } from '../src/protocol.ts';
import { loadFixture } from './fixture.ts';

const HERE = dirname(fileURLToPath(import.meta.url));
export const GOLDENS_PATH = resolve(HERE, 'fixtures', 'kom1195_compat_goldens.json.gz');
const KOM1198_PATH = resolve(HERE, 'fixtures', 'kom1198_golden_replies.json');

export const GOLDEN_SIMS = 96;
export const GOLDEN_BATCHES = [1, 8] as const;
export const NEVER_MS = 1e12;
const GOLDEN_LEVEL: LevelName = 'casual';

// 'plain': `new WorkerEngine`, no init, `defaultBatchEvaluator()`.
// 'level-wasm-mt': init()ed onto the fake wasm multi-thread backend (the
// one level case whose resolved batch is > 1, so the chunked loop runs),
// then `currentBatchEvaluator`.
export type GoldenEngine = 'plain' | 'level-wasm-mt';
export type GoldenKind = 'parity' | 'level' | 'history' | 'kom1198';

export interface GoldenRecord {
  name: string;
  kind: GoldenKind;
  position: string | null; // parity record id, null for kom1198 requests
  engine: GoldenEngine;
  request: WorkerRequest | WorkerLevelRequest;
  reply: WorkerReply;
}

export function zeroMs(reply: WorkerReply): WorkerReply {
  if ('err' in reply) return reply;
  return { ...reply, stats: { ...reply.stats, ms: 0 } };
}

function fakeBatchEvaluator(): MlhBatchEvaluator {
  const leaf = makeFakeMlhEvaluator(ACTION_SIZE);
  return (states) => states.map((state) => leaf(state));
}

// levels_worker.test.ts's own wasm-mt `initFor` case (fake probe sessions
// on a fake probe clock, isolated, 8 cores).
async function initWasmMt(engine: WorkerEngine): Promise<void> {
  let clock = 0;
  const session: ProbeSession = {
    async warmup(): Promise<void> {},
    async run(batch: number): Promise<void> {
      clock += 14 * batch;
    },
  };
  const factory: SessionFactory = {
    setNumThreads(): void {},
    async createWebgpu(): Promise<ProbeSession> {
      return session;
    },
    async createWasm(): Promise<ProbeSession> {
      return session;
    },
  };
  const createBatchBackend: WorkerInitEnv['createBatchBackend'] = async () => fakeBatchEvaluator();
  const reply = await engine.init(
    { id: 'kom1195-init', modelUrl: 'model.onnx', modelSha256: 'sha-kom1195-wasm-mt' },
    { sessionFactory: factory, isolated: true, hardwareConcurrency: 8, now: () => clock, createBatchBackend },
  );
  if ('err' in reply) throw new Error(`kom1195 golden init failed: ${reply.err}`);
}

// Runs one golden request exactly as it was recorded: fresh engine, ponder
// off, `ms` zeroed. Shared by the recorder and the compat test.
export async function runGolden(
  engineKind: GoldenEngine,
  request: WorkerRequest | WorkerLevelRequest,
): Promise<WorkerReply> {
  const engine = new WorkerEngine({ clock: () => 0 });
  engine.handlePonderSettings({ id: 'kom1195-golden', type: 'ponderSettings', enabled: false });
  let evaluator: MlhBatchEvaluator;
  if (engineKind === 'plain') {
    evaluator = engine.defaultBatchEvaluator();
  } else {
    await initWasmMt(engine);
    const current = engine.currentBatchEvaluator;
    if (current === null || current === undefined) throw new Error('kom1195 golden: no batch evaluator after init');
    evaluator = current;
  }
  return zeroMs(await engine.handleFast(request, evaluator));
}

export function parityGoldenName(id: string, batch: number, withMaxMs: boolean): string {
  return `parity-${id}-b${batch}-${withMaxMs ? 'maxms' : 'nomaxms'}`;
}

async function buildRecords(): Promise<GoldenRecord[]> {
  const records: GoldenRecord[] = [];
  const add = async (
    name: string,
    kind: GoldenKind,
    position: string | null,
    engine: GoldenEngine,
    request: WorkerRequest | WorkerLevelRequest,
  ): Promise<void> => {
    records.push({ name, kind, position, engine, request, reply: await runGolden(engine, request) });
  };

  const positions = loadFixture().filter((rec) => !rec.terminal);
  for (const pos of positions) {
    for (const batch of GOLDEN_BATCHES) {
      for (const withMaxMs of [false, true]) {
        const name = parityGoldenName(pos.id, batch, withMaxMs);
        const request: WorkerRequest = withMaxMs
          ? { id: name, fen: pos.fen, sims: GOLDEN_SIMS, batch, maxMs: NEVER_MS }
          : { id: name, fen: pos.fen, sims: GOLDEN_SIMS, batch };
        await add(name, 'parity', pos.id, 'plain', request);
      }
    }
  }

  const classes = [...new Set(positions.map((pos) => pos.phase))];
  for (const phase of classes) {
    const inClass = positions.filter((pos) => pos.phase === phase);
    const first = inClass[0];
    if (first === undefined) continue;
    const levelName = `level-${phase}-${first.id}`;
    await add(levelName, 'level', first.id, 'level-wasm-mt', { id: levelName, fen: first.fen, level: GOLDEN_LEVEL });
    const withHistory = inClass.find((pos) => pos.history.length > 0) ?? first;
    const historyName = `history-${phase}-${withHistory.id}`;
    await add(historyName, 'history', withHistory.id, 'plain', {
      id: historyName,
      fen: withHistory.fen,
      history: withHistory.history,
      sims: GOLDEN_SIMS,
      batch: 8,
    });
  }

  const kom1198 = JSON.parse(readFileSync(KOM1198_PATH, 'utf8')) as { name: string; request: WorkerRequest }[];
  for (const golden of kom1198) {
    await add(`kom1198-${golden.name}`, 'kom1198', null, 'plain', golden.request);
  }
  return records;
}

export function encodeGoldens(records: GoldenRecord[]): Buffer {
  // zlib's gzip header carries mtime 0 and no file name.
  return gzipSync(Buffer.from(JSON.stringify(records, null, 1) + '\n', 'utf8'), { level: 9 });
}

function parseOut(argv: string[]): string {
  const i = argv.indexOf('--out');
  if (i === -1) return GOLDENS_PATH;
  const value = argv[i + 1];
  if (value === undefined) throw new Error('--out requires a path argument');
  return resolve(value);
}

async function main(): Promise<void> {
  const out = parseOut(process.argv.slice(2));
  const records = await buildRecords();
  writeFileSync(out, encodeGoldens(records));
  const errs = records.filter((rec) => 'err' in rec.reply).map((rec) => rec.name);
  console.log(`kom1195 goldens: ${records.length} records -> ${out}; err replies: ${errs.length ? errs.join(',') : 'none'}`);
}

if (process.argv[1] !== undefined && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  await main();
}
