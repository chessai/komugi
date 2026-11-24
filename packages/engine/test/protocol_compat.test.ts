// KOM-1198 unit 4 (docs/design/kom1198-move-io.md section 5/6/7, AC3): each
// golden request from `test/fixtures/kom1198_golden_replies.json` (recorded
// on the pre-change base commit by `test/support/record_kom1198_golden.ts`,
// unit 0) is re-run on the patched engine, `ms` zeroed and `descriptor` /
// `predictedReply.descriptor` stripped, and must deep-equal AND
// JSON.stringify-equal (key order) its golden reply. `fast-predicted-reply`
// is the one record that went through `handleFast`; every other record went
// through `handle()` (`record_kom1198_golden.ts`'s own `record`/`recordFast`
// split).

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';

import { WorkerEngine } from '../src/engine.ts';
import type { WorkerInitRequest, WorkerReply, WorkerRequest } from '../src/protocol.ts';

const FIXTURE_PATH = fileURLToPath(new URL('./fixtures/kom1198_golden_replies.json', import.meta.url));

interface GoldenRecord {
  name: string;
  request: WorkerRequest;
  reply: WorkerReply;
}

function loadGoldens(): GoldenRecord[] {
  return JSON.parse(readFileSync(FIXTURE_PATH, 'utf8')) as GoldenRecord[];
}

// Strips the two new descriptor fields (N9: predictedReply's own descriptor
// first, then the reply's own trailing descriptor) and zeroes `stats.ms`,
// preserving every other key's relative order for the JSON.stringify
// comparison.
function normalize(reply: WorkerReply): WorkerReply {
  if ('err' in reply) return reply;
  const { descriptor, predictedReply, stats, ...rest } = reply;
  void descriptor;
  const normalizedStats = { ...stats, ms: 0 };
  if (predictedReply === undefined) {
    return { ...rest, stats: normalizedStats };
  }
  const { descriptor: _predictedDescriptor, ...restPredicted } = predictedReply;
  return { ...rest, stats: normalizedStats, predictedReply: restPredicted };
}

async function replay(engine: WorkerEngine, record: GoldenRecord): Promise<WorkerReply> {
  if (record.name === 'fast-predicted-reply') {
    engine.handlePonderSettings({ id: 'golden', type: 'ponderSettings', enabled: false });
    return engine.handleFast(record.request, engine.defaultBatchEvaluator());
  }
  return engine.handle(record.request);
}

test('protocol_compat: every kom1198 golden reply is byte-identical once descriptor/predictedReply.descriptor are stripped', async () => {
  const goldens = loadGoldens();
  assert.ok(goldens.length > 0, 'empty golden fixture');
  for (const golden of goldens) {
    const engine = new WorkerEngine();
    const reply = await replay(engine, golden);
    const got = normalize(reply);
    const want = normalize(golden.reply);
    assert.deepEqual(got, want, golden.name);
    assert.equal(JSON.stringify(got), JSON.stringify(want), golden.name);
  }
});

// KOM-1226 AC3: a new optional init field must not force every existing
// caller to set it -- an init message built exactly as every caller built
// it before this field existed still satisfies WorkerInitRequest.
test('protocol_compat: WorkerInitRequest without ortLogSeverityLevel stays valid, and the field is typed as an optional number', () => {
  const withoutLevel: WorkerInitRequest = { id: 'i', type: 'init', modelUrl: 'model.onnx', modelSha256: 'sha' };
  assert.equal(withoutLevel.ortLogSeverityLevel, undefined);
  const withLevel: WorkerInitRequest = { ...withoutLevel, ortLogSeverityLevel: 2 };
  assert.equal(withLevel.ortLogSeverityLevel, 2);
});
