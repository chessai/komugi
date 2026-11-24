// KOM-1198 unit 6 (docs/design/kom1198-move-io.md section 6/7, AC2): for
// each golden history (test/fixtures/kom1198_golden_replies.json, unit 0)
// whose every SAN resolves unambiguously, derive a parallel descriptor
// history as `describeMove(board, resolveSan(board, san))` per ply (section
// 3's SAN-is-not-injective note means a ply with more than one legal move
// sharing a SAN can't be vouched for, so it excludes that golden history
// from this check rather than guessing). Running the all-SAN and
// all-descriptor requests on fresh WorkerEngines must give byte-identical
// replies (`ms` zeroed, `descriptor` included), and replaying both
// histories ply-by-ply must agree on `repetitionCount()`, `turns` and
// `winState()` throughout (the loop pattern of root_start.test.ts:122).

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';

import { ADVANCED_POSITION } from '../src/constants.ts';
import { WorkerEngine } from '../src/engine.ts';
import { boardFromFen } from '../src/fen.ts';
import { Game } from '../src/game.ts';
import { describeMove, resolveMoveDescriptor, type MoveDescriptor } from '../src/move_io.ts';
import type { WorkerReply, WorkerRequest } from '../src/protocol.ts';
import { RepetitionRule } from '../src/repetition.ts';
import { sanRecords } from '../src/san.ts';

const FIXTURE_PATH = fileURLToPath(new URL('./fixtures/kom1198_golden_replies.json', import.meta.url));

interface GoldenRecord {
  name: string;
  request: WorkerRequest;
  reply: WorkerReply;
}

function loadGoldens(): GoldenRecord[] {
  return JSON.parse(readFileSync(FIXTURE_PATH, 'utf8')) as GoldenRecord[];
}

// Replays `sanHistory` from ADVANCED_POSITION (unit 0's own replay base, N3)
// under PLACEMENT, deriving one descriptor per ply and comparing a
// lockstep descriptor-driven Game against the SAN-driven one at every ply.
// Returns null the moment a ply's SAN matches more than one legal move:
// that golden history can't be vouched for as "SAN resolved to the move
// actually played" and is excluded from the AC2 check below.
function deriveDescriptors(name: string, sanHistory: readonly string[]): MoveDescriptor[] | null {
  const sanGame = new Game(boardFromFen(ADVANCED_POSITION), RepetitionRule.PLACEMENT);
  const descGame = new Game(boardFromFen(ADVANCED_POSITION), RepetitionRule.PLACEMENT);
  const descriptors: MoveDescriptor[] = [];
  for (let ply = 0; ply < sanHistory.length; ply++) {
    const san = sanHistory[ply]!;
    const normalized = san.replace(/[=#]+$/, '');
    const matches = sanRecords(sanGame.board).filter((r) => r.san === normalized || r.san === san);
    if (matches.length !== 1) return null;
    const action = matches[0]!.id;
    const descriptor = describeMove(sanGame.board, action);

    sanGame.play(action);
    descGame.play(resolveMoveDescriptor(descGame.board, descriptor));
    assert.equal(descGame.repetitionCount(), sanGame.repetitionCount(), `${name} ply ${ply}: repetitionCount`);
    assert.equal(descGame.turns, sanGame.turns, `${name} ply ${ply}: turns`);
    assert.deepEqual(descGame.winState(), sanGame.winState(), `${name} ply ${ply}: winState`);

    descriptors.push(descriptor);
  }
  return descriptors;
}

function zeroMs(reply: WorkerReply): WorkerReply {
  if ('err' in reply) return reply;
  return { ...reply, stats: { ...reply.stats, ms: 0 } };
}

async function run(engine: WorkerEngine, golden: GoldenRecord, request: WorkerRequest): Promise<WorkerReply> {
  if (golden.name === 'fast-predicted-reply') {
    engine.handlePonderSettings({ id: 'history-descriptor', type: 'ponderSettings', enabled: false });
    return engine.handleFast(request, engine.defaultBatchEvaluator());
  }
  return engine.handle(request);
}

test('history_descriptor: an all-descriptor history gives the byte-identical reply to the all-SAN history, for every golden history SAN resolves unambiguously', async () => {
  const goldens = loadGoldens().filter(
    (golden) => Array.isArray(golden.request.history) && golden.request.history.length > 0,
  );
  assert.ok(goldens.length > 0, 'no golden history records to cross-check');

  let tested = 0;
  for (const golden of goldens) {
    const sanHistory = golden.request.history as string[];
    const descriptors = deriveDescriptors(golden.name, sanHistory);
    if (descriptors === null) continue;
    tested += 1;

    const sanEngine = new WorkerEngine();
    const sanReply = zeroMs(await run(sanEngine, golden, { ...golden.request, history: sanHistory }));

    const descEngine = new WorkerEngine();
    const descReply = zeroMs(await run(descEngine, golden, { ...golden.request, history: descriptors }));

    assert.deepEqual(descReply, sanReply, golden.name);
    assert.equal(JSON.stringify(descReply), JSON.stringify(sanReply), golden.name);
  }
  assert.ok(tested > 0, 'every golden history was excluded as SAN-ambiguous; nothing was actually cross-checked');
});
