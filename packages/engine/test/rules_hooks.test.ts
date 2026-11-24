// KOM-1203 unit 6 (design docs/design/kom1203-rules-config.md section 4, AC4):
// the engine-scoped scoreTerminal hook. Section 4's five numbered tests --
// test 1 (the predicate-matches-exactly-what-it-should corpus, across every
// TerminalReason plus non-terminal/draft) is driven directly against
// Game.winState() (the exact seam section 4's own pseudocode hooks into; a
// corpus this wide has no equivalent root-only path through WorkerEngine,
// since a terminal ROOT position fails buildRoot before any search runs).
// Tests 2-5 drive a real WorkerEngine/installWorkerHost with the
// engine-scoped option, never through WorkerRequest.rules -- matching the
// design's own closing line for section 4.

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { Board, cellIndex, makeCode } from '../src/board.ts';
import { ACTION_SIZE, ADVANCED_POSITION, MARSHAL, MODE_ADVANCED, SOLDIER, WHITE, BLACK } from '../src/constants.ts';
import { boardFromFen } from '../src/fen.ts';
import { Game } from '../src/game.ts';
import { RepetitionRule } from '../src/repetition.ts';
import { RULES_PRESETS } from '../src/rules_config.ts';
import type { AbsoluteWdl, ScoreTerminalHook, TerminalReason } from '../src/rules_config.ts';
import { sanRecords } from '../src/san.ts';
import { WorkerEngine } from '../src/engine.ts';
import type { WorkerInitEnv } from '../src/engine.ts';
import { installWorkerHost } from '../src/worker_host.ts';
import type { ProbeSession, SessionFactory } from '../src/backend.ts';
import type { MlhBatchEvaluator } from '../src/batch_evaluator.ts';
import { makeFakeMlhEvaluator } from '../src/fake_evaluator.ts';
import type { WorkerInitReply, WorkerInitRequest, WorkerPonderNowReply, WorkerPonderSettingsReply, WorkerReply, WorkerRequest } from '../src/protocol.ts';

// ---- shared board builders (mirrors test/root.test.ts's own local helpers) --

function emptyBoard(): Board {
  const b = new Board();
  b.mode = MODE_ADVANCED;
  return b;
}

function place(board: Board, rank: number, file: number, ti: number, color: number, tier: number): void {
  board.cells[cellIndex(rank - 1, file - 1, tier - 1)] = makeCode(ti, color);
}

function buildTwoMarshalBoard(): Board {
  const b = emptyBoard();
  b.turn = WHITE;
  place(b, 1, 1, MARSHAL, WHITE, 1);
  place(b, 9, 9, MARSHAL, BLACK, 1);
  return b;
}

function buildNonTerminalBoard(): Board {
  const b = buildTwoMarshalBoard();
  place(b, 5, 5, SOLDIER, WHITE, 1);
  place(b, 5, 4, SOLDIER, BLACK, 1);
  return b;
}

function buildMarshalCapturedBoard(): Board {
  const b = emptyBoard();
  b.turn = WHITE;
  place(b, 1, 1, MARSHAL, WHITE, 1);
  return b;
}

// ---- 1. terminal-reason corpus (one Game per TerminalReason, plus
// non-terminal and draft) --------------------------------------------------

// Reused fixture positions (ligungi_terminal.test.ts): a false mate that is a
// genuine checkmate once marshalCaptureIsEscape is off (RULES_PRESETS.gungijs
// shares every other komugi knob value, so this is the 'gungijs' preset
// itself, not a one-off literal).
const FALSE_MATE_FEN =
  'r1|s:n|m2r|s:g|1/4k|a:J|j2/|n:K:F|1ww1f3/1nt6/2c6/a8/1|D:D:G||S:d|6' + '/1|N:N||S:M|6/9 I1R1F1C1A1T1/d2 w 3 - 33';
const MATING_LOOKING_MOVE = '小(2-4-2)(2-5-2)付';

function buildCheckmateGame(): Game {
  const g = new Game(boardFromFen(FALSE_MATE_FEN), RepetitionRule.FULL, 1, RULES_PRESETS.gungijs);
  const rec = sanRecords(g.board).find((r) => r.san === MATING_LOOKING_MOVE);
  assert.ok(rec !== undefined, 'expected the mating-looking move to be found by SAN');
  g.play(rec!.id);
  return g;
}

// Black's only winning move leaves White with no legal move and not in
// check -- a genuine stalemate (ligungi_terminal.test.ts's own fixture).
const PRE_STALEMATE_FEN = '5g1|d:s|M/7d|d:s|/9/9/9/9/9/9/m8 -/- b 3 - 1';
const STALEMATING_MOVE = '兵(2-2-1)(3-2-1)';

function buildStalemateGame(): Game {
  const g = new Game(boardFromFen(PRE_STALEMATE_FEN), RepetitionRule.FULL, 1, 'ligungi');
  const rec = sanRecords(g.board).find((r) => r.san === STALEMATING_MOVE);
  assert.ok(rec !== undefined, 'expected the stalemating move to be found by SAN');
  g.play(rec!.id);
  return g;
}

// The test-only bridge constructor param (`initialOccurrences`, game.ts's own
// doc comment) reaches a repetition count without having played it.
function buildRepetitionGame(): Game {
  return new Game(buildNonTerminalBoard(), RepetitionRule.FULL, 4, RULES_PRESETS.komugi);
}

function buildInsufficientGame(): Game {
  return new Game(buildTwoMarshalBoard(), RepetitionRule.FULL, 1, RULES_PRESETS.komugi);
}

function buildMarshalCapturedGame(): Game {
  return new Game(buildMarshalCapturedBoard(), RepetitionRule.FULL, 1, RULES_PRESETS.komugi);
}

function buildPlyCapGame(): Game {
  const g = new Game(buildNonTerminalBoard(), RepetitionRule.FULL, 1, RULES_PRESETS.komugi);
  g.turns = 512;
  return g;
}

function buildNonTerminalGame(): Game {
  return new Game(buildNonTerminalBoard(), RepetitionRule.FULL, 1, RULES_PRESETS.komugi);
}

function buildDraftGame(): Game {
  const board = buildNonTerminalBoard();
  board.draftW = 1;
  return new Game(board, RepetitionRule.FULL, 1, RULES_PRESETS.komugi);
}

const CORPUS: ReadonlyArray<{ label: string; reason: TerminalReason | null; build: () => Game }> = [
  { label: 'checkmate', reason: 'checkmate', build: buildCheckmateGame },
  { label: 'stalemate', reason: 'stalemate', build: buildStalemateGame },
  { label: 'repetition', reason: 'repetition', build: buildRepetitionGame },
  { label: 'insufficient', reason: 'insufficient', build: buildInsufficientGame },
  { label: 'marshal-captured', reason: 'marshal-captured', build: buildMarshalCapturedGame },
  { label: 'ply-cap', reason: 'ply-cap', build: buildPlyCapGame },
  { label: 'non-terminal', reason: null, build: buildNonTerminalGame },
  { label: 'draft', reason: null, build: buildDraftGame },
];

test('AC4 test 1: scoreTerminal changes winState exactly on the reasons its predicate matches (one position per TerminalReason, plus non-terminal/draft)', () => {
  const MATCHED = new Set<TerminalReason>(['checkmate', 'ply-cap']);
  const hook: ScoreTerminalHook = (ctx, defaultResult) => {
    if (!MATCHED.has(ctx.reason)) return undefined;
    // A deliberately different, still one-hot, override -- whichever of the
    // two decisive slots the default result did NOT set.
    return defaultResult[0] === 1 ? [0, 1, 0] : [1, 0, 0];
  };

  let matchedCount = 0;
  for (const entry of CORPUS) {
    const without = entry.build().winState();
    const withHookGame = entry.build();
    withHookGame.scoreTerminal = hook;
    const withHook = withHookGame.winState();

    if (entry.reason !== null && MATCHED.has(entry.reason)) {
      matchedCount += 1;
      assert.notDeepEqual(withHook, without, `${entry.label}: expected the hook's override to change winState`);
    } else {
      assert.deepEqual(
        withHook,
        without,
        `${entry.label}: winState must be unchanged (non-terminal/draft never reaches the hook; an unmatched reason returns undefined)`,
      );
    }
  }
  assert.ok(matchedCount > 0, 'expected the matched set (checkmate, ply-cap) to be non-empty');
});

// ---- 2. a stalemate-as-loss hook changes the search's recorded root
// witness, matching what ligungi's native stalemateResult knob does without
// any hook at all (section 4's AC4 test 2) ----------------------------------

test('AC4 test 2: an engine-scoped stalemate-as-loss hook makes a komugi search witness the stalemating move, exactly like ligungi does natively', () => {
  const probe = new Game(boardFromFen(PRE_STALEMATE_FEN), RepetitionRule.FULL, 1, 'komugi');
  const rec = sanRecords(probe.board).find((r) => r.san === STALEMATING_MOVE);
  assert.ok(rec !== undefined, 'expected the stalemating move to be found by SAN');

  const stalemateAsLoss: ScoreTerminalHook = (ctx) => {
    if (ctx.reason !== 'stalemate') return undefined;
    const result: [number, number, number] = [0, 0, 0];
    result[1 - ctx.board.turn] = 1;
    return result;
  };

  const withHook = new WorkerEngine({ scoreTerminal: stalemateAsLoss });
  const hooked = withHook.handle({ id: 1, fen: PRE_STALEMATE_FEN, sims: 1, discovery: true, rules: 'komugi' });
  assert.ok(!('err' in hooked), JSON.stringify(hooked));
  if ('err' in hooked) return;
  assert.equal(hooked.stats.tactical?.rootWitness, rec!.id, 'the stalemate-as-loss hook must witness the stalemating move');

  const withoutHook = new WorkerEngine();
  const plain = withoutHook.handle({ id: 1, fen: PRE_STALEMATE_FEN, sims: 1, discovery: true, rules: 'komugi' });
  assert.ok(!('err' in plain), JSON.stringify(plain));
  if ('err' in plain) return;
  assert.equal(plain.stats.tactical?.rootWitness, null, 'without the hook, komugi scores the position a draw -- no witness');
});

// ---- 3. a control hook whose predicate never matches leaves the reply
// byte-identical to no hook at all (section 4's AC4 test 3) ----------------

test('AC4 test 3: a hook whose predicate never matches this search leaves the reply byte-identical (after stripping timing)', () => {
  let matchCount = 0;
  // A condition this search's terminal nodes (the advanced start position,
  // 16 simulations deep) never satisfies.
  const controlHook: ScoreTerminalHook = (ctx) => {
    if (ctx.reason === 'ply-cap' && ctx.turns > 100_000) {
      matchCount += 1;
      return [1, 0, 0];
    }
    return undefined;
  };

  const request: WorkerRequest = { id: 1, fen: ADVANCED_POSITION, sims: 16 };
  const replyWith = new WorkerEngine({ scoreTerminal: controlHook }).handle(request);
  const replyWithout = new WorkerEngine().handle(request);

  const strip = (reply: WorkerReply): unknown => {
    if ('err' in reply) return reply;
    return { ...reply, stats: { ...reply.stats, ms: 0 } };
  };
  assert.deepEqual(strip(replyWith), strip(replyWithout));
  assert.equal(matchCount, 0, "the control hook's own predicate must never match during this search");
});

// ---- 4. a hook returning a non-one-hot vector errors the request, instead
// of silently corrupting the result (section 4's AC4 test 4) ---------------

test('AC4 test 4: a hook returning [0,0,0] makes the request that reaches it return an error, not a corrupted result', () => {
  const badHook: ScoreTerminalHook = () => [0, 0, 0] as AbsoluteWdl;
  const engine = new WorkerEngine({ scoreTerminal: badHook });
  // discovery:true's root-win scan calls winState() on the stalemating child
  // during this 1-sim search -- the same seam AC4 test 2 exercises, now with
  // a hook that always rejects.
  const reply = engine.handle({ id: 1, fen: PRE_STALEMATE_FEN, sims: 1, discovery: true, rules: 'ligungi' });
  assert.ok('err' in reply, `expected an error reply, got ${JSON.stringify(reply)}`);
});

// ---- 5. installWorkerHost applies the engine-scoped hook to a request that
// was itself passed through structuredClone (section 4's AC4 test 5) -------

type Posted = WorkerReply | WorkerInitReply | WorkerPonderSettingsReply | WorkerPonderNowReply;

function makeFakeScope(): {
  scope: { addEventListener: (type: 'message', listener: (event: { data: unknown }) => void) => void; postMessage: (reply: Posted) => void };
  posted: Posted[];
  send: (data: unknown) => void;
} {
  const listeners: Array<(event: { data: unknown }) => void> = [];
  const posted: Posted[] = [];
  const scope = {
    addEventListener(_type: 'message', listener: (event: { data: unknown }) => void): void {
      listeners.push(listener);
    },
    postMessage(reply: Posted): void {
      posted.push(reply);
    },
  };
  const send = (data: unknown): void => {
    for (const listener of listeners) listener({ data });
  };
  return { scope, posted, send };
}

function fakeProbeSession(): ProbeSession {
  return { warmup: async () => undefined, run: async () => undefined };
}

function fakeSessionFactory(): SessionFactory {
  return {
    setNumThreads: () => undefined,
    createWebgpu: async () => fakeProbeSession(),
    createWasm: async () => fakeProbeSession(),
  };
}

function spyBatchEvaluator(): MlhBatchEvaluator {
  const leaf = makeFakeMlhEvaluator(ACTION_SIZE);
  return (states) => states.map((state) => leaf(state));
}

function fakeInitEnv(evaluator: MlhBatchEvaluator): WorkerInitEnv {
  return { sessionFactory: fakeSessionFactory(), createBatchBackend: async () => evaluator };
}

async function flushUntil(predicate: () => boolean, maxTicks = 10_000): Promise<void> {
  for (let i = 0; i < maxTicks; i++) {
    if (predicate()) return;
    await Promise.resolve();
  }
}

test('AC4 test 5: installWorkerHost applies the engine-scoped hook to a request passed through structuredClone', async () => {
  const stalemateAsLoss: ScoreTerminalHook = (ctx) => {
    if (ctx.reason !== 'stalemate') return undefined;
    const result: [number, number, number] = [0, 0, 0];
    result[1 - ctx.board.turn] = 1;
    return result;
  };

  const probe = new Game(boardFromFen(PRE_STALEMATE_FEN), RepetitionRule.FULL, 1, 'komugi');
  const rec = sanRecords(probe.board).find((r) => r.san === STALEMATING_MOVE);
  assert.ok(rec !== undefined, 'expected the stalemating move to be found by SAN');

  const { scope, posted, send } = makeFakeScope();
  installWorkerHost(scope, { makeInitEnv: async () => fakeInitEnv(spyBatchEvaluator()), scoreTerminal: stalemateAsLoss });

  send({ id: 'init-1', type: 'init', modelUrl: 'model.onnx', modelSha256: 'sha-a' });
  const request: WorkerRequest = { id: 'search-1', fen: PRE_STALEMATE_FEN, sims: 1, batch: 1, discovery: true, rules: 'komugi' };
  send(structuredClone(request));
  await flushUntil(() => posted.length >= 2);

  assert.equal(posted.length, 2, `expected an init reply and a search reply, got ${JSON.stringify(posted)}`);
  const [, searchReply] = posted;
  assert.ok(searchReply !== undefined && !('err' in searchReply), `search should succeed, got ${JSON.stringify(searchReply)}`);
  if (searchReply === undefined || 'err' in searchReply) return;
  assert.equal(
    (searchReply as WorkerReply & { stats: { tactical?: { rootWitness: number | null } } }).stats.tactical?.rootWitness,
    rec!.id,
    'the structuredClone-passed request must still be scored by the engine-scoped hook',
  );
});
