// KOM-1105: rules profile plumbing (no rule change). All three profiles
// ('komugi' default, 'gungijs', 'ligungi') behave identically here -- these
// tests check the plumbing (Game.rules round-trips, rootGame/gameFromHistory
// thread it through, the worker validates/echoes/keys on it, a rules change
// resets the bias table), not any rule divergence (that is
// KOM-1106/1107/1110).

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { decodeAction } from '../src/actions.ts';
import { ACTION_SIZE, ADVANCED_POSITION } from '../src/constants.ts';
import { makeFakeMlhEvaluator } from '../src/fake_evaluator.ts';
import { boardFromFen, boardToFen } from '../src/fen.ts';
import { Game } from '../src/game.ts';
import type { MlhBatchEvaluator } from '../src/batch_evaluator.ts';
import { predictReplies } from '../src/ponder.ts';
import type { WorkerReply, WorkerRequest } from '../src/protocol.ts';
import { DEFAULT_RULES_PROFILE, RULES_PROFILES, RepetitionRule, parseRulesProfile } from '../src/repetition.ts';
import type { RulesProfile } from '../src/repetition.ts';
import { HISTORY_ABSENT, HISTORY_MISMATCH, HISTORY_PRESENT, gameFromHistory, rootGame } from '../src/root.ts';
import { buildSan, sanRecords } from '../src/san.ts';
import type { MatchNode } from '../src/search/node.ts';
import { WorkerEngine } from '../src/engine.ts';

function isOk(reply: WorkerReply): reply is WorkerReply & { action: number; san: string; stats: Record<string, unknown> } {
  return !('err' in reply);
}

function statsOf(reply: WorkerReply): Record<string, unknown> {
  const stats = { ...(reply as { stats: Record<string, unknown> }).stats };
  delete stats.ms;
  delete stats.pondered;
  return stats;
}

function fakeBatchEvaluator(): MlhBatchEvaluator {
  const leaf = makeFakeMlhEvaluator(ACTION_SIZE);
  return (states) => states.map((state) => leaf(state));
}

// A short real, legal history from the advanced start (same convention as
// ponder.test.ts/root.test.ts): always the current position's first
// enumerated legal move.
function buildHistory(n: number): { history: string[]; fen: string } {
  const history: string[] = [];
  let board = gameFromHistory(history).board;
  for (let i = 0; i < n; i++) {
    const records = sanRecords(board);
    assert.ok(records.length > 0, `no legal move at ply ${i}`);
    history.push(records[0].san);
    board = gameFromHistory(history).board;
  }
  return { history, fen: boardToFen(board) };
}

const BASE = buildHistory(2);

function baseRequest(id: number | string, overrides: Partial<WorkerRequest> = {}): WorkerRequest {
  return { id, fen: BASE.fen, history: BASE.history, sims: 16, batch: 4, ...overrides };
}

function nextRequest(
  id: number | string,
  history: readonly string[],
  engineSan: string,
  replySan: string,
  overrides: Partial<WorkerRequest> = {},
): WorkerRequest {
  const nextHistory = [...history, engineSan, replySan];
  const fen = boardToFen(gameFromHistory(nextHistory).board);
  return { id, fen, history: nextHistory, sims: 16, batch: 4, ...overrides };
}

function topPredictionSan(
  root: MatchNode,
  afterEngineMoveBoard: ReturnType<typeof gameFromHistory>['board'],
  chosenAction: number,
): string {
  const [predicted] = predictReplies(root, chosenAction, 1);
  assert.ok(predicted !== undefined, 'expected at least one predicted reply under the chosen child');
  return buildSan(afterEngineMoveBoard, decodeAction(afterEngineMoveBoard, predicted).struct);
}

// ---- game.rules round-trips through clone ----------------------------------

test('game.rules defaults to komugi and round-trips through clone', () => {
  const plain = new Game(boardFromFen(ADVANCED_POSITION), RepetitionRule.FULL);
  assert.equal(plain.rules, DEFAULT_RULES_PROFILE);
  assert.equal(plain.clone().rules, DEFAULT_RULES_PROFILE);

  const ligungi = new Game(boardFromFen(ADVANCED_POSITION), RepetitionRule.FULL, 1, 'ligungi');
  assert.equal(ligungi.rules, 'ligungi');
  assert.equal(ligungi.clone().rules, 'ligungi');

  // Playing a move and cloning again must not lose the profile either.
  const records = sanRecords(ligungi.board);
  ligungi.play(records[0].id);
  assert.equal(ligungi.clone().rules, 'ligungi');
});

// ---- all three root stamps carry `rules` into the reconstructed Game -------

test('rootGame: stamp absent carries `rules` into the FEN-path Game', () => {
  const { game, stamp } = rootGame(ADVANCED_POSITION, undefined, undefined, 'gungijs');
  assert.equal(stamp, HISTORY_ABSENT);
  assert.equal(game.rules, 'gungijs');
});

test('rootGame: stamp present carries `rules` into the replayed Game', () => {
  const draftMove = sanRecords(boardFromFen(ADVANCED_POSITION))[0].san;
  const reachedFen = boardToFen(gameFromHistory([draftMove]).board);
  const { game, stamp } = rootGame(reachedFen, [draftMove], undefined, 'ligungi');
  assert.equal(stamp, HISTORY_PRESENT);
  assert.equal(game.rules, 'ligungi');
});

test('rootGame: stamp mismatch carries `rules` into the FEN-path fallback Game', () => {
  const { game, stamp } = rootGame(ADVANCED_POSITION, ['not-a-legal-san'], undefined, 'gungijs');
  assert.equal(stamp, HISTORY_MISMATCH);
  assert.equal(game.rules, 'gungijs');
});

test('rootGame/gameFromHistory: an explicit komugi request equals an absent `rules`', () => {
  const draftMove = sanRecords(boardFromFen(ADVANCED_POSITION))[0].san;
  const withDefault = gameFromHistory([draftMove]);
  const withExplicitKomugi = gameFromHistory([draftMove], ADVANCED_POSITION, 'komugi');
  assert.equal(withDefault.rules, withExplicitKomugi.rules);
  assert.equal(boardToFen(withDefault.board), boardToFen(withExplicitKomugi.board));

  const reachedFen = boardToFen(withDefault.board);
  const a = rootGame(reachedFen, [draftMove]);
  const b = rootGame(reachedFen, [draftMove], undefined, 'komugi');
  assert.equal(a.stamp, b.stamp);
  assert.equal(a.game.rules, b.game.rules);
  assert.equal(boardToFen(a.game.board), boardToFen(b.game.board));
});

// ---- worker: validation, default-path equality, bias reset -----------------

test('WorkerEngine.handle: an invalid `rules` is a request error', () => {
  const engine = new WorkerEngine();
  // A runtime-invalid value (e.g. off a real postMessage payload), simulated
  // via a widened `string` -> `RulesProfile` cast rather than `as unknown as`.
  const invalidRules: string = 'not-a-profile';
  const request: WorkerRequest = { id: 1, fen: ADVANCED_POSITION, sims: 4, rules: invalidRules as RulesProfile };
  const reply = engine.handle(request);
  assert.ok('err' in reply, JSON.stringify(reply));
});

// KOM-1111 (verifier gap #2): `handle()`'s own validation above is not
// `handleFast()`'s -- worker.ts's handleFast re-validates independently
// (before the ponder cache lookup AND buildRoot), so this must be checked
// through that path too. KOM-1110 (ALSO): a request with NO history never
// reaches rootGame's history-replay branch at all (gameFromHistory's loop
// body is just never entered), so an earlier version of this test could
// pass even if the before-ponder validation only fired on the FEN-path --
// `history: BASE.history` forces a real (would-be) replay to exist so the
// rejection is proven to happen before that replay/buildRoot ever runs.
test('WorkerEngine.handleFast: an invalid `rules` is a request error, rejected before the ponder lookup', async () => {
  const engine = new WorkerEngine();
  const invalidRules: string = 'not-a-profile';
  const request: WorkerRequest = {
    id: 1,
    fen: BASE.fen,
    history: BASE.history,
    sims: 4,
    rules: invalidRules as RulesProfile,
  };
  const reply = await engine.handleFast(request, fakeBatchEvaluator());
  assert.ok('err' in reply, JSON.stringify(reply));
  // Rejected before ever starting (or looking up) pondering for this request.
  assert.equal(engine.ponderCacheSize, 0);
});

test('WorkerEngine.handle: an explicit komugi request is byte-identical to an absent `rules`', () => {
  const absent = new WorkerEngine();
  const explicit = new WorkerEngine();
  const replyAbsent = absent.handle({ id: 1, fen: ADVANCED_POSITION, sims: 16 });
  const replyExplicit = explicit.handle({ id: 1, fen: ADVANCED_POSITION, sims: 16, rules: 'komugi' });
  assert.ok(isOk(replyAbsent) && isOk(replyExplicit), JSON.stringify({ replyAbsent, replyExplicit }));
  if (!isOk(replyAbsent) || !isOk(replyExplicit)) return;
  assert.equal(replyAbsent.action, replyExplicit.action);
  assert.equal(replyAbsent.san, replyExplicit.san);
  assert.deepEqual(statsOf(replyAbsent), statsOf(replyExplicit));
  // The default path never echoes `rules` at all.
  assert.ok(!('rules' in replyAbsent.stats));
  assert.ok(!('rules' in replyExplicit.stats));
});

test('WorkerEngine.handle: a non-default `rules` is echoed in stats', () => {
  const engine = new WorkerEngine();
  const reply = engine.handle({ id: 1, fen: ADVANCED_POSITION, sims: 4, rules: 'ligungi' });
  assert.ok(isOk(reply), JSON.stringify(reply));
  if (!isOk(reply)) return;
  assert.equal(reply.stats.rules, 'ligungi');
});

test('a rules change resets the bias table between requests', () => {
  const engine = new WorkerEngine();
  const reply1 = engine.handle({ id: 1, fen: ADVANCED_POSITION, sims: 32 });
  assert.ok(isOk(reply1), JSON.stringify(reply1));
  const tree = engine.tree;
  assert.ok(tree !== null);
  if (tree === null) return;

  // A synthetic marker entry, independent of anything the search itself
  // would ever write under this key -- proves the NEXT request's bias
  // table is cleared because `rules` changed, not merely that the search
  // happened to decay/overwrite it.
  const MARKER_KEY = 0x7fffffff;
  tree.bias.entries.set(MARKER_KEY, { errSum: new Float64Array([1, 2, 3]), weightSum: 1, gen: 0 });
  const sizeBeforeReset = tree.bias.size;
  assert.ok(tree.bias.entries.has(MARKER_KEY));

  // sims: 0 -- only the root is evaluated; a fresh node construction always
  // reads (and so lazily creates) a zero-weight bias entry for its own key
  // as a harmless side effect (bias.ts's `generation()`), so the table is
  // not necessarily empty afterwards -- but it must be strictly SMALLER
  // than before, and the marker (and everything else request 1 wrote) must
  // be gone.
  const reply2 = engine.handle({ id: 2, fen: ADVANCED_POSITION, sims: 0, rules: 'gungijs' });
  assert.ok(isOk(reply2), JSON.stringify(reply2));
  assert.equal(engine.tree, tree, 'the one retained tree never gets replaced');
  assert.ok(!tree.bias.entries.has(MARKER_KEY), 'the bias table must be cleared when `rules` changes');
  assert.ok(
    tree.bias.size < sizeBeforeReset,
    `expected the bias table to shrink on a rules change: before=${sizeBeforeReset} after=${tree.bias.size}`,
  );
});

test('no rules change: the bias table is NOT reset between two komugi requests', () => {
  const engine = new WorkerEngine();
  const reply1 = engine.handle({ id: 1, fen: ADVANCED_POSITION, sims: 32 });
  assert.ok(isOk(reply1), JSON.stringify(reply1));
  const tree = engine.tree;
  assert.ok(tree !== null);
  if (tree === null) return;

  const MARKER_KEY = 0x7ffffffe;
  tree.bias.entries.set(MARKER_KEY, { errSum: new Float64Array([1, 2, 3]), weightSum: 1, gen: 0 });

  const reply2 = engine.handle({ id: 2, fen: ADVANCED_POSITION, sims: 0, rules: 'komugi' });
  assert.ok(isOk(reply2), JSON.stringify(reply2));
  assert.ok(tree.bias.entries.has(MARKER_KEY), 'same `rules`: the marker must survive');
});

// ---- ponder: rules is in both ponderKey sites -------------------------------

test('ponder: a hit under the SAME non-default `rules` is served from cache', async () => {
  const engine = new WorkerEngine();
  const req1 = baseRequest(1, { rules: 'ligungi' });
  const reply1 = await engine.handleFast(req1, fakeBatchEvaluator());
  assert.ok(isOk(reply1), JSON.stringify(reply1));
  if (!isOk(reply1)) return;

  await engine.ponderingPromise();
  assert.ok(engine.ponderCacheSize > 0, 'expected at least one cached predicted reply');

  const afterEngineMove = gameFromHistory([...BASE.history, reply1.san]).board;
  const root = engine.tree!.root!;
  const replySan = topPredictionSan(root, afterEngineMove, reply1.action);
  const req2 = nextRequest(2, BASE.history, reply1.san, replySan, { rules: 'ligungi' });

  const reply2 = await engine.handleFast(req2, fakeBatchEvaluator());
  assert.ok(isOk(reply2), JSON.stringify(reply2));
  if (!isOk(reply2)) return;
  assert.equal(reply2.stats.pondered, true);
  // KOM-1111 (verifier gap #1): the ponder hit's reply must still echo the
  // non-default `rules` it was served under, same as any other reply.
  assert.equal(reply2.stats.rules, 'ligungi');
});

test('ponder: pondering under one profile, then a request under another, misses', async () => {
  const engine = new WorkerEngine();
  const req1 = baseRequest(1, { rules: 'komugi' });
  const reply1 = await engine.handleFast(req1, fakeBatchEvaluator());
  assert.ok(isOk(reply1), JSON.stringify(reply1));
  if (!isOk(reply1)) return;

  await engine.ponderingPromise();
  assert.ok(engine.ponderCacheSize > 0, 'expected at least one cached predicted reply');

  const afterEngineMove = gameFromHistory([...BASE.history, reply1.san]).board;
  const root = engine.tree!.root!;
  const replySan = topPredictionSan(root, afterEngineMove, reply1.action);
  // Same history/fen/sims/batch as the predicted continuation would key on,
  // but under a DIFFERENT rules profile -- must miss.
  const req2 = nextRequest(2, BASE.history, reply1.san, replySan, { rules: 'ligungi' });

  const reply2 = await engine.handleFast(req2, fakeBatchEvaluator());
  assert.ok(isOk(reply2), JSON.stringify(reply2));
  if (!isOk(reply2)) return;
  assert.notEqual(reply2.stats.pondered, true);
});

// ---- RULES_PROFILES / parseRulesProfile -------------------------------------

test('RULES_PROFILES lists exactly komugi, gungijs, ligungi', () => {
  assert.deepEqual(RULES_PROFILES, ['komugi', 'gungijs', 'ligungi']);
});

test('parseRulesProfile returns each listed profile unchanged', () => {
  for (const profile of RULES_PROFILES) {
    assert.equal(parseRulesProfile(profile), profile);
  }
});

test('parseRulesProfile rejects an unknown value, naming exactly the three profiles', () => {
  assert.throws(
    () => parseRulesProfile('unknown'),
    /rules must be one of 'komugi'\|'gungijs'\|'ligungi', got "unknown"/,
  );
});
