// U8: worker protocol (design doc section 2.7, section 6 U8 acceptance).
// Fake evaluator throughout (WorkerEngine's own default) -- real-net parity
// is ort.test.ts and U9's job, not this unit's.

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { ADVANCED_POSITION } from '../src/constants.ts';
import { boardFromFen, boardToFen } from '../src/fen.ts';
import { Game } from '../src/game.ts';
import { RepetitionRule } from '../src/repetition.ts';
import { sanRecords } from '../src/san.ts';
import { gameFromHistory } from '../src/root.ts';
import { WorkerEngine } from '../src/engine.ts';
import type { WorkerReply, WorkerRequest } from '../src/protocol.ts';

function isOk(reply: WorkerReply): reply is WorkerReply & { stats: object } {
  return !('err' in reply);
}

test('absent history: reply schema with S7 provenance stamps', () => {
  const engine = new WorkerEngine({ modelSha256: 'deadbeef' });
  const request: WorkerRequest = { id: 1, fen: ADVANCED_POSITION, sims: 4 };
  const reply = engine.handle(request);
  assert.ok(isOk(reply), `expected an ok reply, got ${JSON.stringify(reply)}`);
  if (!isOk(reply)) return;
  assert.equal(reply.id, 1);
  assert.equal(reply.stamp, 'absent');
  assert.equal(typeof reply.san, 'string');
  assert.equal(typeof reply.action, 'number');
  const stats = (reply as { stats: Record<string, unknown> }).stats;
  assert.equal(stats.sims, 4);
  assert.equal(stats.model_sha256, 'deadbeef');
  // Production identity is the worker's own default (task instructions):
  // discovery on, draftmin 10, the production MLH stamp.
  assert.equal(stats.discovery, 'on');
  assert.equal(stats.draftmin, 10);
  assert.equal(stats.mlh, '0.05/0.8/0.9/32');
  assert.equal(typeof stats.playouts, 'number');
  assert.equal(typeof stats.evaluations, 'number');
  assert.equal(typeof stats.nodes, 'number');
  assert.ok(Array.isArray(stats.rootWdl) && stats.rootWdl.length === 3);
  assert.equal(typeof stats.q, 'number');
  assert.equal(typeof stats.ms, 'number');
});

test('config overrides the production identity field by field', () => {
  const engine = new WorkerEngine();
  const request: WorkerRequest = {
    id: 2,
    fen: ADVANCED_POSITION,
    sims: 4,
    config: { discovery: false, draftmin: 0, mlh: { lam: 0 } },
  };
  const reply = engine.handle(request);
  assert.ok(isOk(reply));
  if (!isOk(reply)) return;
  const stats = (reply as { stats: Record<string, unknown> }).stats;
  assert.equal(stats.discovery, 'off');
  assert.equal(stats.draftmin, 0);
  // Only lam was overridden; tau0/tau1/slope stay the production identity's.
  assert.equal(stats.mlh, '0/0.8/0.9/32');
});

test('a present history: reply carries stamp present and a legal move', () => {
  const engine = new WorkerEngine();
  const san = sanRecords(boardFromFen(ADVANCED_POSITION))[0].san;
  const replayed = gameFromHistory([san]);
  const fen = boardToFen(replayed.board);
  const reply = engine.handle({ id: 3, fen, history: [san], sims: 4 });
  assert.ok(isOk(reply), `expected an ok reply, got ${JSON.stringify(reply)}`);
  if (!isOk(reply)) return;
  assert.equal(reply.stamp, 'present');
});

test('a malformed request produces an err reply, not a throw', () => {
  const engine = new WorkerEngine();
  // @ts-expect-error -- deliberately wrong shape, exactly what a hostile or
  // buggy page could send.
  const reply = engine.handle({ id: 4, fen: 12345, sims: 4 });
  assert.ok('err' in reply);
  assert.equal(reply.id, 4);
  assert.equal(typeof reply.err, 'string');
});

test('an illegal history: err never leaks out, falls back to the FEN path with stamp mismatch', () => {
  const engine = new WorkerEngine();
  const reply = engine.handle({
    id: 5,
    fen: ADVANCED_POSITION,
    history: ['not-a-legal-san'],
    sims: 4,
  });
  assert.ok(isOk(reply));
  if (!isOk(reply)) return;
  assert.equal(reply.stamp, 'mismatch');
});

// M5: a production caller may drop `history` from a certain ply on
// (HISTORYLESS_PLY); the worker only ever sees that as an absent-history
// request, so a "ply 505" request (no history at all) must answer normally
// with turns = 0 and stamp absent.
test('M5: a history-less request answers normally regardless of nominal site ply', () => {
  const engine = new WorkerEngine();
  const reply = engine.handle({ id: 6, fen: ADVANCED_POSITION, sims: 4 });
  assert.ok(isOk(reply));
  if (!isOk(reply)) return;
  assert.equal(reply.stamp, 'absent');
});

// M5: a history reaching the engine's own 512-ply cap makes the root
// terminal, which the search refuses -- the worker must answer
// `{ id, err: "turn_cap" }`, never a crash and never a move.
test('M5: a history reaching ply 512 returns err: turn_cap', () => {
  const engine = new WorkerEngine();
  // Built directly on one live Game (O(n) total) rather than by re-replaying
  // the growing history from scratch every ply: picking the first
  // enumerated (pseudo-legal) move each time is enough to reach ply 512 --
  // sanRecords never gates on terminality, so even a repetition the Game
  // itself would call a draw along the way does not stop the loop, and
  // `play()` always increments `turns` regardless (game.ts, a verbatim port
  // of Game.play_action, which does not gate on terminality either).
  const game = new Game(boardFromFen(ADVANCED_POSITION), RepetitionRule.PLACEMENT);
  const history: string[] = [];
  for (let i = 0; i < 512; i++) {
    const records = sanRecords(game.board);
    assert.ok(records.length > 0, `no legal move at ply ${i}`);
    const { id, san } = records[0];
    history.push(san);
    game.play(id);
  }
  const fen = boardToFen(game.board);
  const reply = engine.handle({ id: 7, fen, history, sims: 4 });
  assert.ok('err' in reply, `expected err: turn_cap, got ${JSON.stringify(reply)}`);
  if ('err' in reply) assert.equal(reply.err, 'turn_cap');
});

test('a negative or non-integer sims is an err reply', () => {
  const engine = new WorkerEngine();
  const bad = engine.handle({ id: 8, fen: ADVANCED_POSITION, sims: -1 });
  assert.ok('err' in bad);
  const alsoBad = engine.handle({ id: 9, fen: ADVANCED_POSITION, sims: 1.5 });
  assert.ok('err' in alsoBad);
});

function biasSnapshot(engine: WorkerEngine): Array<[number, number[], number, number]> {
  const tree = engine.tree;
  assert.ok(tree !== null, 'the engine has searched, so it holds a tree');
  if (tree === null) return [];
  return tree.bias.keys().map((key) => {
    const entry = tree.bias.entry(key)!;
    return [key, Array.from(entry.errSum), entry.weightSum, entry.gen];
  });
}

function statsOf(reply: WorkerReply): Record<string, unknown> {
  return { ...(reply as { stats: Record<string, unknown> }).stats, ms: 0 };
}

test('KOM-1076: one tree and its bias table persist across requests, like EngineServer', () => {
  // The Python match server resets the player before every request, but that
  // only clears the nodes (search.py:548-550); tree.py:253-261 keeps the
  // bias table. So a second request reads what the first one wrote.
  const first: WorkerRequest = { id: 20, fen: ADVANCED_POSITION, sims: 16 };
  const second: WorkerRequest = { id: 21, fen: ADVANCED_POSITION, sims: 16 };

  const carried = new WorkerEngine();
  assert.equal(carried.tree, null);
  assert.ok(isOk(carried.handle(first)));
  const tree = carried.tree;
  const afterFirst = biasSnapshot(carried);
  assert.ok(afterFirst.some(([, , weight]) => weight > 0), 'the first search wrote the bias table');
  const carriedReply = carried.handle(second);
  assert.ok(isOk(carriedReply));
  assert.equal(carried.tree, tree, 'the same MatchTree object answers every request');
  const carriedBias = biasSnapshot(carried);
  for (const [key] of afterFirst) {
    assert.ok(carriedBias.some(([k]) => k === key), `bias key ${key} survived the next request`);
  }

  // A fresh engine answering only the second request starts from an empty
  // table, so its table differs: the carried state is real.
  const fresh = new WorkerEngine();
  assert.ok(isOk(fresh.handle(second)));
  assert.notDeepEqual(biasSnapshot(fresh), carriedBias);

  // And the carried state is deterministic: the same request sequence on a
  // new engine reproduces the reply and the table exactly.
  const replay = new WorkerEngine();
  replay.handle(first);
  const replayReply = replay.handle(second);
  assert.ok(isOk(replayReply));
  if (!isOk(replayReply) || !isOk(carriedReply)) return;
  assert.equal(replayReply.action, carriedReply.action);
  assert.deepEqual(statsOf(replayReply), statsOf(carriedReply));
  assert.deepEqual(biasSnapshot(replay), carriedBias);
});
