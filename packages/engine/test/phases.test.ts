// KOM-1081: per-phase timing in search/worker stats. What this file pins:
// the timing is instrumentation only (a timed search computes exactly what
// an untimed one does, sync and async), the phases add up to the total, the
// evaluator share is measured through to promise settlement, the tree's
// evaluator is always restored, and the worker reports `phases` only when a
// request asks for `timing` (so the existing stats equality tests, which
// zero only `ms`, stay meaningful).

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { ADVANCED_POSITION } from '../src/constants.ts';
import { makeFakeMlhEvaluator, makeFakeMlhEvaluatorAsync } from '../src/fake_evaluator.ts';
import { boardFromFen } from '../src/fen.ts';
import { Game } from '../src/game.ts';
import { RepetitionRule } from '../src/repetition.ts';
import { MlhContext, wrapMlhEvaluator, wrapMlhEvaluatorAsync } from '../src/search/mlh.ts';
import { search, searchAsync, type MatchResult, type SearchConfig, type SearchPhases } from '../src/search/search.ts';
import { MatchGameState } from '../src/search/state.ts';
import { MatchTree, type MatchConfig, type MaybeAsyncWdlEvaluator } from '../src/search/tree.ts';
import { WorkerEngine } from '../src/engine.ts';
import type { WorkerReply, WorkerRequest } from '../src/protocol.ts';

const ACTION_SIZE = 37341;
const MLH = { lam: 0.05, tau0: 0.8, tau1: 0.9, slope: 32 };

const BASE_CONFIG: Omit<MatchConfig, 'playouts'> = {
  num_actions: ACTION_SIZE,
  contempt: 0.0,
  dynamic_cpuct: true,
  c_puct: 1.0,
  c_explore: 1.0,
  c_log: 0.45,
  c_base: 500.0,
  sigma_prior: 0.4,
  sigma_prior_weight: 2.0,
  sigma_scale: 0.85,
  sigma_floor: 0.02,
  fpu_reduction: 0.2,
  fpu_reduction_root: 0.1,
  subtree_bias_factor: 0.45,
  bias_weight_exponent: 0.85,
  bias_free_prop: 0.8,
  bias_weight_floor: 1e-6,
  k_lcb: 5.0,
  min_visit_prop_for_lcb: 0.15,
  var_floor: 0.05,
  variance_mode: 'moment',
};

const SEARCH_PARTS: (keyof SearchPhases)[] = [
  'setupMs',
  'rootEvalMs',
  'rootWinMs',
  'playoutsMs',
  'selectMs',
  'surveyMs',
  'resultMs',
];

// A clock that advances by exactly 1 per reading: phase sums are integers,
// so "the parts add up to the total" is an exact equality.
function counterClock(): { clock: () => number; now: () => number; advance: (d: number) => void } {
  let t = 0;
  return {
    clock: () => ++t,
    now: () => t,
    advance: (d: number) => {
      t += d;
    },
  };
}

function draftRoot(): MatchGameState {
  const game = new Game(boardFromFen(ADVANCED_POSITION), RepetitionRule.PLACEMENT);
  return new MatchGameState(game, { minPlacements: 10, isRoot: true });
}

// Plays the first legal action until neither side is drafting, so the
// root-win scan and the refutation survey both run.
function postDraftRoot(): MatchGameState {
  const game = new Game(boardFromFen(ADVANCED_POSITION), RepetitionRule.PLACEMENT);
  const walker = new MatchGameState(game, { minPlacements: 0, isRoot: false });
  for (let ply = 0; ply < 400 && (game.board.draftW || game.board.draftB); ply++) {
    const mask = walker.legalMask();
    const action = mask.indexOf(1);
    assert.ok(action >= 0, 'draft walk ran out of legal actions');
    walker.play(action);
    assert.ok(!walker.isTerminal(), 'draft walk reached a terminal position');
  }
  assert.ok(!game.board.draftW && !game.board.draftB, 'draft walk did not finish the draft');
  return new MatchGameState(game, { minPlacements: 0, isRoot: true });
}

function syncTree(config: MatchConfig): MatchTree {
  return new MatchTree(config, wrapMlhEvaluator(makeFakeMlhEvaluator(ACTION_SIZE), MLH, new MlhContext()));
}

function asyncTree(config: MatchConfig): MatchTree {
  return new MatchTree(config, wrapMlhEvaluatorAsync(makeFakeMlhEvaluatorAsync(ACTION_SIZE), MLH, new MlhContext()));
}

function snapshot(r: MatchResult) {
  return {
    action: r.action,
    evaluationsUsed: r.evaluationsUsed,
    playoutsUsed: r.playoutsUsed,
    nodesAtEntry: r.nodesAtEntry,
    rootWdl: Array.from(r.rootWdl),
    rootUtility: r.rootUtility,
    actions: Array.from(r.actions),
    visits: Array.from(r.visits),
    weights: Array.from(r.weights),
    lcbScores: Array.from(r.lcbScores),
    tactical: r.tactical,
  };
}

function checkPhases(p: SearchPhases | null, r: MatchResult): SearchPhases {
  assert.ok(p !== null, 'a clocked search returns phases');
  for (const k of Object.keys(p!) as (keyof SearchPhases)[]) {
    assert.ok(Number.isFinite(p![k]) && p![k] >= 0, `${k} = ${p![k]}`);
  }
  let sum = 0;
  for (const k of SEARCH_PARTS) sum += p![k];
  assert.equal(sum, p!.searchMs, 'the phases add up to searchMs');
  assert.equal(p!.evaluatorCalls, r.evaluationsUsed, 'every evaluation went through the timed wrapper');
  assert.ok(p!.evaluateMs <= p!.rootEvalMs + p!.playoutsMs, 'evaluator time is inside rootEval + playouts');
  return p!;
}

const POSITIONS: [string, () => MatchGameState][] = [
  ['draft root (survey skipped)', draftRoot],
  ['post-draft root (root-win scan + survey)', postDraftRoot],
];

for (const [label, makeRoot] of POSITIONS) {
  test(`search(): a clocked search is bit-identical to an unclocked one, phases sum exactly -- ${label}`, () => {
    const config: SearchConfig = { ...BASE_CONFIG, playouts: 24 };
    const plain = search(syncTree(config), makeRoot(), config, { discovery: true, ply: 0 });
    assert.equal(plain.phases, null, 'no clock, no phases');

    const tree = syncTree(config);
    const inner = tree.evaluator;
    const { clock } = counterClock();
    const timed = search(tree, makeRoot(), config, { discovery: true, ply: 0, clock });
    assert.deepEqual(snapshot(timed), snapshot(plain));
    assert.equal(tree.evaluator, inner, 'the tree evaluator is restored after the search');
    const p = checkPhases(timed.phases, timed);
    if (timed.tactical !== null && timed.tactical.skippedReason === null) {
      assert.ok(p.rootWinMs > 0, 'the root-win scan was timed');
    } else {
      assert.equal(p.rootWinMs, 0);
      assert.equal(p.surveyMs, 0);
    }
  });

  test(`searchAsync(): clocked == unclocked == sync, phases sum exactly -- ${label}`, async () => {
    const config: SearchConfig = { ...BASE_CONFIG, playouts: 24 };
    const reference = search(syncTree(config), makeRoot(), config, { discovery: true, ply: 0 });
    const plain = await searchAsync(asyncTree(config), makeRoot(), config, { discovery: true, ply: 0 });
    assert.equal(plain.phases, null);
    const tree = asyncTree(config);
    const inner = tree.evaluator;
    const timed = await searchAsync(tree, makeRoot(), config, { discovery: true, ply: 0, clock: counterClock().clock });
    assert.deepEqual(snapshot(plain), snapshot(reference));
    assert.deepEqual(snapshot(timed), snapshot(reference));
    assert.equal(tree.evaluator, inner);
    checkPhases(timed.phases, timed);
  });
}

test('the survey phase is timed when the refutation survey runs', () => {
  const config: SearchConfig = { ...BASE_CONFIG, playouts: 8 };
  const r = search(syncTree(config), postDraftRoot(), config, { discovery: true, ply: 0, clock: counterClock().clock });
  assert.ok(r.tactical !== null && r.tactical.skippedReason === null);
  if (r.tactical!.rootWitness === null) {
    assert.ok(r.phases!.surveyMs > 0, 'surveyMs counted');
    assert.ok(r.phases!.playoutsMs > 0, 'playoutsMs counted');
  }
});

test('evaluateMs runs until an async evaluation settles, not until it is issued', async () => {
  const config: SearchConfig = { ...BASE_CONFIG, playouts: 6 };
  const c = counterClock();
  const fake = wrapMlhEvaluator(makeFakeMlhEvaluator(ACTION_SIZE), MLH, new MlhContext());
  const slow: MaybeAsyncWdlEvaluator = (state) => {
    const value = fake(state);
    return new Promise((resolve) =>
      setImmediate(() => {
        c.advance(1000);
        resolve(value as Awaited<ReturnType<typeof fake>>);
      }),
    );
  };
  const tree = new MatchTree(config, slow);
  const r = await searchAsync(tree, draftRoot(), config, { discovery: false, ply: 0, clock: c.clock });
  const p = checkPhases(r.phases, r);
  assert.ok(p.evaluatorCalls >= 1);
  assert.ok(p.evaluateMs >= 1000 * p.evaluatorCalls, `evaluateMs ${p.evaluateMs} for ${p.evaluatorCalls} calls`);
  assert.ok(p.rootEvalMs >= 1000, 'the root evaluation is inside rootEvalMs');
});

test('the tree evaluator is restored even when the evaluator throws', async () => {
  const config: SearchConfig = { ...BASE_CONFIG, playouts: 4 };
  const boom: MaybeAsyncWdlEvaluator = () => {
    throw new Error('boom');
  };
  const tree = new MatchTree(config, boom);
  assert.throws(() => search(tree, draftRoot(), config, { discovery: false, clock: counterClock().clock }), /boom/);
  assert.equal(tree.evaluator, boom);
  const rejecting: MaybeAsyncWdlEvaluator = () => Promise.reject(new Error('later'));
  tree.evaluator = rejecting;
  await assert.rejects(searchAsync(tree, draftRoot(), config, { discovery: false, clock: counterClock().clock }), /later/);
  assert.equal(tree.evaluator, rejecting);
});

function okStats(reply: WorkerReply): Record<string, unknown> {
  assert.ok(!('err' in reply), JSON.stringify(reply));
  return (reply as { stats: Record<string, unknown> }).stats;
}

function withoutTiming(stats: Record<string, unknown>): Record<string, unknown> {
  const out: Record<string, unknown> = { ...stats, ms: 0 };
  delete out.phases;
  return out;
}

test('worker: phases only on request; prepare + search + finish == total; stats otherwise unchanged', async () => {
  const requests: WorkerRequest[] = [
    { id: 1, fen: ADVANCED_POSITION, sims: 12 },
    { id: 2, fen: ADVANCED_POSITION, sims: 12 },
  ];
  const plain = new WorkerEngine();
  const timedSync = new WorkerEngine({ clock: counterClock().clock });
  const timedAsync = new WorkerEngine({ clock: counterClock().clock });
  for (const request of requests) {
    const p = plain.handle(request);
    const s = timedSync.handle({ ...request, timing: true });
    const a = await timedAsync.handleAsync({ ...request, timing: true });
    const ps = okStats(p);
    assert.ok(!('phases' in ps), 'no phases without timing: true');
    for (const [label, reply] of [['sync', s], ['async', a]] as const) {
      const stats = okStats(reply);
      assert.deepEqual(withoutTiming(stats), withoutTiming(ps), `${label}: timing changes nothing else`);
      assert.equal((reply as { action: number }).action, (p as { action: number }).action);
      const phases = stats.phases as Record<string, number>;
      assert.ok(phases !== undefined, `${label}: phases present`);
      for (const [k, v] of Object.entries(phases)) assert.ok(Number.isFinite(v) && v >= 0, `${label}: ${k}=${v}`);
      let sum = 0;
      for (const k of SEARCH_PARTS) sum += phases[k];
      assert.equal(sum, phases.searchMs, `${label}: search parts sum`);
      assert.ok(phases.prepareMs > 0 && phases.finishMs > 0, `${label}: prepare/finish timed`);
      // totalMs spans every reading, so it is at least the three parts.
      assert.ok(phases.prepareMs + phases.searchMs + phases.finishMs <= phases.totalMs, `${label}: parts <= total`);
      assert.equal(phases.evaluatorCalls, stats.evaluations, `${label}: evaluator calls`);
      assert.equal(JSON.parse(JSON.stringify(stats)).phases.searchMs, phases.searchMs, 'JSON-safe');
    }
  }
});

test('worker: the default clock is performance.now() (real milliseconds, monotonic total)', () => {
  const engine = new WorkerEngine();
  const before = performance.now();
  const stats = okStats(engine.handle({ id: 9, fen: ADVANCED_POSITION, sims: 8, timing: true }));
  const elapsed = performance.now() - before;
  const phases = stats.phases as Record<string, number>;
  assert.ok(phases.totalMs > 0 && phases.totalMs <= elapsed, `totalMs ${phases.totalMs} within ${elapsed}`);
  const parts = phases.prepareMs + phases.searchMs + phases.finishMs;
  assert.ok(parts <= phases.totalMs + 1e-9, `parts ${parts} <= totalMs ${phases.totalMs}`);
  assert.ok(phases.totalMs - parts < 50, `totalMs ${phases.totalMs} - parts ${parts} < 50 (real ms)`);
});
