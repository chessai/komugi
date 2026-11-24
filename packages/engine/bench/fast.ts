// Fast-mode Node bench (docs/design/clientside-fast-mode.md section 8 unit
// U8; ticket KOM-1094; re-review docs/reviews/2026-11-12-clientside-fast-mode-re-review.md
// M-2's encodeMs/treeMs/netMs split). Runs searchFast at batch 1/4/8/16 on
// the fake evaluator, and -- given a real session -- on a `--batch dynamic`
// ONNX model over ORT wasm. Reports sims/sec and a per-move phase split:
// netMs (the ORT run, or the fake net's arithmetic), encodeMs (observation
// encoding, or the fake evaluator's placementKeyBytes read), treeMs
// (selection/backup/overlay -- everything else) and surveyMs (the
// refutation survey, read straight off SearchPhases).
//
// WebGPU cannot run in Node (design preamble, section 7): this bench only
// ever drives the fake evaluator or ORT wasm; the browser WebGPU default is
// decided by the site bench, not here (ticket SCOPE). `numThreads` is frozen
// at a session's first `create()` (design 3.2 S-2, evaluator.ts's own
// order), so comparing 1-thread and 4-thread timings needs two separate
// process invocations, each loading its own session -- this module takes an
// already-built session and records the caller's `threads` as metadata, it
// never creates a second session itself.

import { writeFileSync } from 'node:fs';

import { ACTION_SIZE, INTRO_POSITION } from '../src/constants.ts';
import type { OrtModule, OrtSession } from '../src/ort_core.ts';
import { fakeLogits, fakeWdl, fnv1a32 } from '../src/fake_evaluator.ts';
import { boardFromFen } from '../src/fen.ts';
import { Game } from '../src/game.ts';
import { encodeState } from '../src/host_encode.ts';
import { observationSize } from '../src/observation.ts';
import { RepetitionRule } from '../src/repetition.ts';
import type { PlaneMapVersion } from '../src/rule_features.ts';
import { runOrtBatch } from '../src/batch_evaluator.ts';
import {
  searchFast,
  searchFastAsync,
  type MatchResult,
  type SearchConfig,
  type SearchOptions,
} from '../src/search/search.ts';
import { MatchGameState, type MatchState, type WdlEval } from '../src/search/state.ts';
import {
  MatchTree,
  type BatchWdlEvaluator,
  type MaybeAsyncWdlEvaluator,
  type MatchConfig as TreeConfig,
} from '../src/search/tree.ts';

// worker.ts's BASE_MATCH_CONFIG (src/komugi/search/match/config.py's own
// defaults), duplicated rather than imported: worker.ts does not export it,
// and this bench sits outside the product-identity layer (discovery/
// draftmin/mlh) worker.ts/protocol.ts own.
const BASE_MATCH_CONFIG: Omit<TreeConfig, 'playouts'> = {
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

// The worker's own evaluation ceiling (worker.ts's CEILING_SLACK, tree_batch
// test's own bound): a search never asks for more than `sims + 2`
// evaluations.
export const EVAL_CEILING_SLACK = 2;

export function benchConfig(sims: number): SearchConfig {
  return { ...BASE_MATCH_CONFIG, playouts: sims };
}

// A fixed, non-drafting position (INTRO_POSITION's 5th FEN field is '-':
// fen.ts's own drafting parse, so draftIsOpen is false and the refutation
// survey actually runs, unlike constants.ts's ADVANCED_POSITION which is
// mid-draft): wide enough that the playout loop and the survey both run,
// every time.
// `planeMapVersion` (KOM-1168, default 3): carried onto the root state, and
// through every clone the search takes of it, so a caller benching a v4
// model encodes v4 rather than an assumed v3.
export function benchState(planeMapVersion: PlaneMapVersion = 3): MatchGameState {
  const board = boardFromFen(INTRO_POSITION);
  const game = new Game(board, RepetitionRule.PLACEMENT);
  return new MatchGameState(game, { isRoot: true, planeMapVersion });
}

export const BENCH_SEARCH_OPTS: SearchOptions = { discovery: true };

export type PhaseSplit = {
  netMs: number;
  encodeMs: number;
  treeMs: number;
  surveyMs: number;
};

export type FastBenchResult = {
  backend: 'fake' | 'ort-wasm';
  threads: number | null;
  batch: number;
  sims: number;
  playoutsUsed: number;
  evaluationsUsed: number;
  totalMs: number;
  simsPerSec: number;
  phases: PhaseSplit;
};

function emptySplit(): PhaseSplit {
  return { netMs: 0, encodeMs: 0, treeMs: 0, surveyMs: 0 };
}

// Turns one MatchResult + the wall time it took into a FastBenchResult,
// deriving treeMs as the complement of everything independently timed
// (design M-2: netMs/encodeMs come from the evaluator wrapper below, surveyMs
// from SearchPhases) -- so the four reported numbers always sum to `totalMs`
// to within clock jitter, never by construction overclaiming agreement.
function toBenchResult(
  backend: 'fake' | 'ort-wasm',
  threads: number | null,
  batch: number,
  sims: number,
  result: MatchResult,
  totalMs: number,
  acc: PhaseSplit,
): FastBenchResult {
  const surveyMs = result.phases === null ? 0 : result.phases.surveyMs;
  const split: PhaseSplit = {
    netMs: acc.netMs,
    encodeMs: acc.encodeMs,
    surveyMs,
    treeMs: Math.max(0, totalMs - acc.netMs - acc.encodeMs - surveyMs),
  };
  return {
    backend,
    threads,
    batch,
    sims,
    playoutsUsed: result.playoutsUsed,
    evaluationsUsed: result.evaluationsUsed,
    totalMs,
    simsPerSec: totalMs > 0 ? (result.playoutsUsed * 1000) / totalMs : 0,
    phases: split,
  };
}

// The fake "net": fnv1a32 + fakeLogits/fakeWdl, timed separately from the
// (trivial) "encode" step -- placementKeyBytes -- so a fake-evaluator bench
// run exercises the same net/encode/tree split a real backend does, pinning
// the schema before a real model is involved. Both the single-leaf seam
// (the root call, and every playout at batch 1) and the batch seam (every
// playout at batch > 1) accumulate into the same `acc`, since searchFast
// routes the root through the single-leaf seam regardless of batch (design
// 6.3 S-1's shared-seam note).
function fakeEvaluators(
  numActions: number,
  acc: PhaseSplit,
): { single: MaybeAsyncWdlEvaluator; batch: BatchWdlEvaluator } {
  const evalOne = (state: MatchState): WdlEval => {
    const t0 = performance.now();
    const key = state.placementKeyBytes();
    const t1 = performance.now();
    acc.encodeMs += t1 - t0;
    const h = fnv1a32(key);
    const out = { policyLogits: fakeLogits(h, numActions), wdl: fakeWdl(h) };
    acc.netMs += performance.now() - t1;
    return out;
  };
  return {
    single: evalOne,
    batch: (states: MatchState[]): WdlEval[] => states.map(evalOne),
  };
}

// Batch 1/4/8/16 on the fake evaluator (ticket SCOPE's first bullet).
export function runFastBenchFake(batch: number, sims: number, planeMapVersion: PlaneMapVersion = 3): FastBenchResult {
  const acc = emptySplit();
  const config = benchConfig(sims);
  const { single, batch: batchEvaluator } = fakeEvaluators(config.num_actions, acc);
  const tree = new MatchTree(config, single, batchEvaluator);
  const state = benchState(planeMapVersion);
  const started = performance.now();
  const result = searchFast(tree, state, config, { ...BENCH_SEARCH_OPTS, clock: () => performance.now() }, batch);
  const totalMs = performance.now() - started;
  return toBenchResult('fake', null, batch, sims, result, totalMs, acc);
}

// A real `--batch dynamic` ONNX model on ORT wasm (ticket SCOPE's second
// bullet). `threads` is metadata only -- the caller already built `session`
// with its own `loadOrtSession({ threads })` call, since numThreads is
// frozen at the session's first `create()` (design 3.2 S-2).
export async function runFastBenchOrt(
  session: OrtSession,
  ort: OrtModule,
  batch: number,
  sims: number,
  threads: number,
  planeMapVersion: PlaneMapVersion = 3,
): Promise<FastBenchResult> {
  const acc = emptySplit();
  const obsShape = observationSize();
  const rowSize = obsShape.reduce((a, b) => a * b, 1);
  const config = benchConfig(sims);

  // KOM-1168: encoded at the state's own planeMapVersion (host_encode.ts),
  // not an assumed v3.
  const encodeRow = (state: MatchState): Float32Array => {
    const t0 = performance.now();
    const obs = encodeState(state);
    acc.encodeMs += performance.now() - t0;
    return obs;
  };

  const single: MaybeAsyncWdlEvaluator = async (state: MatchState): Promise<WdlEval> => {
    const obs = encodeRow(state);
    const t0 = performance.now();
    const rows = await runOrtBatch(session, ort, obs, 1, obsShape);
    acc.netMs += performance.now() - t0;
    return { policyLogits: rows[0].policyLogits, wdl: rows[0].wdl };
  };

  const batchEvaluator: BatchWdlEvaluator = async (states: MatchState[]): Promise<WdlEval[]> => {
    const obsRows = new Float32Array(states.length * rowSize);
    states.forEach((s, i) => obsRows.set(encodeRow(s), i * rowSize));
    const t0 = performance.now();
    const rows = await runOrtBatch(session, ort, obsRows, states.length, obsShape);
    acc.netMs += performance.now() - t0;
    return rows.map((r) => ({ policyLogits: r.policyLogits, wdl: r.wdl }));
  };

  const tree = new MatchTree(config, single, batchEvaluator);
  const state = benchState(planeMapVersion);
  const started = performance.now();
  const result = await searchFastAsync(
    tree,
    state,
    config,
    { ...BENCH_SEARCH_OPTS, clock: () => performance.now() },
    batch,
  );
  const totalMs = performance.now() - started;
  return toBenchResult('ort-wasm', threads, batch, sims, result, totalMs, acc);
}

// Saves a batch of results as JSON, for the advisor's post-merge gen-11 run
// (AFTER MERGE). The caller always passes a path under a temp directory --
// this function never picks one itself, so it writes exactly where it is
// told and nowhere else.
export function writeReport(results: FastBenchResult[], path: string): void {
  writeFileSync(path, JSON.stringify({ results }, null, 2));
}
