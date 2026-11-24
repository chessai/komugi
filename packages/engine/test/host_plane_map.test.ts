// KOM-1168 (review F3, docs/reviews/2026-11-15-kom1162-ts-observation-v4-review.md):
// the three hosts outside worker.ts that build their own ORT calls --
// tools/e2e_driver.mjs, tools/fast_match.mjs, bench/fast.ts -- used to call
// `observation(state.game)` with no version, always encoding v3 regardless
// of what a v4 request/caller asked for. Each now threads a planeMapVersion
// (CLI flag for the two tools, a parameter for bench/fast.ts) onto every
// state it builds and encodes through the shared `src/host_encode.ts`
// helper. One test per host: a v4 state encodes 121-123 per rule_features.ts
// on a non-Advanced (beginner/intro) board, matching `observation(game,
// { planeMapVersion: 4 })`; the v3 default is byte-identical to
// `observation(game)`; an invalid version is refused with
// parsePlaneMapVersion's own error. Fake ORT session/ort only -- no real
// model, no network, no browser.

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';

import { ACTION_SIZE, MODE_INTRO, NUM_SQUARES } from '../src/constants.ts';
import type { OrtModule, OrtSession } from '../src/ort_core.ts';
import { boardFromFen } from '../src/fen.ts';
import { Game } from '../src/game.ts';
import { observation, observationSize } from '../src/observation.ts';
import { RepetitionRule } from '../src/repetition.ts';
import { RULE_FEATURES } from '../src/rule_features.ts';
import type { PlaneMapVersion } from '../src/rule_features.ts';
import { MatchGameState } from '../src/search/state.ts';

import { benchState, runFastBenchOrt } from '../bench/fast.ts';
import { makeOrtBatchLeafEvaluator, resolvePlaneMapVersion as resolveE2E } from '../tools/e2e_driver.mjs';
import { makeEvaluatorsFromSession, resolvePlaneMapVersion as resolveFastMatch } from '../tools/fast_match.mjs';

const RULE_PLANES = [121, 122, 123] as const;
const OBS_SHAPE = observationSize();
const ROW_SIZE = OBS_SHAPE.reduce((a, b) => a * b, 1);

function beginnerStart(): string {
  const url = new URL('./fixtures/observation_g3_modes.json', import.meta.url);
  const g3 = JSON.parse(readFileSync(url, 'utf8')) as { modes: { mode: string; start_fen: string }[] };
  const mode = g3.modes.find((m) => m.mode === 'beginner');
  assert.ok(mode, 'expected a beginner mode entry in the G3 fixture');
  return mode!.start_fen;
}

// Reads planes 121/122/123's (constant) value out of one encoded row.
function rulePlaneValues(row: Float32Array): number[] {
  return RULE_PLANES.map((plane) => row[plane * NUM_SQUARES]);
}

// A fake onnxruntime-web session/ort: records every row (minus the
// bucket's padding) that `runOrtBatch` fed it, and returns finite,
// shape-correct zero outputs so the real `runOrtBatch`/search code never
// trips its own finite-value guard. No real model, no network.
function fakeOrt(numActions: number): { ort: OrtModule; session: OrtSession; captured: Float32Array[] } {
  const captured: Float32Array[] = [];
  class FakeTensor {
    kind: string;
    data: Float32Array;
    dims: readonly number[];
    constructor(kind: string, data: Float32Array, dims: readonly number[]) {
      this.kind = kind;
      this.data = data;
      this.dims = dims;
    }
  }
  const ort = { Tensor: FakeTensor } as unknown as OrtModule;
  const session: OrtSession = {
    async run(feeds: Record<string, unknown>) {
      const tensor = feeds.obs as { data: Float32Array; dims: readonly number[] };
      const bucket = tensor.dims[0];
      for (let r = 0; r < bucket; r++) {
        captured.push(Float32Array.from(tensor.data.subarray(r * ROW_SIZE, (r + 1) * ROW_SIZE)));
      }
      return {
        policy: { data: new Float32Array(bucket * numActions) },
        wdl: { data: new Float32Array(bucket * 3) },
        moves_left: { data: new Float32Array(bucket) },
      };
    },
  };
  return { ort, session, captured };
}

test('e2e_driver.mjs: makeOrtBatchLeafEvaluator encodes each state at its own planeMapVersion', async () => {
  const fen = beginnerStart();
  const { ort, session, captured } = fakeOrt(ACTION_SIZE);
  const batchEvaluator = makeOrtBatchLeafEvaluator(session, ort, OBS_SHAPE);

  const game4 = new Game(boardFromFen(fen), RepetitionRule.PLACEMENT);
  const state4 = new MatchGameState(game4, { isRoot: true, planeMapVersion: 4 });
  await batchEvaluator([state4]);
  const row4 = captured.at(-1)!;
  assert.deepEqual(row4, observation(game4, { planeMapVersion: 4 }), 'v4 row must match observation(game, {planeMapVersion:4})');
  assert.deepEqual(rulePlaneValues(row4), [...RULE_FEATURES.get(game4.board.mode)!], 'v4 rule planes per rule_features.ts');
  assert.notDeepEqual(rulePlaneValues(row4), [0, 0, 0], 'beginner v4 rule planes must be non-zero');

  const game3 = new Game(boardFromFen(fen), RepetitionRule.PLACEMENT);
  const state3 = new MatchGameState(game3, { isRoot: true });
  await batchEvaluator([state3]);
  const row3 = captured.at(-1)!;
  assert.deepEqual(row3, observation(game3), 'v3 default row must be byte-identical to observation(game)');
  assert.deepEqual(rulePlaneValues(row3), [0, 0, 0], 'v3 default rule planes must stay zero');

  assert.throws(
    () => resolveE2E({ 'plane-map-version': '7' }),
    /planeMapVersion must be 3 or 4/,
    'an invalid --plane-map-version must be refused with parsePlaneMapVersion\'s error',
  );
  assert.equal(resolveE2E({}), 3, 'absent --plane-map-version defaults to 3');
});

test('fast_match.mjs: makeEvaluatorsFromSession encodes each state at its own planeMapVersion', async () => {
  const fen = beginnerStart();
  const { ort, session, captured } = fakeOrt(ACTION_SIZE);
  const { leaf } = makeEvaluatorsFromSession(session, ort, OBS_SHAPE);

  const game4 = new Game(boardFromFen(fen), RepetitionRule.PLACEMENT);
  const state4 = new MatchGameState(game4, { isRoot: false, planeMapVersion: 4 });
  await leaf(state4);
  const row4 = captured.at(-1)!;
  assert.deepEqual(row4, observation(game4, { planeMapVersion: 4 }), 'v4 row must match observation(game, {planeMapVersion:4})');
  assert.deepEqual(rulePlaneValues(row4), [...RULE_FEATURES.get(game4.board.mode)!], 'v4 rule planes per rule_features.ts');
  assert.notDeepEqual(rulePlaneValues(row4), [0, 0, 0], 'beginner v4 rule planes must be non-zero');

  const game3 = new Game(boardFromFen(fen), RepetitionRule.PLACEMENT);
  const state3 = new MatchGameState(game3, { isRoot: false });
  await leaf(state3);
  const row3 = captured.at(-1)!;
  assert.deepEqual(row3, observation(game3), 'v3 default row must be byte-identical to observation(game)');
  assert.deepEqual(rulePlaneValues(row3), [0, 0, 0], 'v3 default rule planes must stay zero');

  assert.throws(
    () => resolveFastMatch({ 'plane-map-version': '7' }),
    /planeMapVersion must be 3 or 4/,
    'an invalid --plane-map-version must be refused with parsePlaneMapVersion\'s error',
  );
  assert.equal(resolveFastMatch({}), 3, 'absent --plane-map-version defaults to 3');
});

test('bench/fast.ts: runFastBenchOrt encodes every evaluated state at its own planeMapVersion', async () => {
  // benchState()'s board is INTRO_POSITION -- MODE_INTRO, a non-Advanced
  // board whose v4 rule planes (rule_features.ts) are [1, 1, 1].
  const introFeatures = [...RULE_FEATURES.get(MODE_INTRO)!];
  assert.notDeepEqual(introFeatures, [0, 0, 0], 'sanity: intro must be a non-Advanced (non-zero) rule profile');

  {
    const { ort, session, captured } = fakeOrt(ACTION_SIZE);
    await runFastBenchOrt(session, ort, 1, 4, 1, 4);
    assert.ok(captured.length > 0, 'expected at least one encoded row');
    for (const row of captured) assert.deepEqual(rulePlaneValues(row), introFeatures, 'every v4-evaluated row carries the intro rule planes');
  }

  {
    const { ort, session, captured } = fakeOrt(ACTION_SIZE);
    await runFastBenchOrt(session, ort, 1, 4, 1); // default planeMapVersion = 3
    assert.ok(captured.length > 0, 'expected at least one encoded row');
    for (const row of captured) assert.deepEqual(rulePlaneValues(row), [0, 0, 0], 'every v3-default row leaves the rule planes at 0');
  }

  // Byte-identical to observation(game) at the root: compares benchState's
  // own root encoding (first captured row of a single-sim, batch-1 run)
  // against observation() called directly on the same fresh root game.
  {
    const { ort, session, captured } = fakeOrt(ACTION_SIZE);
    await runFastBenchOrt(session, ort, 1, 1, 1);
    const rootGame = benchState().game;
    assert.deepEqual(captured[0], observation(rootGame), 'v3-default root row must be byte-identical to observation(game)');
  }

  assert.throws(
    () => benchState(7 as PlaneMapVersion),
    /planeMapVersion must be 3 or 4/,
    'an invalid planeMapVersion must be refused with parsePlaneMapVersion\'s error',
  );
});
