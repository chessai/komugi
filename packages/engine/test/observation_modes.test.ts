// KOM-1162 (docs/design/multi-mode-training.md rev 3, section 5.1): the G3
// four-mode golden (komugi KOM-1154's tests/encoding/goldens/
// observation_g3_modes.json, generated at plane-map version 3 under LIGUNGI)
// replayed through the TS Game from each mode's own start FEN. Per ply:
//   - planeMapVersion 3: all 128 per-plane digests match;
//   - planeMapVersion 4: every digest except 121-123 matches, and 121-123
//     are constant planes of RULE_FEATURES[mode] -- checked against the
//     design's 1.2 table written out again below, not against Python.
// Plus the plumbing ACs: an invalid planeMapVersion throws (encoder and
// request), and the ponder key differs when only the version differs.

import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';

import { decodeAction } from '../src/actions.ts';
import { ACTION_SIZE, ADVANCED_POSITION, MODE_ADVANCED, MODE_BEGINNER, MODE_INTERMEDIATE, MODE_INTRO, NUM_SQUARES } from '../src/constants.ts';
import { makeFakeMlhEvaluator } from '../src/fake_evaluator.ts';
import { boardFromFen, boardToFen, canonicalFen } from '../src/fen.ts';
import { Game } from '../src/game.ts';
import type { MlhBatchEvaluator } from '../src/batch_evaluator.ts';
import { OBS_PLANES, observation } from '../src/observation.ts';
import { ponderKey, predictReplies } from '../src/ponder.ts';
import type { PonderKeyInput } from '../src/ponder.ts';
import { RepetitionRule } from '../src/repetition.ts';
import { gameFromHistory } from '../src/root.ts';
import { RULE_FEATURES, fillRulePlanes, parsePlaneMapVersion } from '../src/rule_features.ts';
import type { PlaneMapVersion } from '../src/rule_features.ts';
import { buildSan, sanRecords } from '../src/san.ts';
import { MatchGameState } from '../src/search/state.ts';
import type { WorkerRequest } from '../src/protocol.ts';
import { WorkerEngine } from '../src/engine.ts';

const FIXTURE_URL = new URL('./fixtures/observation_g3_modes.json', import.meta.url);
// Owner-pinned (ticket KOM-1162 owner step, 2026-11-15).
const FIXTURE_SHA256 = '1179b1dec1f82fb4110ed0b0edf0b046562ae6c186cc32e7913a90d0acaac7f0';

const RULE_PLANES = [121, 122, 123];

// Design 1.2, written out independently of src/rule_features.ts:
// mode -> (tier_cap_2, marshal_no_stack, fixed_setup).
const HAND_TABLE: Record<string, { code: number; features: [number, number, number] }> = {
  intro: { code: MODE_INTRO, features: [1, 1, 1] },
  beginner: { code: MODE_BEGINNER, features: [1, 1, 1] },
  intermediate: { code: MODE_INTERMEDIATE, features: [1, 0, 0] },
  advanced: { code: MODE_ADVANCED, features: [0, 0, 0] },
};

type G3Ply = {
  mode: string;
  ply: number;
  action: number | null;
  fen: string;
  turns: number;
  repetition_count: number;
  progress_plies: number;
  last_move: [number, number, number, number] | null;
  planes: string[];
  rule_planes: number[];
};

type G3 = {
  manifest: { plane_map_version: number; n_planes: number; engine_rules: string };
  modes: { mode: string; start_fen: string; plies: number }[];
  plies: G3Ply[];
};

function loadG3(): G3 {
  const bytes = readFileSync(FIXTURE_URL);
  const digest = createHash('sha256').update(bytes).digest('hex');
  assert.equal(digest, FIXTURE_SHA256, 'observation_g3_modes.json is not the owner-pinned file');
  return JSON.parse(bytes.toString('utf8')) as G3;
}

// sha256(plane float32 bytes, C order)[:8] -- the manifest's digest.
function planeDigest(obs: Float32Array, plane: number): string {
  const view = Buffer.from(obs.buffer, obs.byteOffset + plane * NUM_SQUARES * 4, NUM_SQUARES * 4);
  return createHash('sha256').update(view).digest('hex').slice(0, 8);
}

function constantPlane(value: number): string {
  return planeDigest(new Float32Array(OBS_PLANES * NUM_SQUARES).fill(value), 0);
}

// Replays every mode's plies through the TS Game and calls `check` with the
// game standing at each recorded ply (ply 0 = the start FEN).
function replay(g3: G3, check: (ply: G3Ply, game: Game, label: string) => void): number {
  let seen = 0;
  for (const m of g3.modes) {
    const plies = g3.plies.filter((p) => p.mode === m.mode).sort((a, b) => a.ply - b.ply);
    // `modes[].plies` counts records, ply 0 (the start FEN) included.
    assert.equal(plies.length, m.plies, `${m.mode}: ply count`);
    assert.deepEqual(plies.map((p) => p.ply), [...Array(m.plies).keys()], `${m.mode}: contiguous plies`);
    assert.equal(plies[0].action, null, `${m.mode}: ply 0 has no action`);
    assert.equal(canonicalFen(plies[0].fen), canonicalFen(m.start_fen), `${m.mode}: ply 0 is the start FEN`);
    // G3 is generated under LIGUNGI, whose repetition key is FULL (root.ts);
    // LIGUNGI is a subset of LEGACY, so the default profile accepts every ply.
    const game = new Game(boardFromFen(m.start_fen), RepetitionRule.FULL);
    for (const ply of plies) {
      const label = `${m.mode}#ply${ply.ply}`;
      if (ply.action !== null) {
        assert.equal(game.validMoves()[ply.action], 1, `${label}: action ${ply.action} is not legal in TS`);
        game.play(ply.action);
      }
      assert.equal(canonicalFen(boardToFen(game.board)), canonicalFen(ply.fen), `${label}: fen`);
      assert.equal(game.turns, ply.turns, `${label}: turns`);
      assert.equal(game.repetitionCount(), ply.repetition_count, `${label}: repetition_count`);
      assert.equal(game.progressPlies, ply.progress_plies, `${label}: progress_plies`);
      const lm = game.lastMove;
      assert.deepEqual(lm === null ? null : [lm.fromSq, lm.fromTier, lm.toSq, lm.toTier], ply.last_move, `${label}: last_move`);
      check(ply, game, label);
      seen++;
    }
  }
  return seen;
}

test('G3 golden: manifest, four modes, mode digits and the hand table agree', () => {
  const g3 = loadG3();
  assert.equal(g3.manifest.plane_map_version, 3);
  assert.equal(g3.manifest.n_planes, OBS_PLANES);
  assert.deepEqual(g3.modes.map((m) => m.mode).sort(), Object.keys(HAND_TABLE).sort());
  for (const m of g3.modes) {
    assert.equal(boardFromFen(m.start_fen).mode, HAND_TABLE[m.mode].code, `${m.mode}: FEN mode digit`);
    assert.deepEqual([...RULE_FEATURES.get(HAND_TABLE[m.mode].code)!], HAND_TABLE[m.mode].features, `${m.mode}: RULE_FEATURES`);
  }
  // A v3 golden's rule slice (121-127) is all zero.
  for (const p of g3.plies) assert.deepEqual(p.rule_planes, [0, 0, 0, 0, 0, 0, 0], `${p.mode}#ply${p.ply}: rule_planes`);
});

test('G3 golden at planeMapVersion 3: all 128 plane digests match every ply', () => {
  const g3 = loadG3();
  let mismatches = 0;
  let first = '';
  const seen = replay(g3, (ply, game, label) => {
    for (const obs of [observation(game), observation(game, { planeMapVersion: 3 })]) {
      for (let k = 0; k < OBS_PLANES; k++) {
        if (planeDigest(obs, k) !== ply.planes[k]) {
          mismatches++;
          if (!first) first = `${label}#plane${k}`;
        }
      }
    }
  });
  console.log(`observation G3 v3: plies=${seen} mismatches=${mismatches} first=${first || '-'}`);
  assert.equal(seen, g3.plies.length);
  assert.equal(mismatches, 0, `first mismatch at ${first}`);
});

test('G3 golden at planeMapVersion 4: 121-123 are the hand-table constants, every other digest matches', () => {
  const g3 = loadG3();
  const zero = constantPlane(0);
  const one = constantPlane(1);
  let mismatches = 0;
  let first = '';
  const out = new Float32Array(OBS_PLANES * NUM_SQUARES).fill(7);
  const seen = replay(g3, (ply, game, label) => {
    const want = HAND_TABLE[ply.mode].features;
    const obs = observation(game, { planeMapVersion: 4, out });
    assert.equal(obs, out, 'must write into the supplied buffer');
    for (let k = 0; k < OBS_PLANES; k++) {
      const idx = RULE_PLANES.indexOf(k);
      const expected = idx >= 0 ? (want[idx] === 1 ? one : zero) : ply.planes[k];
      if (planeDigest(obs, k) !== expected) {
        mismatches++;
        if (!first) first = `${label}#plane${k}`;
      }
    }
    // Advanced is byte-equal to v3 (design 1.2 "deviation from Advanced").
    if (ply.mode === 'advanced') {
      assert.deepEqual(obs, observation(game, { planeMapVersion: 3 }), `${label}: v4 advanced != v3`);
    }
  });
  console.log(`observation G3 v4: plies=${seen} mismatches=${mismatches} first=${first || '-'}`);
  assert.equal(seen, g3.plies.length);
  assert.equal(mismatches, 0, `first mismatch at ${first}`);
});

test('an invalid planeMapVersion throws: parser, encoder, fillRulePlanes, MatchGameState', () => {
  const game = new Game(boardFromFen(ADVANCED_POSITION), RepetitionRule.PLACEMENT);
  for (const bad of [2, 5, 0, '3', '4', null, 3.5, Number.NaN]) {
    assert.throws(() => parsePlaneMapVersion(bad), /planeMapVersion must be 3 or 4/, `parse ${String(bad)}`);
    assert.throws(
      () => observation(game, { planeMapVersion: bad as PlaneMapVersion }),
      /planeMapVersion must be 3 or 4/,
      `observation ${String(bad)}`,
    );
    assert.throws(
      () => fillRulePlanes(new Float32Array(OBS_PLANES * NUM_SQUARES), MODE_ADVANCED, bad as PlaneMapVersion),
      /planeMapVersion must be 3 or 4/,
    );
    assert.throws(() => new MatchGameState(game, { planeMapVersion: bad as PlaneMapVersion }), /planeMapVersion/);
  }
  assert.equal(parsePlaneMapVersion(undefined), 3);
  assert.equal(parsePlaneMapVersion(3), 3);
  assert.equal(parsePlaneMapVersion(4), 4);
});

test('WorkerEngine: an invalid planeMapVersion is a request error (handle and handleFast)', async () => {
  const bad = 5 as unknown as PlaneMapVersion;
  const sync = new WorkerEngine().handle({ id: 1, fen: ADVANCED_POSITION, sims: 4, planeMapVersion: bad });
  assert.ok('err' in sync, 'handle must reject an invalid planeMapVersion');
  assert.match(sync.err, /planeMapVersion must be 3 or 4/);
  const engine = new WorkerEngine();
  const fast = await engine.handleFast(
    { id: 2, fen: ADVANCED_POSITION, history: [], sims: 4, planeMapVersion: bad },
    engine.defaultBatchEvaluator(),
  );
  assert.ok('err' in fast, 'handleFast must reject an invalid planeMapVersion');
  assert.match(fast.err, /planeMapVersion must be 3 or 4/);
});

test('WorkerEngine: the request planeMapVersion reaches every leaf evaluation', () => {
  for (const version of [undefined, 3, 4] as const) {
    const seen = new Set<unknown>();
    const engine = new WorkerEngine({
      leafEvaluator: (state) => {
        seen.add(state.planeMapVersion);
        const obs = observation(state.game, { planeMapVersion: state.planeMapVersion });
        assert.equal(obs[121 * NUM_SQUARES], version === 4 ? 1 : 0, 'beginner plane 121');
        return { policyLogits: new Float32Array(ACTION_SIZE), wdl: new Float64Array([0.3, 0.4, 0.3]), movesLeft: 0 };
      },
    });
    const request: WorkerRequest = { id: 1, fen: beginnerStart(), sims: 8 };
    if (version !== undefined) request.planeMapVersion = version;
    const reply = engine.handle(request);
    assert.ok(!('err' in reply), `handle failed: ${'err' in reply ? reply.err : ''}`);
    assert.deepEqual([...seen], [version ?? 3]);
  }
});

function beginnerStart(): string {
  const g3 = loadG3();
  return g3.modes.find((m) => m.mode === 'beginner')!.start_fen;
}

test('ponderKey differs when only planeMapVersion differs; absent equals 3', () => {
  const common: PonderKeyInput = {
    history: ['m1'],
    fen: ADVANCED_POSITION,
    sims: 16,
    batch: 1,
    configIdentity: 'id',
    backend: null,
    modelSha256: null,
    maxMs: null,
    mainTableEpoch: 0,
  };
  const absent = ponderKey(common);
  const v3 = ponderKey({ ...common, planeMapVersion: 3 });
  const v4 = ponderKey({ ...common, planeMapVersion: 4 });
  assert.equal(absent, v3);
  assert.notEqual(v3, v4);
});

test('MatchGameState carries planeMapVersion through clone', () => {
  const game = new Game(boardFromFen(ADVANCED_POSITION), RepetitionRule.PLACEMENT);
  assert.equal(new MatchGameState(game).planeMapVersion, 3);
  const s4 = new MatchGameState(game, { planeMapVersion: 4 });
  assert.equal(s4.clone().planeMapVersion, 4);
});

// ---- KOM-1162 review finding #1: ponder-path version tests -----------------
// `runPonderSession` (worker.ts ~:972-1063) builds its scratch root state
// from `ctx.planeMapVersion` (~:995) and stores the cache entry keyed, and
// tagged, under that same version (~:1031-1053, incl. ~:1036). Dropping
// either assignment falls back to the default (3): a v4 ponder would
// evaluate (or key/store) as v3, so a v4 request would either be served
// v3-encoded pondered results as a silent "hit", or (if only the stored key
// drops it) never hit at all. Recording every `planeMapVersion` the batch
// evaluator sees -- across both the live search and the ponder session it
// starts -- makes a drop at either site visible: `seen` must land on
// exactly {4}, never {3} or {3, 4}.

function ponderHistory(n: number): string[] {
  const history: string[] = [];
  for (let i = 0; i < n; i++) history.push(sanRecords(gameFromHistory(history).board)[0].san);
  return history;
}
const PONDER_BASE = ponderHistory(2);

function recordingBatchEvaluator(seen: Set<unknown>): MlhBatchEvaluator {
  const leaf = makeFakeMlhEvaluator(ACTION_SIZE);
  return (states) =>
    states.map((state) => {
      seen.add(state.planeMapVersion);
      return leaf(state);
    });
}

test('ponder: a v4 request -> v4 ponder hit, every evaluation (incl. pondering) is at 4', async () => {
  const seen = new Set<unknown>();
  const ev = recordingBatchEvaluator(seen);
  const engine = new WorkerEngine();

  const r1 = await engine.handleFast(
    { id: 1, fen: boardToFen(gameFromHistory(PONDER_BASE).board), history: PONDER_BASE, sims: 16, batch: 4, planeMapVersion: 4 },
    ev,
  );
  assert.ok(!('err' in r1), JSON.stringify(r1));
  if ('err' in r1) return;
  await engine.ponderingPromise();
  assert.ok(engine.ponderCacheSize > 0, 'expected at least one cached predicted reply');

  const root = engine.tree?.root;
  assert.ok(root != null, 'expected a retained tree with a root after r1');
  if (root == null) return;
  const [predicted] = predictReplies(root, r1.action, 1);
  assert.ok(predicted !== undefined, 'expected a predicted reply under the chosen child');
  const afterEngineMove = gameFromHistory([...PONDER_BASE, r1.san]).board;
  const replySan = buildSan(afterEngineMove, decodeAction(afterEngineMove, predicted).struct);
  const history2 = [...PONDER_BASE, r1.san, replySan];

  // Same sims/batch/planeMapVersion as r1: everything `ponderKey` compares
  // on the request side matches what the ponder session stored, so this
  // must be a hit (r2.stats.pondered === true), not a fresh search.
  const r2 = await engine.handleFast(
    { id: 2, fen: boardToFen(gameFromHistory(history2).board), history: history2, sims: 16, batch: 4, planeMapVersion: 4 },
    ev,
  );
  assert.ok(!('err' in r2), JSON.stringify(r2));
  if ('err' in r2) return;
  assert.equal(r2.stats.pondered, true, 'expected a ponder hit on r2');
  await engine.ponderingPromise();

  // Every batch-evaluator call across r1's own search and the ponder
  // session it started saw 4 -- never the default 3 a dropped
  // `planeMapVersion` would fall back to.
  assert.deepEqual([...seen], [4]);
});

test('ponder: a v4 ponder, then a request with NO planeMapVersion (defaults to 3), is a MISS', async () => {
  const engine = new WorkerEngine();
  const seen = new Set<unknown>();
  const ev = recordingBatchEvaluator(seen);

  const r1 = await engine.handleFast(
    { id: 1, fen: boardToFen(gameFromHistory(PONDER_BASE).board), history: PONDER_BASE, sims: 16, batch: 4, planeMapVersion: 4 },
    ev,
  );
  assert.ok(!('err' in r1), JSON.stringify(r1));
  if ('err' in r1) return;
  await engine.ponderingPromise();
  assert.ok(engine.ponderCacheSize > 0, 'expected at least one cached predicted reply');

  const root = engine.tree?.root;
  assert.ok(root != null, 'expected a retained tree with a root after r1');
  if (root == null) return;
  const [predicted] = predictReplies(root, r1.action, 1);
  assert.ok(predicted !== undefined, 'expected a predicted reply under the chosen child');
  const afterEngineMove = gameFromHistory([...PONDER_BASE, r1.san]).board;
  const replySan = buildSan(afterEngineMove, decodeAction(afterEngineMove, predicted).struct);
  const history2 = [...PONDER_BASE, r1.san, replySan];

  // Same history/fen/sims/batch as the pondered entry, but `planeMapVersion`
  // is ABSENT here (`ponderKey`'s own `?? DEFAULT_PLANE_MAP_VERSION` makes
  // that key component 3) -- it no longer matches the v4 entry's key, so
  // this must miss the cache and run a fresh (v3) search rather than being
  // served the v4-pondered result.
  const r2 = await engine.handleFast(
    { id: 2, fen: boardToFen(gameFromHistory(history2).board), history: history2, sims: 16, batch: 4 },
    ev,
  );
  assert.ok(!('err' in r2), JSON.stringify(r2));
  if ('err' in r2) return;
  assert.notEqual(r2.stats.pondered, true, 'a planeMapVersion mismatch must miss the ponder cache');
});

// KOM-1162 review finding #1 (worker.ts ~:1110): on a ponder hit,
// `lookupPonder` rebuilds a `MatchGameState` for `PreparedSearch.state` from
// the cached `entry.context.planeMapVersion`. Verified locally (by deleting
// that one assignment and re-running this file, rules_profile.test.ts, and a
// hand-built "hit, then a larger-sims follow-up request" probe) that nothing
// currently reads `prepared.state.planeMapVersion` on the hit path --
// `finish()` only reads `.tree`/`.rootBoard`/`.sims`/`.rules`/etc off
// `PreparedSearch`, never `.state`, and the ponder session kicked off right
// after (`maybeStartPonder`'s `ctx.planeMapVersion`) is sourced straight
// from `entry.context.planeMapVersion`, not from this reconstructed state.
// So this is currently NOT reachable by any behavioral test through the
// public `WorkerEngine` API (confirmed red-less under the deletion above);
// this is a source-level guard instead, standing in for the "no observer
// exists yet" gap -- it fails the moment the assignment itself is removed,
// which is the one thing anchoring this state to the version it claims to
// carry should a future caller ever start reading it.
test('ponder hit path: the reconstructed root state is built with the cached planeMapVersion (worker.ts ~:1110)', () => {
  const src = readFileSync(new URL('../src/engine.ts', import.meta.url), 'utf8');
  const stateBlock = src.match(/state: new MatchGameState\(entry\.context\.game, \{[^}]*\}\)/);
  assert.ok(stateBlock, 'expected to find the ponder-hit MatchGameState construction in lookupPonder');
  assert.match(
    stateBlock![0],
    /planeMapVersion:\s*entry\.context\.planeMapVersion/,
    'the hit-path root state must be built with entry.context.planeMapVersion',
  );
});

// ---- KOM-1162 review finding #2: bias-table reset on planeMapVersion change
// (worker.ts ~:620-630 `resetBiasOnRulesChange`). Copies the rules-marker
// pattern from rules_profile.test.ts (~line 198/231), swapping the varying
// field for `planeMapVersion` -- a `rules` change already clears the bias
// table there; this is the SAME guard's other trigger, untested until now.

test('a planeMapVersion change resets the bias table between requests', () => {
  const engine = new WorkerEngine();
  const reply1 = engine.handle({ id: 1, fen: ADVANCED_POSITION, sims: 32, planeMapVersion: 3 });
  assert.ok(!('err' in reply1), JSON.stringify(reply1));
  const tree = engine.tree;
  assert.ok(tree !== null);
  if (tree === null) return;

  // A synthetic marker entry, independent of anything the search itself
  // would ever write under this key -- proves the NEXT request's bias
  // table is cleared because `planeMapVersion` changed, not merely that the
  // search happened to decay/overwrite it.
  const MARKER_KEY = 0x7ffffffd;
  tree.bias.entries.set(MARKER_KEY, { errSum: new Float64Array([1, 2, 3]), weightSum: 1, gen: 0 });
  const sizeBeforeReset = tree.bias.size;
  assert.ok(tree.bias.entries.has(MARKER_KEY));

  // sims: 0 -- only the root is evaluated; a fresh node construction always
  // reads (and so lazily creates) a zero-weight bias entry for its own key
  // as a harmless side effect, so the table is not necessarily empty
  // afterwards -- but it must be strictly SMALLER than before, and the
  // marker (and everything else request 1 wrote) must be gone.
  const reply2 = engine.handle({ id: 2, fen: ADVANCED_POSITION, sims: 0, planeMapVersion: 4 });
  assert.ok(!('err' in reply2), JSON.stringify(reply2));
  assert.equal(engine.tree, tree, 'the one retained tree never gets replaced');
  assert.ok(!tree.bias.entries.has(MARKER_KEY), 'the bias table must be cleared when planeMapVersion changes');
  assert.ok(
    tree.bias.size < sizeBeforeReset,
    `expected the bias table to shrink on a planeMapVersion change: before=${sizeBeforeReset} after=${tree.bias.size}`,
  );
});

test('no planeMapVersion change: the bias table is NOT reset between two requests', () => {
  const engine = new WorkerEngine();
  const reply1 = engine.handle({ id: 1, fen: ADVANCED_POSITION, sims: 32, planeMapVersion: 4 });
  assert.ok(!('err' in reply1), JSON.stringify(reply1));
  const tree = engine.tree;
  assert.ok(tree !== null);
  if (tree === null) return;

  const MARKER_KEY = 0x7ffffffc;
  tree.bias.entries.set(MARKER_KEY, { errSum: new Float64Array([1, 2, 3]), weightSum: 1, gen: 0 });

  const reply2 = engine.handle({ id: 2, fen: ADVANCED_POSITION, sims: 0, planeMapVersion: 4 });
  assert.ok(!('err' in reply2), JSON.stringify(reply2));
  assert.ok(tree.bias.entries.has(MARKER_KEY), 'same planeMapVersion: the marker must survive');
});
