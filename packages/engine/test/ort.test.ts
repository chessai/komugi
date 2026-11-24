// U8: the ORT evaluator seam against the Python ONNX exporter's fixture (design doc
// section 2.7, section 6 U8 acceptance, S9). A SEPARATE file from
// worker.test.ts/root.test.ts on purpose (S9): it must FAIL -- not skip --
// when KOM_MODEL is unset, so an env-skip can never hide this unit's own
// checks the way a quiet `t.skip` would. KOM-1196: that holds under
// KOM_FULL_REGRESSION=1; without the flag the case skips with that reason.

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';

import { ADVANCED_POSITION } from '../src/constants.ts';
import { canonicalFen } from '../src/fen.ts';
import { boardToFen } from '../src/fen.ts';
import { observation } from '../src/observation.ts';
import { gameFromHistory } from '../src/root.ts';
import { runOrt } from '../src/ort_core.ts';
import { isOrtModule } from '../src/worker_host.ts';
import { loadOrtSession } from './support/ort_node.ts';
import { FULL_REGRESSION, assertNoStrayKomVars } from './fixture.ts';
import { parsePlaneMapVersion } from '../src/rule_features.ts';
import type { PlaneMapVersion } from '../src/rule_features.ts';

const ATOL = 1e-4; // the Python ONNX exporter's / the ORT benchmark's own fixed parity tolerance (KOM-1054/KOM-1056).

function requiredEnv(name: string): string {
  const value = process.env[name];
  if (!value) {
    // Fail loudly: this is the one test file the model/fixture are required
    // for, and an env-skip here would hide every ORT check silently (S9).
    throw new Error(`${name} is not set: ort.test.ts must fail, not skip, without it`);
  }
  return value;
}

type FixturePosition = {
  fen: string;
  history: string[];
  // KOM-1162: a per-mode fixture's start FEN; absent = the advanced start.
  start?: string;
  obs: string;
  policy: string;
  wdl: string;
  moves_left: string;
};

type Fixture = {
  shapes: { obs: number[]; policy: number[]; wdl: number[]; moves_left: number[] };
  positions: FixturePosition[];
  // KOM-1162: the encoding the fixture's `obs` was written under (the
  // model sidecar's `plane_map_version`); absent = 3.
  plane_map_version?: PlaneMapVersion;
};

// Inverse of the Python ONNX exporter's encode_array: base64 of float32 little-endian
// C-order bytes, decoded to a flat Float32Array of the given shape's size.
function decodeFloat32(base64: string, shape: readonly number[]): Float32Array {
  const buf = Buffer.from(base64, 'base64');
  const data = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
  const expected = shape.reduce((a, b) => a * b, 1);
  assert.equal(data.length, expected, `decodeFloat32: expected ${expected} floats, got ${data.length}`);
  return data;
}

function maxAbsDelta(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let max = 0;
  for (let i = 0; i < a.length; i++) {
    const delta = Math.abs(a[i] - b[i]);
    if (delta > max) max = delta;
  }
  return max;
}

test('ORT evaluator: TS observation matches the fixture bit for bit, ORT outputs agree within 1e-4', async (t) => {
  // KOM-1196 (golden-fixtures design section 3.4): no committed model, so
  // this runs only under the full-regression flag; a stray KOM_* without the
  // flag still fails rather than skips (N3).
  if (!FULL_REGRESSION) {
    assertNoStrayKomVars();
    t.skip('requires KOM_FULL_REGRESSION=1');
    return;
  }
  const modelPath = requiredEnv('KOM_MODEL');
  const fixturePath = requiredEnv('KOM_ONNX_FIXTURE');
  const ortwebDir = process.env.ORTWEB_DIR; // evaluator.ts falls back to the vendored default itself.

  const fixture = JSON.parse(readFileSync(fixturePath, 'utf8')) as Fixture;
  assert.ok(fixture.positions.length > 0, `${fixturePath} holds no positions`);

  const { ort, session } = await loadOrtSession(modelPath, { ortwebDir, threads: 1 });
  // KOM-1225: the real vendored onnxruntime-web module (loaded above, no
  // fake) must itself satisfy worker_host.ts's isOrtModule shape check --
  // the gap that let defaultLoadOrt reject every real-ORT init.
  assert.equal(isOrtModule(ort), true, 'isOrtModule must accept the real vendored onnxruntime-web module');

  let worstPolicyDelta = 0;
  let worstWdlDelta = 0;
  let worstMovesLeftDelta = 0;

  for (const pos of fixture.positions) {
    // The Python match server's own history path (its game_from_history,
    // root.ts's gameFromHistory): replays `history` from the advanced start
    // under PLACEMENT, exactly as the Python ONNX exporter's encode_position does.
    const game = gameFromHistory(pos.history, pos.start ?? ADVANCED_POSITION);
    assert.equal(
      canonicalFen(boardToFen(game.board)),
      canonicalFen(pos.fen),
      `${pos.fen}: TS replay of the fixture's own history landed elsewhere`,
    );

    const tsObs = observation(game, { planeMapVersion: parsePlaneMapVersion(fixture.plane_map_version) });
    const wantObs = decodeFloat32(pos.obs, fixture.shapes.obs);
    assert.equal(tsObs.length, wantObs.length, `${pos.fen}: observation length`);
    for (let i = 0; i < tsObs.length; i++) {
      assert.ok(
        Object.is(tsObs[i], wantObs[i]),
        `${pos.fen}: obs[${i}] = ${tsObs[i]}, fixture has ${wantObs[i]} (bit-exact required)`,
      );
    }

    const raw = await runOrt(session, ort, tsObs, fixture.shapes.obs);
    const wantPolicy = decodeFloat32(pos.policy, fixture.shapes.policy);
    const wantWdl = decodeFloat32(pos.wdl, fixture.shapes.wdl);
    const wantMovesLeft = decodeFloat32(pos.moves_left, fixture.shapes.moves_left);

    worstPolicyDelta = Math.max(worstPolicyDelta, maxAbsDelta(raw.policy, wantPolicy));
    worstWdlDelta = Math.max(worstWdlDelta, maxAbsDelta(raw.wdl, wantWdl));
    worstMovesLeftDelta = Math.max(worstMovesLeftDelta, Math.abs(raw.movesLeft - wantMovesLeft[0]));
  }

  console.log(
    `ort: positions=${fixture.positions.length} max|delta| policy=${worstPolicyDelta} ` +
      `wdl=${worstWdlDelta} moves_left=${worstMovesLeftDelta}`,
  );
  assert.ok(worstPolicyDelta <= ATOL, `policy delta ${worstPolicyDelta} exceeds ${ATOL}`);
  assert.ok(worstWdlDelta <= ATOL, `wdl delta ${worstWdlDelta} exceeds ${ATOL}`);
  assert.ok(worstMovesLeftDelta <= ATOL, `moves_left delta ${worstMovesLeftDelta} exceeds ${ATOL}`);
});
