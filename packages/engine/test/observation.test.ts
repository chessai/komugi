// U5: the 128-plane observation encoder against the KOM-1055 fixture
// (the recorded parity fixture, replayed under PLACEMENT per F-A) and the
// kom239 replays of the rare-branch rules fixture (the recorded rules-case fixture),
// which hit planes 116/117 under both repetition rules (docs/design/
// clientside-engine-port.md section 4.1 row "observation (U5)").
//
// KOM_RULES-backed cases below run unconditionally via loadRulesCases()
// (committed fixture by default, docs/design/kom1196-golden-fixtures.md).

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { ADVANCED_POSITION } from '../src/constants.ts';
import { boardFromFen } from '../src/fen.ts';
import { Game } from '../src/game.ts';
import { observation, observationFromBoard, observationSize } from '../src/observation.ts';
import { RepetitionRule } from '../src/repetition.ts';
import { resolveSan } from '../src/san.ts';
import { loadFixture, loadRulesCases, sha256Hex } from './fixture.ts';

test('observation sha256 matches the KOM-1055 fixture, replayed under PLACEMENT', () => {
  const records = loadFixture();
  let shaMismatches = 0;
  let shapeMismatches = 0;
  let first = '';

  for (const rec of records) {
    const board = boardFromFen(ADVANCED_POSITION);
    const game = new Game(board, RepetitionRule.PLACEMENT);
    for (const san of rec.history) game.play(resolveSan(game.board, san));

    const obs = observation(game);
    const got = sha256Hex(obs);
    if (got !== rec.obs_sha256) {
      shaMismatches++;
      if (!first) first = `${rec.id}#sha256`;
    }
    if (JSON.stringify(observationSize()) !== JSON.stringify(rec.obs_shape)) {
      shapeMismatches++;
      if (!first) first = `${rec.id}#shape`;
    }
  }

  console.log(
    `observation fixture: records=${records.length} sha256=${shaMismatches} shape=${shapeMismatches} first=${first || '-'}`,
  );
  assert.ok(records.length > 0, 'no fixture records');
  assert.equal(shapeMismatches, 0, `first shape mismatch at ${first}`);
  assert.equal(shaMismatches, 0, `first sha256 mismatch at ${first}`);
});

test('observationFromBoard clears a dirty reused buffer rather than accumulating', () => {
  const board = boardFromFen(ADVANCED_POSITION);
  const dirty = new Float32Array(128 * 81).fill(1);
  const out = observationFromBoard(board, 1, 0, null, 0, dirty);
  assert.equal(out, dirty, 'must write into the supplied buffer, not allocate a new one');
  // Plane 121 (reserved) must read back to 0, not the pre-fill 1 -- proof the
  // function clears `out` itself rather than trusting the caller.
  assert.equal(out[121 * 81], 0);
});

// ---------------------------------------------------------------------------
// KOM_RULES (the recorded rules-case fixture): kom239 replayed under both repetition
// rules, observation sha256 checked every ply -- the only fixture source
// that ever sets planes 116/117 (repetition >= 2) or feeds an arata last
// move through plane 118's "no origin" branch during the draft.
// ---------------------------------------------------------------------------

type RulesPly = {
  obs_sha256: string;
  obs_shape: number[];
};

type Kom239Entry = {
  name: string;
  sans: string[];
  replays: Record<string, RulesPly[]>;
};

type RulesCases = {
  kom239: Kom239Entry[];
};

test('observation sha256 matches the recorded rules-case fixture for every kom239 ply, under both rules', () => {
  const cases = loadRulesCases<RulesCases>();

  let plies = 0;
  let shaMismatches = 0;
  let shapeMismatches = 0;
  let first = '';

  for (const entry of cases.kom239) {
    for (const [ruleName, want] of Object.entries(entry.replays)) {
      const rule = ruleName as RepetitionRule;
      const board = boardFromFen(ADVANCED_POSITION);
      const game = new Game(board, rule);

      const check = (ply: RulesPly, label: string) => {
        plies++;
        const obs = observation(game);
        const got = sha256Hex(obs);
        if (got !== ply.obs_sha256) {
          shaMismatches++;
          if (!first) first = `${label}#sha256`;
        }
        if (JSON.stringify(observationSize()) !== JSON.stringify(ply.obs_shape)) {
          shapeMismatches++;
          if (!first) first = `${label}#shape`;
        }
      };

      check(want[0], `${entry.name}#${ruleName}#ply0`);
      for (let i = 0; i < entry.sans.length; i++) {
        const id = resolveSan(game.board, entry.sans[i]);
        game.play(id);
        check(want[i + 1], `${entry.name}#${ruleName}#ply${i + 1}`);
      }
    }
  }

  console.log(
    `observation kom239: plies=${plies} sha256=${shaMismatches} shape=${shapeMismatches} first=${first || '-'}`,
  );
  assert.ok(plies > 0, 'no kom239 plies replayed');
  assert.equal(shapeMismatches, 0, `first shape mismatch at ${first}`);
  assert.equal(shaMismatches, 0, `first sha256 mismatch at ${first}`);
});
