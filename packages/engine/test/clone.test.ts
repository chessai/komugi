// KOM-1079: regression for the clone bug the KOM-1077 verifier caught
// (tickets/KOM-1077.md Findings). The KOM-1077 speed-up replaced Game's
// private `keyCounts: Map<string, number>` with `counts: RepetitionCounts`
// (+ keyHi/keyLo), but search/state.ts's clone still did
// `target.keyCounts = new Map(source.keyCounts)` -- a write to a field that
// no longer exists -- so every cloned MatchGameState silently started with
// NO repetition history at all: its first `repetitionCount()` reads 1
// (Game's own constructor re-seeds the current key), not whatever count the
// source had actually accumulated.
//
// Replays the kom239 repetition fixtures (the recorded rules-case fixture)
// under both PLACEMENT and FULL (docs/design/clientside-engine-port.md
// section 2.3). kom239_b0_g4/g6 specifically drive a 4-count draw under
// PLACEMENT (the rules-case fixture generator's own docstring), which is
// exactly the "one move
// from a 4-count repetition" case the ticket asks for: a clone taken right
// before that move must see the same draw a search would see on the
// original, via the same MatchState primitives (`isTerminal`/`terminalWdl`)
// the match search actually calls.

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { ADVANCED_POSITION } from '../src/constants.ts';
import { boardFromFen } from '../src/fen.ts';
import { Game } from '../src/game.ts';
import { keyFor, RepetitionRule } from '../src/repetition.ts';
import { resolveSan } from '../src/san.ts';
import { MatchGameState } from '../src/search/state.ts';
import { loadRulesCases } from './fixture.ts';

type RulesPly = { repetition: number };

type Kom239Entry = {
  name: string;
  sans: string[];
  replays: Record<string, RulesPly[]>;
};

type RulesCases = { kom239: Kom239Entry[] };

function bytesEqual(a: Uint8Array, b: Uint8Array): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

function wdlEqual(a: Float64Array, b: Float64Array): boolean {
  return a.length === b.length && a.every((v, i) => v === b[i]);
}

test('cloning a MatchGameState carries repetition history across kom239, both rules', () => {
  const cases = loadRulesCases<RulesCases>();

  let checkedPlies = 0;
  let repMismatches = 0;
  let keyMismatches = 0;
  let terminalMismatches = 0;
  let wdlMismatches = 0;
  let fourfoldCasesSeen = 0;
  let first = '';

  for (const entry of cases.kom239) {
    for (const ruleName of Object.keys(entry.replays)) {
      const rule = ruleName as RepetitionRule;
      const wantReps = entry.replays[ruleName];
      const board = boardFromFen(ADVANCED_POSITION);
      const game = new Game(board, rule);
      const state = new MatchGameState(game);

      for (let i = 0; i < entry.sans.length; i++) {
        const label = `${entry.name}#${ruleName}#ply${i}->${i + 1}`;
        const id = resolveSan(state.game.board, entry.sans[i]);

        // Clone right before playing move i (state as of ply i): the clone
        // must start with the SAME repetition history as the source, not a
        // reset one.
        const clone = state.clone();
        checkedPlies++;

        const repOk = clone.game.repetitionCount() === state.game.repetitionCount();
        if (!repOk) {
          repMismatches++;
          if (!first) first = `${label}#repetitionCount`;
        }
        assert.equal(
          state.game.repetitionCount(),
          wantReps[i].repetition,
          `${label}: fixture repetition at ply ${i}`,
        );

        const keyOk = bytesEqual(keyFor(clone.game.board, rule), keyFor(state.game.board, rule));
        if (!keyOk) {
          keyMismatches++;
          if (!first) first = `${label}#key`;
        }

        // Play the SAME next move on both. A clone with intact history and
        // a clone with reset history can reach different win_state here --
        // this is the actual search-visible consequence of the bug.
        clone.play(id);
        state.play(id);

        const cloneTerminal = clone.isTerminal();
        const stateTerminal = state.isTerminal();
        if (cloneTerminal !== stateTerminal) {
          terminalMismatches++;
          if (!first) first = `${label}#isTerminal`;
        } else if (cloneTerminal && !wdlEqual(clone.terminalWdl(), state.terminalWdl())) {
          wdlMismatches++;
          if (!first) first = `${label}#terminalWdl`;
        }

        if (wantReps[i + 1]?.repetition >= 4) fourfoldCasesSeen++;
      }
    }
  }

  console.log(
    `clone: plies=${checkedPlies} rep=${repMismatches} key=${keyMismatches} ` +
      `terminal=${terminalMismatches} wdl=${wdlMismatches} fourfoldCases=${fourfoldCasesSeen} first=${first || '-'}`,
  );
  assert.ok(checkedPlies > 0, 'no plies checked');
  assert.ok(fourfoldCasesSeen > 0, 'no fourfold (repetition >= 4) ply reached by the kom239 fixtures');
  assert.equal(repMismatches, 0, `first repetitionCount mismatch at ${first}`);
  assert.equal(keyMismatches, 0, `first key-bytes mismatch at ${first}`);
  assert.equal(terminalMismatches, 0, `first isTerminal mismatch at ${first}`);
  assert.equal(wdlMismatches, 0, `first terminalWdl mismatch at ${first}`);
});
