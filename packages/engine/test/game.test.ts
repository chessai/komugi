// U4b: game, repetition, undo, draft root mask against the KOM-1055 fixture
// (the recorded parity fixture, under PLACEMENT per F-A) and the rare-branch
// rules fixture (the recorded rules-case fixture) (docs/design/clientside-engine-port.md
// section 4.1 row "game / repetition / undo (U4b)").
//
// KOM_RULES-backed cases below run unconditionally via loadRulesCases()
// (committed fixture by default, docs/design/kom1196-golden-fixtures.md).

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { ACTION_SIZE, ADVANCED_POSITION, C_START } from '../src/constants.ts';
import { boardFromFen, boardToFen, canonicalFen } from '../src/fen.ts';
import { draftPlacements, draftRootMask, Game } from '../src/game.ts';
import { fillLegalMask } from '../src/movegen.ts';
import { RepetitionRule } from '../src/repetition.ts';
import { resolveSan } from '../src/san.ts';
import { loadFixture, loadRulesCases } from './fixture.ts';

function maskIds(mask: Uint8Array): number[] {
  const out: number[] = [];
  for (let i = 0; i < mask.length; i++) if (mask[i]) out.push(i);
  return out;
}

function winStateOf(game: Game): { terminal: boolean; winState: number[] | null } {
  const win = game.winState();
  const terminal = win.some((x) => x !== 0);
  return { terminal, winState: terminal ? Array.from(win) : null };
}

test('replaying a fixture record\'s history under PLACEMENT reproduces its legal/terminal/win_state/repetition', () => {
  const records = loadFixture();
  let legalMismatches = 0;
  let terminalMismatches = 0;
  let winMismatches = 0;
  let repMismatches = 0;
  let first = '';

  for (const rec of records) {
    const board = boardFromFen(ADVANCED_POSITION);
    const game = new Game(board, RepetitionRule.PLACEMENT);
    for (const san of rec.history) {
      game.play(resolveSan(game.board, san));
    }

    const legal = maskIds(game.validMoves());
    const { terminal, winState } = winStateOf(game);
    const repetition = game.repetitionCount();

    if (JSON.stringify(legal) !== JSON.stringify(rec.legal)) {
      legalMismatches++;
      if (!first) first = `${rec.id}#legal`;
    }
    if (terminal !== rec.terminal) {
      terminalMismatches++;
      if (!first) first = `${rec.id}#terminal`;
    }
    if (JSON.stringify(winState) !== JSON.stringify(rec.win_state)) {
      winMismatches++;
      if (!first) first = `${rec.id}#win_state`;
    }
    if (repetition !== rec.repetition) {
      repMismatches++;
      if (!first) first = `${rec.id}#repetition`;
    }
  }

  console.log(
    `game fixture: records=${records.length} legal=${legalMismatches} terminal=${terminalMismatches} ` +
      `win_state=${winMismatches} repetition=${repMismatches} first=${first || '-'}`,
  );
  assert.ok(records.length > 0, 'no fixture records');
  assert.equal(legalMismatches, 0, `first legal mismatch at ${first}`);
  assert.equal(terminalMismatches, 0, `first terminal mismatch at ${first}`);
  assert.equal(winMismatches, 0, `first win_state mismatch at ${first}`);
  assert.equal(repMismatches, 0, `first repetition mismatch at ${first}`);
});

// ---------------------------------------------------------------------------
// KOM_RULES (the recorded rules-case fixture): kom239 replays under both rules, undo
// byte-exactness, hand-written fen_cases and draft_root_mask_cases.
// ---------------------------------------------------------------------------

type RulesPly = {
  ply: number;
  fen: string;
  legal: number[];
  terminal: boolean;
  win_state: number[] | null;
  repetition: number;
};

type Kom239Entry = {
  name: string;
  seat: string;
  sans: string[];
  replays: Record<string, RulesPly[]>;
};

type FenCase = {
  id: string;
  fen: string;
  repetition_rule: string;
  history_occurrences: number;
  legal: number[];
  terminal: boolean;
  win_state: number[] | null;
  repetition: number;
};

type DraftMaskCase = {
  id: string;
  fen: string;
  min_placements: number;
  legal_before: number[];
  legal_after: number[];
};

type RulesCases = {
  kom239: Kom239Entry[];
  fen_cases: FenCase[];
  draft_root_mask_cases: DraftMaskCase[];
};

test('kom239 replays under both rules match the recorded rules-case fixture ply by ply, with byte-exact undo', () => {
  const cases = loadRulesCases<RulesCases>();

  let mismatches = 0;
  let undoMismatches = 0;
  let first = '';
  let plies = 0;

  for (const entry of cases.kom239) {
    for (const [ruleName, want] of Object.entries(entry.replays)) {
      const rule = ruleName as RepetitionRule;
      const board = boardFromFen(ADVANCED_POSITION);
      const game = new Game(board, rule);

      const check = (ply: RulesPly, label: string) => {
        plies++;
        const legal = maskIds(game.validMoves());
        const { terminal, winState } = winStateOf(game);
        const ok =
          JSON.stringify(legal) === JSON.stringify(ply.legal) &&
          terminal === ply.terminal &&
          JSON.stringify(winState) === JSON.stringify(ply.win_state) &&
          game.repetitionCount() === ply.repetition;
        if (!ok) {
          mismatches++;
          if (!first) first = label;
        }
      };

      check(want[0], `${entry.name}#${ruleName}#ply0`);

      for (let i = 0; i < entry.sans.length; i++) {
        const id = resolveSan(game.board, entry.sans[i]);

        const beforeFen = canonicalFen(boardToFen(game.board));
        const beforeRep = game.repetitionCount();
        const beforeTurns = game.turns;
        const beforeProgress = game.progressPlies;
        const beforeLastMove = JSON.stringify(game.lastMove);

        game.play(id);
        check(want[i + 1], `${entry.name}#${ruleName}#ply${i + 1}`);
        game.undo();

        const restored =
          canonicalFen(boardToFen(game.board)) === beforeFen &&
          game.repetitionCount() === beforeRep &&
          game.turns === beforeTurns &&
          game.progressPlies === beforeProgress &&
          JSON.stringify(game.lastMove) === beforeLastMove;
        if (!restored) {
          undoMismatches++;
          if (!first) first = `${entry.name}#${ruleName}#undo${i + 1}`;
        }

        game.play(id); // redo, to continue the replay
      }
    }
  }

  console.log(
    `kom239: plies=${plies} mismatches=${mismatches} undoMismatches=${undoMismatches} first=${first || '-'}`,
  );
  assert.ok(plies > 0, 'no kom239 plies replayed');
  assert.equal(mismatches, 0, `first ply mismatch at ${first}`);
  assert.equal(undoMismatches, 0, `first undo mismatch at ${first}`);
});

test('hand-written fen_cases: checkmate-plus-fourfold precedence and insufficient material', () => {
  const cases = loadRulesCases<RulesCases>();

  for (const fc of cases.fen_cases) {
    const board = boardFromFen(fc.fen);
    const game = new Game(board, fc.repetition_rule as RepetitionRule, fc.history_occurrences);

    assert.equal(game.repetitionCount(), fc.repetition, fc.id);
    const legal = maskIds(game.validMoves());
    assert.deepEqual(legal, fc.legal, `${fc.id}: legal ids`);
    const { terminal, winState } = winStateOf(game);
    assert.equal(terminal, fc.terminal, `${fc.id}: terminal`);
    assert.deepEqual(winState, fc.win_state, `${fc.id}: win_state`);
  }
});

test('draft_root_mask both fallbacks and the ordinary masking case', () => {
  const cases = loadRulesCases<RulesCases>();

  for (const dc of cases.draft_root_mask_cases) {
    const board = boardFromFen(dc.fen);
    const mask = new Uint8Array(ACTION_SIZE);
    fillLegalMask(board, mask);
    assert.deepEqual(maskIds(mask), dc.legal_before, `${dc.id}: legal_before`);

    const masked = draftRootMask(board, mask, dc.min_placements);
    assert.deepEqual(maskIds(masked), dc.legal_after, `${dc.id}: legal_after`);

    if (dc.id === 'draft_mask_normal') {
      // The ordinary case must actually mask: every withheld id is in the
      // C-block, and draftRootMask must not have mutated its input.
      assert.ok(dc.legal_after.every((id) => id < C_START));
      assert.deepEqual(maskIds(mask), dc.legal_before, 'draftRootMask mutated its input');
    } else {
      // Both fallbacks are a no-op.
      assert.deepEqual(dc.legal_after, dc.legal_before, `${dc.id}: fallback must not mask`);
    }
  }
});

test('draftPlacements counts board pieces of the drafting side', () => {
  const cases = loadRulesCases<RulesCases>();

  const normal = cases.draft_root_mask_cases.find((c) => c.id === 'draft_mask_normal');
  assert.ok(normal, 'draft_mask_normal case missing');
  const board = boardFromFen(normal.fen);
  assert.equal(draftPlacements(board, board.turn), 0);
  assert.throws(() => draftPlacements(board, 1 - board.turn));
});
