// U4a: attack and terminal predicates against the KOM-1055 fixture and
// the recorded move-record fixture (docs/design/clientside-engine-port.md section 4.1
// row "terminal predicates (U4a)").

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { cellIndex, typeIndex } from '../src/board.ts';
import { BLACK, MARSHAL, WHITE } from '../src/constants.ts';
import { boardFromFen } from '../src/fen.ts';
import {
  inCheck,
  isCheckmate,
  isGameOver,
  isInsufficientMaterial,
  isRepetitionDraw,
  isStalemate,
} from '../src/terminal.ts';
import { loadFixture, loadMoveRecords } from './fixture.ts';

test('inCheck(white)/inCheck(black) equal the exported in_check_w/in_check_b on every move-record position', () => {
  const positions = loadMoveRecords();
  let mismatchesW = 0;
  let mismatchesB = 0;
  let first = '';

  for (const pos of positions) {
    const board = boardFromFen(pos.fen);
    const gotW = inCheck(board, WHITE);
    const gotB = inCheck(board, BLACK);
    if (gotW !== pos.in_check_w) {
      mismatchesW++;
      if (!first) first = `${pos.id}#w`;
    }
    if (gotB !== pos.in_check_b) {
      mismatchesB++;
      if (!first) first = `${pos.id}#b`;
    }
  }

  console.log(
    `inCheck: positions=${positions.length} mismatchesW=${mismatchesW} mismatchesB=${mismatchesB} first=${first || '-'}`,
  );
  assert.ok(positions.length > 0, 'no move-record positions');
  assert.equal(mismatchesW, 0, `first in_check_w mismatch at ${first}`);
  assert.equal(mismatchesB, 0, `first in_check_b mismatch at ${first}`);
});

// Counts marshals directly off the board, independent of terminal.ts, so this
// test does not lean on the function it is partly checking (isGameOver's own
// internal marshal count stays private).
function marshalCount(board: ReturnType<typeof boardFromFen>): number {
  let count = 0;
  for (let r = 0; r < 9; r++) {
    for (let f = 0; f < 9; f++) {
      for (let t = 0; t < 3; t++) {
        const code = board.cells[cellIndex(r, f, t)];
        if (code !== 0 && typeIndex(code) === MARSHAL) count++;
      }
    }
  }
  return count;
}

test('terminal flag and win_state are exact on the KOM-1055 fixture (is_checkmate/is_stalemate precedence)', () => {
  const records = loadFixture();
  let terminalMismatches = 0;
  let winStateMismatches = 0;
  let first = '';
  let checked = 0;

  for (const rec of records) {
    const board = boardFromFen(rec.fen);
    // Out of scope for U4a: the ply cap (turns >= 512) is a game.py/U4b
    // concern layered above this predicate (section 2.3); none of the
    // KOM-1055 fixture's sampled games reach it.
    const gameOver = isGameOver(board, rec.repetition);
    if (gameOver !== rec.terminal) {
      terminalMismatches++;
      if (!first) first = rec.id;
      continue;
    }
    if (!rec.terminal) continue;
    checked++;

    // Replay game.py's win_state precedence (checkmate > stalemate >
    // fourfold > insufficient > marshal captured) using only the exported
    // predicates, to pin is_checkmate/is_stalemate exactly against the
    // fixture's win_state vector.
    const want = rec.win_state ?? [];
    let got: number[];
    if (isCheckmate(board)) {
      const winner = 1 - board.turn;
      got = winner === WHITE ? [1, 0, 0] : [0, 1, 0];
    } else if (isStalemate(board)) {
      got = [0, 0, 1];
    } else if (isRepetitionDraw(rec.repetition)) {
      got = [0, 0, 1];
    } else if (isInsufficientMaterial(board)) {
      got = [0, 0, 1];
    } else {
      assert.notEqual(marshalCount(board), 2, `${rec.id}: terminal with neither mate/draw/insufficient nor a captured marshal`);
      const winner = 1 - board.turn;
      got = winner === WHITE ? [1, 0, 0] : [0, 1, 0];
    }

    if (JSON.stringify(got) !== JSON.stringify(want)) {
      winStateMismatches++;
      if (!first) first = rec.id;
    }
  }

  console.log(
    `terminal: records=${records.length} terminalMismatches=${terminalMismatches} winStateChecked=${checked} winStateMismatches=${winStateMismatches} first=${first || '-'}`,
  );
  assert.ok(records.length > 0, 'no fixture records');
  assert.equal(terminalMismatches, 0, `first terminal mismatch at ${first}`);
  assert.equal(winStateMismatches, 0, `first win_state mismatch at ${first}`);
});
