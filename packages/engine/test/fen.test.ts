// U1: FEN parse/serialize against the KOM-1055 fixture, and canonicalFen against
// outputs recorded from Python (the match server's canonical_fen, cyboard board_to_fen∘board_from_fen).

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { Board } from '../src/board.ts';
import { ADVANCED_POSITION, NUM_CELLS } from '../src/constants.ts';
import { boardFromFen, boardToFen, canonicalFen, validateFen } from '../src/fen.ts';
import { loadFixture } from './fixture.ts';

test('boardToFen(boardFromFen(fen)) === fen and canonicalFen(fen) === fen on every fixture record', () => {
  const records = loadFixture();
  let boardMiss = 0;
  let canonMiss = 0;
  let firstMiss = '';
  for (const rec of records) {
    let ok = true;
    if (boardToFen(boardFromFen(rec.fen)) !== rec.fen) {
      boardMiss++;
      ok = false;
    }
    if (canonicalFen(rec.fen) !== rec.fen) {
      canonMiss++;
      ok = false;
    }
    if (!ok && !firstMiss) firstMiss = rec.id;
  }
  console.log(`fen: records=${records.length} boardToFen_mismatch=${boardMiss} canonicalFen_mismatch=${canonMiss} first=${firstMiss || '-'}`);
  assert.ok(records.length > 0, 'empty fixture');
  assert.equal(boardMiss, 0, `first mismatch ${firstMiss}`);
  assert.equal(canonMiss, 0, `first mismatch ${firstMiss}`);
  assert.ok(records.some((r) => r.phase === 'draft') && records.some((r) => r.phase === 'play'));
  assert.ok(records.some((r) => r.fen.includes('|')), 'no tower in the fixture');
});

// [input, canonical_fen(input), board_to_fen(board_from_fen(input))], recorded from Python.
const E = '9/9/9/9/9/9/9/9/9';
const SPELLINGS: [string, string, string][] = [
  [`${E} D4M1/t1m1 w 3 wb 1`, `${E} D4M1/t1m1 w 3 wb 1`, `${E} D4M1/t1m1 w 3 wb 1`],
  ['54/9/9/9/9/9/9/9/9 M1/m1 w 3 wb +01', `${E} M1/m1 w 3 wb 1`, `${E} M1/m1 w 3 wb 1`],
  ['9/9/9/9/9/9/9/9/|M|8 M1/m1 w 3 b 1', '9/9/9/9/9/9/9/9/M8 M1/m1 w 3 b 1', '9/9/9/9/9/9/9/9/M8 M1/m1 w 3 b 1'],
  [`${E} M1/m1 w 3  1`, `${E} M1/m1 w 3 - 1`, `${E} M1/m1 w 3 - 1`],
  [`${E} M1/m1 w 3 wb 1_0`, `${E} M1/m1 w 3 wb 10`, `${E} M1/m1 w 3 wb 10`],
  ['9/9/9/9/9/9/9/9/|M:d:G|8 -/- b 0 - 7', '9/9/9/9/9/9/9/9/|M:d:G|8 -/- b 0 - 7', '9/9/9/9/9/9/9/9/|M:d:G|8 -/- b 0 - 7'],
  [`${E} M1M2/m0 w 3 wb 1`, `${E} M1M2/m0 w 3 wb 1`, `${E} M2M2/- w 3 wb 1`],
  [ADVANCED_POSITION, ADVANCED_POSITION, ADVANCED_POSITION],
];

test('canonicalFen keeps hand order and canonicalises non-canonical spellings (Python outputs)', () => {
  for (const [input, canon, viaBoard] of SPELLINGS) {
    assert.equal(canonicalFen(input), canon, input);
    assert.equal(boardToFen(boardFromFen(input)), viaBoard, input);
  }
});

const MALFORMED: [string, string][] = [
  [`${E} M1/m1 w 3 wb 1 x`, 'expected 6 fields, received 7'],
  [`${E} M1/m1 w 3 wb`, 'expected 6 fields, received 5'],
  [`${E}  M1/m1 w 3 wb 1`, 'expected 6 fields, received 7'],
  [`${E} M1/m1 w 3 bw 1`, "5th field (drafting availability) is invalid [expected '-', 'w', 'b', or 'wb', received bw]"],
  [`${E} M1/m1 w 4 wb 1`, "4th field (setup mode) is invalid [expected '0', '1', '2', or '3', received 4]"],
  [`${E} M/m1 w 3 wb 1`, '2nd field (hand pieces) is invalid [invalid piece]'],
  ['9/9/9/9/9/9/9/9/8 -/- w 3 - 1', '1st field (piece positions) is invalid [expected 9 squares, received 8] in rank: 9'],
  ['9/9/9/9/9/9/9/9/|M:d:G:i|6 -/- w 3 - 1', '1st field (piece positions) is invalid [tower size] @(9-1)'],
  ['9/9/9/9/9/9/9/9/9/9 -/- w 3 - 1', '1st field (piece positions) is invalid [expected 9 ranks, received 10]'],
  [`${E} -/- x 3 - 1`, "3rd field (active player) is invalid [expected 'w' or 'b', received x]"],
  [`${E} -/- w 3 - a`, '6th field (full move number) is invalid [expected an integer, received a]'],
];

test('canonicalFen and boardFromFen reject malformed FENs with Python\'s messages', () => {
  for (const [input, error] of MALFORMED) {
    assert.deepEqual(validateFen(input), { ok: false, error }, input);
    assert.throws(() => canonicalFen(input), { message: `Invalid FEN: ${error}` }, input);
    assert.throws(() => boardFromFen(input), input);
  }
});

test('board state: handOrder, copy independence, hand-built fallback order', () => {
  const b = boardFromFen(`${E} D4M1/t1m1 w 3 wb 1`);
  assert.deepEqual(Array.from(b.handOrder.subarray(0, b.handOrderLen)), [9 * 2, 0, 13 * 2 + 1, 1]);
  const c = b.copy();
  c.hand[0] = 0;
  c.cells[0] = 1;
  c.handOrder[0] = 0;
  assert.equal(boardToFen(b), `${E} D4M1/t1m1 w 3 wb 1`);
  const t = boardFromFen('9/9/9/9/9/9/9/9/|M:d:G|8 -/- b 0 - 7');
  assert.equal(t.height(8, 8), 3);
  assert.equal(t.top(8, 8), 1 * 2 + 0 + 1);
  assert.equal(t.cells.length, NUM_CELLS);
  const h = new Board();
  h.hand[13 * 2] = 1;
  h.hand[0] = 1;
  h.hand[1] = 2;
  assert.equal(boardToFen(h), `${E} M1T1/m2 w 0 - 1`);
});
