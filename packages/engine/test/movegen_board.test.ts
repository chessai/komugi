// U2a: board-move mask (A0/A1) and record emitter against the KOM-1055
// fixture and the recorded move-record fixture
// (docs/design/clientside-engine-port.md section 4.1).

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { cellIndex, typeIndex } from '../src/board.ts';
import { ACTION_SIZE, B_START, MT_ARATA } from '../src/constants.ts';
import { boardFromFen } from '../src/fen.ts';
import { fillBoardMask, forEachBoardMove } from '../src/movegen_board.ts';
import { loadFixture, loadMoveRecords } from './fixture.ts';

test('fillBoardMask ids < B_START equal the fixture legal ids on every non-terminal record', () => {
  const records = loadFixture();
  let positions = 0;
  let mismatches = 0;
  let first = '';

  for (const rec of records) {
    if (rec.terminal) continue;
    positions++;
    const board = boardFromFen(rec.fen);
    const mask = new Uint8Array(ACTION_SIZE);
    fillBoardMask(board, mask);

    const got: number[] = [];
    for (let i = 0; i < B_START; i++) if (mask[i]) got.push(i);
    got.sort((a, b) => a - b);

    const want = rec.legal.filter((id) => id < B_START).slice().sort((a, b) => a - b);

    const ok = got.length === want.length && got.every((v, i) => v === want[i]);
    if (!ok && !first) first = rec.id;
    if (!ok) mismatches++;
  }

  console.log(
    `movegen_board mask: positions=${positions} mismatches=${mismatches} first=${first || '-'}`,
  );
  assert.ok(positions > 0, 'no non-terminal fixture records');
  assert.equal(mismatches, 0, `first mismatch at ${first}`);
});

test('forEachBoardMove reproduces generate_all_moves board-block records in order', () => {
  const positions = loadMoveRecords();
  let total = 0;
  let mismatches = 0;
  let first = '';

  for (const pos of positions) {
    const board = boardFromFen(pos.fen);
    const want = pos.moves.filter((m) => m.move_type !== MT_ARATA);

    const got: Array<{
      id: number;
      move_type: number;
      piece: number;
      color: number;
      from_r: number;
      from_f: number;
      from_tier: number;
      to_r: number;
      to_f: number;
      to_tier: number;
      captured: number[][];
      draft_finished: boolean;
    }> = [];

    forEachBoardMove(board, (id, s) => {
      const toR = s[6];
      const toF = s[7];
      const capturedSel = s[10];
      const captured: number[][] = [];
      for (let t = 0; t < 3; t++) {
        if (capturedSel & (1 << t)) {
          const code = board.cells[cellIndex(toR, toF, t)];
          captured.push([typeIndex(code), t + 1]);
        }
      }
      got.push({
        id,
        move_type: s[0],
        piece: s[1],
        color: s[2],
        from_r: s[3],
        from_f: s[4],
        from_tier: s[5],
        to_r: toR,
        to_f: toF,
        to_tier: s[8],
        captured,
        draft_finished: s[9] !== 0,
      });
    });

    total += want.length;
    const n = Math.max(want.length, got.length);
    for (let i = 0; i < n; i++) {
      const w = want[i];
      const g = got[i];
      const same =
        w !== undefined &&
        g !== undefined &&
        w.id === g.id &&
        w.move_type === g.move_type &&
        w.piece === g.piece &&
        w.color === g.color &&
        w.from_r === g.from_r &&
        w.from_f === g.from_f &&
        w.from_tier === g.from_tier &&
        w.to_r === g.to_r &&
        w.to_f === g.to_f &&
        w.to_tier === g.to_tier &&
        w.draft_finished === g.draft_finished &&
        JSON.stringify(w.captured) === JSON.stringify(g.captured);
      if (!same) {
        mismatches++;
        if (!first) first = `${pos.id}#${i}`;
      }
    }
  }

  console.log(
    `movegen_board records: positions=${positions.length} total=${total} mismatches=${mismatches} first=${first || '-'}`,
  );
  assert.ok(total > 0, 'no board-block records in the move-record fixture');
  assert.equal(mismatches, 0, `first mismatch at ${first}`);
});
