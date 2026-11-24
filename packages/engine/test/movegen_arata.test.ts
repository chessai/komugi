// U2b: arata mask (B/C) and record emitter against the KOM-1055 fixture and
// the recorded move-record fixture, draft-dense set included
// (docs/design/clientside-engine-port.md section 4.1).

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { cellIndex, typeIndex } from '../src/board.ts';
import { ACTION_SIZE, B_START, MT_ARATA } from '../src/constants.ts';
import { boardFromFen } from '../src/fen.ts';
import { fillArataMask, forEachArataMove } from '../src/movegen_arata.ts';
import { loadFixture, loadMoveRecords } from './fixture.ts';

function maskIds(mask: Uint8Array): number[] {
  const out: number[] = [];
  for (let i = B_START; i < ACTION_SIZE; i++) if (mask[i]) out.push(i);
  return out;
}

function sameIds(a: number[], b: number[]): boolean {
  return a.length === b.length && a.every((v, i) => v === b[i]);
}

test('fillArataMask ids >= B_START equal the fixture legal ids on every non-terminal record', () => {
  const records = loadFixture();
  let positions = 0;
  let drafts = 0;
  let mismatches = 0;
  let first = '';
  for (const rec of records) {
    if (rec.terminal) continue;
    positions++;
    if (rec.phase === 'draft') drafts++;
    const mask = new Uint8Array(ACTION_SIZE);
    fillArataMask(boardFromFen(rec.fen), mask);
    for (let i = 0; i < B_START; i++) assert.equal(mask[i], 0, `board-block id ${i} set at ${rec.id}`);
    const want = rec.legal.filter((id) => id >= B_START).sort((a, b) => a - b);
    if (!sameIds(maskIds(mask), want)) {
      mismatches++;
      if (!first) first = rec.id;
    }
  }
  console.log(`movegen_arata mask: positions=${positions} draft=${drafts} mismatches=${mismatches} first=${first || '-'}`);
  assert.ok(positions > 0, 'no non-terminal fixture records');
  assert.ok(drafts > 0, 'no draft fixture records');
  assert.equal(mismatches, 0, `first mismatch at ${first}`);
});

test('fillArataMask ids equal the arata record ids on every move-record position (draft-dense included)', () => {
  const positions = loadMoveRecords();
  let dense = 0;
  let mismatches = 0;
  let first = '';
  for (const pos of positions) {
    if (pos.draft_dense) dense++;
    const mask = new Uint8Array(ACTION_SIZE);
    fillArataMask(boardFromFen(pos.fen), mask);
    const want = [...new Set(pos.moves.filter((m) => m.move_type === MT_ARATA).map((m) => m.id))].sort((a, b) => a - b);
    if (!sameIds(maskIds(mask), want)) {
      mismatches++;
      if (!first) first = pos.id;
    }
  }
  console.log(`movegen_arata mask vs records: positions=${positions.length} draft_dense=${dense} mismatches=${mismatches} first=${first || '-'}`);
  assert.ok(dense >= 40, `expected >= 40 draft-dense positions, got ${dense}`);
  assert.equal(mismatches, 0, `first mismatch at ${first}`);
});

test('forEachArataMove reproduces generate_all_moves arata records in order', () => {
  const positions = loadMoveRecords();
  let total = 0;
  let finishes = 0;
  let mismatches = 0;
  let first = '';
  let firstDetail = '';
  for (const pos of positions) {
    const board = boardFromFen(pos.fen);
    const want = pos.moves.filter((m) => m.move_type === MT_ARATA);
    const got: Array<Record<string, unknown>> = [];
    forEachArataMove(board, (id, s) => {
      const captured: number[][] = [];
      for (let t = 0; t < 3; t++) {
        if (s[10] & (1 << t)) captured.push([typeIndex(board.cells[cellIndex(s[6], s[7], t)]), t + 1]);
      }
      for (let k = 11; k < 16; k++) assert.equal(s[k], 0, 'reserved struct slot written');
      got.push({
        id,
        move_type: s[0],
        piece: s[1],
        color: s[2],
        from_r: s[3],
        from_f: s[4],
        from_tier: s[5],
        to_r: s[6],
        to_f: s[7],
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
      if (w !== undefined && w.draft_finished) finishes++;
      const wj =
        w === undefined
          ? 'missing'
          : JSON.stringify({
              id: w.id,
              move_type: w.move_type,
              piece: w.piece,
              color: w.color,
              from_r: w.from_r,
              from_f: w.from_f,
              from_tier: w.from_tier,
              to_r: w.to_r,
              to_f: w.to_f,
              to_tier: w.to_tier,
              captured: w.captured,
              draft_finished: w.draft_finished,
            });
      const gj = g === undefined ? 'missing' : JSON.stringify(g);
      if (wj !== gj) {
        mismatches++;
        if (!first) {
          first = `${pos.id}#${i}`;
          firstDetail = `want=${wj} got=${gj}`;
        }
      }
    }
  }
  console.log(`movegen_arata records: positions=${positions.length} total=${total} finish=${finishes} mismatches=${mismatches} first=${first || '-'} ${firstDetail}`);
  assert.ok(total > 0, 'no arata records in the move-record fixture');
  assert.ok(finishes > 0, 'no draft-finish (C block) records in the move-record fixture');
  assert.equal(mismatches, 0, `first mismatch at ${first}: ${firstDetail}`);
});
