// U3: dispatcher, actions, apply, SAN, history replay against the KOM-1055
// fixture and the recorded move-record fixture (docs/design/clientside-engine-port.md
// section 4.1 row "apply / SAN / history replay (U3)").

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { decodeAction, capturedFromMask } from '../src/actions.ts';
import { applyMove } from '../src/apply.ts';
import { cellIndex, typeIndex } from '../src/board.ts';
import { ADVANCED_POSITION } from '../src/constants.ts';
import { boardFromFen, boardToFen, canonicalFen } from '../src/fen.ts';
import { generateRecords } from '../src/movegen.ts';
import { buildSan, resolveSan, sanRecords } from '../src/san.ts';
import { loadFixture, loadMoveRecords } from './fixture.ts';

test('replaying a fixture record\'s history from the advanced start lands on its fen', () => {
  const records = loadFixture();
  let mismatches = 0;
  let first = '';

  for (const rec of records) {
    const board = boardFromFen(ADVANCED_POSITION);
    for (const san of rec.history) {
      const id = resolveSan(board, san);
      const { struct } = decodeAction(board, id);
      applyMove(board, struct);
    }
    const got = canonicalFen(boardToFen(board));
    const want = canonicalFen(rec.fen);
    if (got !== want && !first) first = rec.id;
    if (got !== want) mismatches++;
  }

  console.log(`san replay: positions=${records.length} mismatches=${mismatches} first=${first || '-'}`);
  assert.ok(records.length > 0, 'no fixture records');
  assert.equal(mismatches, 0, `first mismatch at ${first}`);
});

test('legal_san equals buildSan(decodeAction(id)) for every fixture legal id, in order', () => {
  const records = loadFixture();
  let positions = 0;
  let mismatches = 0;
  let first = '';

  for (const rec of records) {
    if (rec.legal.length === 0) continue;
    positions++;
    const board = boardFromFen(rec.fen);
    for (let i = 0; i < rec.legal.length; i++) {
      const { struct } = decodeAction(board, rec.legal[i]);
      const got = buildSan(board, struct);
      if (got !== rec.legal_san[i] && !first) first = `${rec.id}#${i}`;
      if (got !== rec.legal_san[i]) mismatches++;
    }
  }

  console.log(`san legal_san: positions=${positions} mismatches=${mismatches} first=${first || '-'}`);
  assert.ok(positions > 0, 'no fixture records with legal moves');
  assert.equal(mismatches, 0, `first mismatch at ${first}`);
});

test('no fixture position has a duplicate move-record id or SAN', () => {
  const positions = loadMoveRecords();
  let dupIds = 0;
  let dupSans = 0;
  let firstId = '';
  let firstSan = '';

  for (const pos of positions) {
    const seenIds = new Set<number>();
    const seenSans = new Set<string>();
    for (const m of pos.moves) {
      if (seenIds.has(m.id)) {
        dupIds++;
        if (!firstId) firstId = `${pos.id}#${m.id}`;
      }
      seenIds.add(m.id);
      if (seenSans.has(m.san)) {
        dupSans++;
        if (!firstSan) firstSan = `${pos.id}:${m.san}`;
      }
      seenSans.add(m.san);
    }
  }

  console.log(
    `san duplicates: positions=${positions.length} dupIds=${dupIds} first=${firstId || '-'} dupSans=${dupSans} first=${firstSan || '-'}`,
  );
  assert.ok(positions.length > 0, 'no move-record positions');
  assert.equal(dupIds, 0, `duplicate id at ${firstId}`);
  assert.equal(dupSans, 0, `duplicate SAN at ${firstSan}`);
});

test('movegen.ts generateRecords/san.ts sanRecords never collide on id or SAN', () => {
  const positions = loadMoveRecords();
  let dupIds = 0;
  let dupSans = 0;
  let first = '';

  for (const pos of positions) {
    const board = boardFromFen(pos.fen);
    const seenIds = new Set<number>();
    const seenSans = new Set<string>();
    for (const rec of sanRecords(board)) {
      if (seenIds.has(rec.id)) dupIds++;
      seenIds.add(rec.id);
      if (seenSans.has(rec.san)) {
        dupSans++;
        if (!first) first = `${pos.id}:${rec.san}`;
      }
      seenSans.add(rec.san);
    }
  }

  console.log(`movegen/san duplicates: positions=${positions.length} dupIds=${dupIds} dupSans=${dupSans} first=${first || '-'}`);
  assert.equal(dupIds, 0, 'generateRecords produced a duplicate id at a single position');
  assert.equal(dupSans, 0, `sanRecords produced a duplicate SAN at ${first}`);
});

test('decodeAction returns the first matching generate_all_moves record, field by field', () => {
  const positions = loadMoveRecords();
  let total = 0;
  let mismatches = 0;
  let first = '';

  for (const pos of positions) {
    const board = boardFromFen(pos.fen);
    for (const want of pos.moves) {
      total++;
      const { struct } = decodeAction(board, want.id);
      const toR = struct[6];
      const toF = struct[7];
      const captured: number[][] = [];
      for (let t = 0; t < 3; t++) {
        if (struct[10] & (1 << t)) {
          captured.push([typeIndex(board.cells[cellIndex(toR, toF, t)]), t + 1]);
        }
      }
      const same =
        struct[0] === want.move_type &&
        struct[1] === want.piece &&
        struct[2] === want.color &&
        struct[3] === want.from_r &&
        struct[4] === want.from_f &&
        struct[5] === want.from_tier &&
        toR === want.to_r &&
        toF === want.to_f &&
        struct[8] === want.to_tier &&
        (struct[9] !== 0) === want.draft_finished &&
        JSON.stringify(captured) === JSON.stringify(want.captured);
      if (!same) {
        mismatches++;
        if (!first) first = `${pos.id}#${want.id}`;
      }
    }
  }

  console.log(`decodeAction: positions=${positions.length} total=${total} mismatches=${mismatches} first=${first || '-'}`);
  assert.ok(total > 0, 'no move records to decode');
  assert.equal(mismatches, 0, `first mismatch at ${first}`);
});

test('decodeAction uses the same captured-mask decode as apply.ts/san.ts (sanity on a live board)', () => {
  const board = boardFromFen(ADVANCED_POSITION);
  const records = generateRecords(board);
  assert.ok(records.length > 0, 'the advanced start has no legal moves');
  const { id, struct } = records[0];
  const decoded = decodeAction(board, id);
  assert.deepEqual(Array.from(decoded.struct), Array.from(struct));
  // capturedFromMask is exercised directly: a plain route has no captures.
  assert.deepEqual(capturedFromMask(board, struct[6], struct[7], struct[10]), []);
});
