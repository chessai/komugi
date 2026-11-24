// KOM-1198 unit 1: move descriptor (docs/design/kom1198-move-io.md section 3).
// AC1 (descriptor half): round trip + invariants (a)-(f) for every legal
// record at every ligungi_oracle.json position plus two hand-built positions
// (section 6) that supply the arata-with-capture / betrayal categories no
// committed fixture position has. Also the perf case (N8).

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { decodeAction } from '../src/actions.ts';
import { applyMove } from '../src/apply.ts';
import { Board, cellIndex, makeCode } from '../src/board.ts';
import { ADVANCED_POSITION, MT_BETRAY, MT_CAPTURE, MT_ROUTE, MT_TSUKE } from '../src/constants.ts';
import { boardFromFen } from '../src/fen.ts';
import { generateRecords } from '../src/movegen.ts';
import {
  describeMove,
  describeRecord,
  resolveMoveDescriptor,
  type CapturedPiece,
  type MoveDescriptor,
  type MoveHistoryEntry,
  type Square,
} from '../src/move_io.ts';
import { buildSan, resolveSan } from '../src/san.ts';
import { loadOracleFixture, positionRecords } from './oracle_helpers.ts';

function idx(sq: Square): number {
  return cellIndex(sq.rank - 1, sq.file - 1, sq.tier - 1);
}

function sanNumerals(san: string): Array<[number, number, number]> {
  const out: Array<[number, number, number]> = [];
  for (const m of san.matchAll(/\((\d+)-(\d+)-(\d+)\)/g)) {
    out.push([Number(m[1]), Number(m[2]), Number(m[3])]);
  }
  return out;
}

// Independent of describeRecord's own mapping, per the MoveEffect comment.
function expectedEffect(struct: Int32Array, captures: readonly CapturedPiece[]): string {
  const moveType = struct[0];
  if (moveType === MT_ROUTE) return 'move';
  if (moveType === MT_TSUKE) return 'stack';
  if (moveType === MT_CAPTURE) return 'capture';
  if (moveType === MT_BETRAY) return 'betray';
  return captures.length > 0 ? 'betray' : struct[8] === 1 ? 'move' : 'stack'; // MT_ARATA
}

type Category = { sawMoveBetrayal: boolean; sawArataCapture: boolean };

function checkPosition(board: Board, category: Category): number {
  const records = generateRecords(board);
  for (const { id, struct } of records) {
    const descriptor = describeRecord(board, struct);

    // round trip
    assert.equal(resolveMoveDescriptor(board, describeMove(board, id)), id);

    // (a) pre-move, for a 'move'
    if (descriptor.kind === 'move') {
      const from = descriptor.from!;
      assert.equal(board.cells[idx(from)], makeCode(descriptor.piece, descriptor.color));
      assert.equal(from.tier, board.height(from.rank - 1, from.file - 1));
    }

    // (b) on an applyMove copy
    const after = board.copy();
    applyMove(after, struct);
    assert.equal(after.cells[idx(descriptor.to)], makeCode(descriptor.piece, descriptor.color));
    assert.equal(descriptor.to.tier, after.height(descriptor.to.rank - 1, descriptor.to.file - 1));

    // (c) pre-move, every capture
    for (const c of descriptor.captures) {
      const cellCode = board.cells[cellIndex(descriptor.to.rank - 1, descriptor.to.file - 1, c.tier - 1)];
      assert.equal(cellCode, makeCode(c.piece, c.color));
      assert.notEqual(c.color, descriptor.color);
    }

    // (d) the (R-F-T) numerals in buildSan equal the descriptor's from/to exactly
    const san = buildSan(board, struct);
    const nums = sanNumerals(san);
    if (descriptor.kind === 'move') {
      const from = descriptor.from!;
      assert.deepEqual(nums[0], [from.rank, from.file, from.tier]);
      assert.deepEqual(nums[1], [descriptor.to.rank, descriptor.to.file, descriptor.to.tier]);
    } else {
      assert.deepEqual(nums[0], [descriptor.to.rank, descriptor.to.file, descriptor.to.tier]);
    }

    // (e) effect
    assert.equal(descriptor.effect, expectedEffect(struct, descriptor.captures));
    if (descriptor.kind === 'drop') {
      assert.equal(descriptor.effect === 'betray', descriptor.captures.length > 0);
    }

    // (f) draftDone
    assert.equal(descriptor.draftDone, struct[9] !== 0);

    if (descriptor.kind === 'move' && descriptor.effect === 'betray') category.sawMoveBetrayal = true;
    if (descriptor.kind === 'drop' && descriptor.captures.length > 0) category.sawArataCapture = true;
  }
  return records.length;
}

test('round trip and invariants (a)-(f) hold for every legal record at every fixture + hand-built position', () => {
  const category: Category = { sawMoveBetrayal: false, sawArataCapture: false };
  let total = 0;

  const fixture = loadOracleFixture();
  const positions = positionRecords(fixture);
  assert.ok(positions.length > 0, 'no fixture position records');
  for (const record of positions) {
    total += checkPosition(boardFromFen(record.komugiFen), category);
  }

  // Section 6's two hand-built positions: no committed fixture position has
  // an arata-with-capture move, so these supply that category (and a
  // tsuke/betrayal choice between three selections) inline.
  const handBuilt = [
    '4m4/9/9/9/9/9/9/4|d:D|4/4M4 T1D1/- w 3 - 20',
    '4m4/9/9/9/9/9/9/3|d:d|5/4|D:T|4 D2/- w 3 - 20',
  ];
  for (const fen of handBuilt) total += checkPosition(boardFromFen(fen), category);

  console.log(`move_io AC1: positions=${positions.length + handBuilt.length} records=${total}`);
  assert.ok(category.sawMoveBetrayal, 'no betrayal record seen across fixture + hand-built positions');
  assert.ok(category.sawArataCapture, 'no arata-with-capture record seen across fixture + hand-built positions');
});

test('MoveHistoryEntry accepts a SAN string and a MoveDescriptor interchangeably', () => {
  const board = boardFromFen(ADVANCED_POSITION);
  const records = generateRecords(board);
  assert.ok(records.length > 0, 'the advanced start has no legal moves');
  const { struct } = records[0]!;
  const san = buildSan(board, struct);
  const descriptor: MoveDescriptor = describeRecord(board, struct);
  const history: MoveHistoryEntry[] = [san, descriptor];
  assert.equal(typeof history[0], 'string');
  assert.equal(typeof history[1], 'object');
});

// ---- N8: perf case ----------------------------------------------------
// A dense >=60-ply history from ADVANCED_POSITION (a short oracle sequence
// replays in well under 1ms, making the 3x bound noise rather than signal).

function buildDenseHistory(plies: number): { sans: string[]; descriptors: MoveDescriptor[] } {
  let board = boardFromFen(ADVANCED_POSITION);
  const sans: string[] = [];
  const descriptors: MoveDescriptor[] = [];
  for (let i = 0; i < plies; i++) {
    const records = generateRecords(board);
    assert.ok(records.length > 0, `no legal moves at ply ${i}`);
    const { struct } = records[0]!;
    sans.push(buildSan(board, struct));
    descriptors.push(describeRecord(board, struct));
    const next = board.copy();
    applyMove(next, struct);
    board = next;
  }
  return { sans, descriptors };
}

function replayBySan(sans: readonly string[]): void {
  let board = boardFromFen(ADVANCED_POSITION);
  for (const san of sans) {
    const id = resolveSan(board, san);
    const { struct } = decodeAction(board, id);
    const next = board.copy();
    applyMove(next, struct);
    board = next;
  }
}

function replayByDescriptor(descriptors: readonly MoveDescriptor[]): void {
  let board = boardFromFen(ADVANCED_POSITION);
  for (const descriptor of descriptors) {
    const id = resolveMoveDescriptor(board, descriptor);
    const { struct } = decodeAction(board, id);
    const next = board.copy();
    applyMove(next, struct);
    board = next;
  }
}

function medianMs(fn: () => void, runs: number): number {
  const samples: number[] = [];
  for (let i = 0; i < runs; i++) {
    const start = process.hrtime.bigint();
    fn();
    const end = process.hrtime.bigint();
    samples.push(Number(end - start) / 1e6);
  }
  samples.sort((a, b) => a - b);
  return samples[Math.floor(samples.length / 2)]!;
}

test('descriptor replay is within 3x SAN replay time over a dense 70-ply history (N8)', () => {
  const { sans, descriptors } = buildDenseHistory(70);
  assert.equal(sans.length, 70);

  replayBySan(sans); // warm-up
  replayByDescriptor(descriptors); // warm-up

  const sanMs = medianMs(() => replayBySan(sans), 5);
  const descriptorMs = medianMs(() => replayByDescriptor(descriptors), 5);

  console.log(`move_io perf: sanMs=${sanMs.toFixed(3)} descriptorMs=${descriptorMs.toFixed(3)}`);
  assert.ok(descriptorMs <= sanMs * 3, `descriptor replay ${descriptorMs}ms exceeds 3x san replay ${sanMs}ms`);
});
