// KOM-1198 unit 2 (docs/design/kom1198-move-io.md section 4, unit 2 in
// section 7): AC1 (position half) plus one red assert.throws case per
// validatePosition rule. Reads test/fixtures/ligungi_oracle.json directly
// (the main file only, never merged with ligungi_oracle_extra.json: section
// 4's "1441 distinct positions, 1442 records" count is this file alone).

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';

import { MARSHAL, MODE_ADVANCED, MODE_BEGINNER, MODE_INTERMEDIATE, MODE_INTRO, SOLDIER } from '../src/constants.ts';
import { canonicalFen } from '../src/fen.ts';
import { buildPosition, parsePosition, validatePosition } from '../src/position_builder.ts';
import type { PlainCell, PlainPosition } from '../src/position_builder.ts';

const FIXTURE_PATH = fileURLToPath(new URL('./fixtures/ligungi_oracle.json', import.meta.url));

type OracleFixture = { records: { komugiFen: string }[] };

function loadFixtureFens(): string[] {
  const fixture = JSON.parse(readFileSync(FIXTURE_PATH, 'utf8')) as OracleFixture;
  return fixture.records.map((r) => r.komugiFen);
}

test('buildPosition(parsePosition(fen)) === canonicalFen(fen) and parsePosition(buildPosition(pos)) round-trips, over every ligungi_oracle.json record', () => {
  const fens = loadFixtureFens();
  assert.ok(fens.length > 0, 'empty fixture');
  assert.equal(fens.length, 1442);
  for (const fen of fens) {
    const pos = parsePosition(fen);
    assert.equal(buildPosition(pos), canonicalFen(fen), fen);
    assert.deepEqual(parsePosition(buildPosition(pos)), pos, fen);
  }
});

function emptyCells(): PlainCell[][] {
  return Array.from({ length: 9 }, () => Array.from({ length: 9 }, () => [] as PlainCell));
}

// White marshal home rank9/file5 (r=8,f=4), black marshal home rank1/file5
// (r=0,f=4); otherwise empty, mode advanced, no drafting.
function basePosition(): PlainPosition {
  const cells = emptyCells();
  cells[8]![4] = [{ piece: MARSHAL, color: 0 }];
  cells[0]![4] = [{ piece: MARSHAL, color: 1 }];
  return {
    cells,
    hand: [],
    turn: 0,
    mode: MODE_ADVANCED,
    drafting: { w: false, b: false },
    moveNumber: 1,
  };
}

function clone(pos: PlainPosition): PlainPosition {
  return JSON.parse(JSON.stringify(pos)) as PlainPosition;
}

const BAD_CASES: [string, () => PlainPosition][] = [
  [
    'cells must have 9 ranks',
    () => {
      const p = basePosition();
      p.cells.pop();
      return p;
    },
  ],
  [
    'rank must have 9 files',
    () => {
      const p = basePosition();
      p.cells[1]!.pop();
      return p;
    },
  ],
  [
    'tower has more than 3 pieces',
    () => {
      const p = basePosition();
      p.cells[4]![4] = [
        { piece: SOLDIER, color: 0 },
        { piece: SOLDIER, color: 0 },
        { piece: SOLDIER, color: 0 },
        { piece: SOLDIER, color: 0 },
      ];
      return p;
    },
  ],
  [
    'piece type out of range',
    () => {
      const p = basePosition();
      p.cells[4]![4] = [{ piece: 14, color: 0 }];
      return p;
    },
  ],
  [
    'color out of range',
    () => {
      const p = basePosition();
      p.cells[4]![4] = [{ piece: SOLDIER, color: 2 }];
      return p;
    },
  ],
  [
    'turn out of range',
    () => {
      const p = basePosition();
      p.turn = 2;
      return p;
    },
  ],
  [
    'mode out of range',
    () => {
      const p = basePosition();
      p.mode = 4;
      return p;
    },
  ],
  [
    'moveNumber must be an integer >= 1 (zero)',
    () => {
      const p = basePosition();
      p.moveNumber = 0;
      return p;
    },
  ],
  [
    'moveNumber must be an integer >= 1 (non-integer)',
    () => {
      const p = basePosition();
      p.moveNumber = 1.5;
      return p;
    },
  ],
  [
    'hand entry piece type out of range',
    () => {
      const p = basePosition();
      p.hand = [{ piece: 14, color: 0, count: 1 }];
      return p;
    },
  ],
  [
    'hand entry color out of range',
    () => {
      const p = basePosition();
      p.hand = [{ piece: SOLDIER, color: 2, count: 1 }];
      return p;
    },
  ],
  [
    'hand entry count out of range',
    () => {
      const p = basePosition();
      p.hand = [{ piece: SOLDIER, color: 0, count: 0 }];
      return p;
    },
  ],
  [
    'duplicate hand entry for the same piece and color',
    () => {
      const p = basePosition();
      p.hand = [
        { piece: SOLDIER, color: 0, count: 1 },
        { piece: SOLDIER, color: 0, count: 2 },
      ];
      return p;
    },
  ],
  [
    'white must have exactly one marshal (two on the board)',
    () => {
      const p = basePosition();
      p.cells[7]![4] = [{ piece: MARSHAL, color: 0 }];
      return p;
    },
  ],
  [
    'black must have exactly one marshal (none)',
    () => {
      const p = basePosition();
      p.cells[0]![4] = [];
      return p;
    },
  ],
  [
    'a marshal cannot be covered by another piece',
    () => {
      const p = basePosition();
      p.cells[8]![4] = [
        { piece: MARSHAL, color: 0 },
        { piece: SOLDIER, color: 0 },
      ];
      return p;
    },
  ],
  [
    'a marshal over other pieces requires mode >= 2',
    () => {
      const p = basePosition();
      p.mode = MODE_BEGINNER;
      p.cells[8]![4] = [
        { piece: SOLDIER, color: 0 },
        { piece: MARSHAL, color: 0 },
      ];
      return p;
    },
  ],
  [
    'tower height > 2 requires mode 3',
    () => {
      const p = basePosition();
      p.mode = MODE_INTERMEDIATE;
      p.cells[4]![4] = [
        { piece: SOLDIER, color: 0 },
        { piece: SOLDIER, color: 1 },
        { piece: SOLDIER, color: 0 },
      ];
      return p;
    },
  ],
  [
    'white-only drafting is impossible',
    () => {
      const p = basePosition();
      p.drafting = { w: true, b: false };
      return p;
    },
  ],
  [
    'drafting is only allowed in mode 2 or 3',
    () => {
      const p = basePosition();
      p.mode = MODE_INTRO;
      p.drafting = { w: true, b: true };
      return p;
    },
  ],
  [
    'turn must be the drafting side when only one side is drafting',
    () => {
      const p = basePosition();
      p.drafting = { w: false, b: true };
      p.turn = 0;
      return p;
    },
  ],
  [
    "while drafting, every piece must sit in its color's home ranks",
    () => {
      const p = basePosition();
      p.drafting = { w: true, b: true };
      p.cells[8]![4] = [];
      p.cells[3]![4] = [{ piece: MARSHAL, color: 0 }];
      return p;
    },
  ],
];

test('validatePosition/buildPosition throws exactly one case per rule', () => {
  for (const [name, make] of BAD_CASES) {
    const pos = make();
    assert.throws(() => validatePosition(clone(pos)), /Invalid position: /, name);
    assert.throws(() => buildPosition(clone(pos)), /Invalid position: /, name);
  }
});

// Section 6: three FENs canonicalFen wrongly accepts (syntax-only), which
// validatePosition must reject once parsePosition hands them over as plain
// data -- canonicalFen alone never catches these.
const CANONICAL_FEN_ACCEPTS_BUT_ILLEGAL: string[] = [
  'm8/9/9/9/9/9/9/9/M7M -/- w 3 - 1',
  '9/9/9/9/9/9/9/9/|M:D|8 -/- w 0 w 1',
  '9/9/9/9/9/9/9/9/|D:D:D|8 D0/- b 1 - 0',
];

test('buildPosition rejects the three FENs canonicalFen wrongly accepts', () => {
  for (const fen of CANONICAL_FEN_ACCEPTS_BUT_ILLEGAL) {
    assert.doesNotThrow(() => canonicalFen(fen), fen);
    const pos = parsePosition(fen);
    assert.throws(() => buildPosition(pos), /Invalid position: /, fen);
  }
});
