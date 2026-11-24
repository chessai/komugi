// KOM-1148: KOM-1139 review fix-ups F1 -- the random-position/seeded-game
// differentials and the targeted a..m fixtures the review's own scratch
// probe (docs/reviews/attachments/kom1139/review_fuzz.test.ts) ran, now
// committed. CLEAN ROOM: rule text comes from the ticket's design-doc
// excerpt only, never from ligungi source.
//
// The review's scratch probe compared two EXPORTED trees (one with
// isMoveLegalUnderRule1 patched to return true, as the naive oracle's
// candidate generator). This file gets the same candidate generator inside
// this one tree, via setRule1FilterEnabledForTest -- a test-only switch
// (KOM-1148 F1) that makes isMoveLegalUnderRule1 unconditionally accept,
// i.e. rules (2)/(3) movegen with no rule-(1) filtering. Always re-enabled
// immediately after the candidate list is built (pseudoRecords' finally),
// so every other check below (mask/records) runs with the real filter.
//
// "Red without its fix" verification (per the worker brief): this file's
// four tests were run against this same tree with
// web/engine/src/ligungi_filter.ts's isMoveLegalUnderRule1 mutated to match
// each of the KOM-1139 review's mutants; the targeted-fixtures test goes red
// under M6 (fixture d) and M9 (fixture a'), the random/game/fixture
// differentials go red under M1/M2/M7/M8/M9/M10, and M10 additionally turns
// fixture m red -- matching the review's mutant table. See the final report
// for the exact commands.

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { Board, cellIndex, makeCode, typeIndex } from '../src/board.ts';
import { applyMove } from '../src/apply.ts';
import {
  ACTION_SIZE,
  ARCHER,
  CANNON,
  GENERAL,
  MARSHAL,
  MODE_ADVANCED,
  MT_CAPTURE,
  MUSKETEER,
  NUM_PIECE_TYPES,
  SOLDIER,
  SPY,
} from '../src/constants.ts';
import { boardFromFen } from '../src/fen.ts';
import { fillLegalMask, generateRecords, type MoveRecord } from '../src/movegen.ts';
import { setRule1FilterEnabledForTest } from '../src/ligungi_filter.ts';
import { inCheck, isTerminalNonmate } from '../src/terminal.ts';
import { loadOracleFixture, positionRecords } from './oracle_helpers.ts';

// The pseudo-ligungi candidate set (rules 2/3 applied, rule 1 bypassed): the
// naive differential's input, reproducing the review's exported "pseudo"
// tree without needing one.
function pseudoRecords(board: Board): MoveRecord[] {
  setRule1FilterEnabledForTest(false);
  try {
    return generateRecords(board, 'ligungi');
  } finally {
    setRule1FilterEnabledForTest(true);
  }
}

// The naive oracle itself, over the pseudo candidate set: apply, then check
// the mover's own marshal -- unless the move captured the opponent's
// marshal outright (the design doc's own exception).
function naiveIds(board: Board): Set<number> {
  const mover = board.turn;
  const out = new Set<number>();
  for (const { id, struct } of pseudoRecords(board)) {
    if (struct[0] === MT_CAPTURE) {
      const top = board.top(struct[6], struct[7]);
      if (top !== 0 && typeIndex(top) === MARSHAL) {
        out.add(id);
        continue;
      }
    }
    const clone = board.copy();
    applyMove(clone, struct);
    if (!inCheck(clone, mover, 'ligungi')) out.add(id);
  }
  return out;
}

function maskIds(board: Board): Set<number> {
  const mask = new Uint8Array(ACTION_SIZE);
  fillLegalMask(board, mask, 'ligungi');
  const out = new Set<number>();
  for (let i = 0; i < mask.length; i++) if (mask[i]) out.add(i);
  return out;
}

function recIds(board: Board): Set<number> {
  return new Set(generateRecords(board, 'ligungi').map((r) => r.id));
}

function setsEqual(a: ReadonlySet<number>, b: ReadonlySet<number>): boolean {
  if (a.size !== b.size) return false;
  for (const x of a) if (!b.has(x)) return false;
  return true;
}

function setDiff(a: ReadonlySet<number>, b: ReadonlySet<number>): number[] {
  return [...a].filter((x) => !b.has(x)).slice(0, 8);
}

function check(board: Board, label: string): void {
  const naive = naiveIds(board);
  const mask = maskIds(board.copy());
  const records = recIds(board.copy());
  assert.ok(
    setsEqual(naive, mask),
    `${label}: mask != naive, extra=${JSON.stringify(setDiff(mask, naive))} missing=${JSON.stringify(setDiff(naive, mask))}`,
  );
  assert.ok(
    setsEqual(naive, records),
    `${label}: records != naive, extra=${JSON.stringify(setDiff(records, naive))} missing=${JSON.stringify(setDiff(naive, records))}`,
  );
}

// Deterministic PRNG (mulberry32), seeded identically on every run.
function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function randint(r: () => number, lo: number, hi: number): number {
  return lo + Math.floor(r() * (hi - lo + 1));
}

const NON_MARSHAL = Array.from({ length: NUM_PIECE_TYPES }, (_, i) => i).filter((i) => i !== MARSHAL);

// Port of the review's randomAdversarial (a port of cymovegen's own
// section-10 generator): mixed towers, marshals standing on 1-2 enemy
// pieces, hands, the occasional draft position.
function randomAdversarial(r: () => number): Board {
  const isDraft = r() < 0.05;
  const turn = randint(r, 0, 1);
  const b = new Board();
  b.mode = MODE_ADVANCED;
  b.turn = turn;
  let dw = 0;
  let db = 0;
  if (isDraft) {
    if (r() < 0.5) {
      dw = db = 1;
    } else if (turn === 0) {
      dw = 1;
    } else {
      db = 1;
    }
  }
  b.draftW = dw;
  b.draftB = db;

  const squares: Array<[number, number]> = [];
  for (let rr = 0; rr < 9; rr++) for (let ff = 0; ff < 9; ff++) squares.push([rr, ff]);
  for (let i = squares.length - 1; i > 0; i--) {
    const j = Math.floor(r() * (i + 1));
    [squares[i], squares[j]] = [squares[j], squares[i]];
  }
  let si = 0;
  const setTower = (sq: [number, number], codes: number[]) => {
    codes.forEach((c, t) => {
      b.cells[cellIndex(sq[0], sq[1], t)] = c;
    });
  };
  const onBoard = [false, false];
  for (const color of [0, 1]) {
    const drafting = color === 0 ? dw : db;
    if (drafting && r() < 0.5) continue;
    const below: number[] = [];
    if (r() < 0.5) {
      const k = randint(r, 1, 2);
      for (let i = 0; i < k; i++) below.push(makeCode(NON_MARSHAL[randint(r, 0, 12)], 1 - color));
    }
    setTower(squares[si++], [...below, makeCode(MARSHAL, color)]);
    onBoard[color] = true;
  }
  const extra = randint(r, 3, 12);
  for (let k = 0; k < extra && si < squares.length; k++) {
    const h = randint(r, 1, 3);
    const codes: number[] = [];
    for (let t = 0; t < h; t++) codes.push(makeCode(NON_MARSHAL[randint(r, 0, 12)], randint(r, 0, 1)));
    setTower(squares[si++], codes);
  }
  const order: number[] = [];
  for (const color of [0, 1]) {
    if (!onBoard[color]) {
      b.hand[MARSHAL * 2 + color] = 1;
      order.push(MARSHAL * 2 + color);
    }
    for (const ti of NON_MARSHAL) {
      if (r() < 0.25) {
        b.hand[ti * 2 + color] = randint(r, 1, 2);
        order.push(ti * 2 + color);
      }
    }
  }
  for (let i = order.length - 1; i > 0; i--) {
    const j = Math.floor(r() * (i + 1));
    [order[i], order[j]] = [order[j], order[i]];
  }
  for (const e of order) b.pushHandOrder(e);
  return b;
}

test('review fuzz: every non-draft fixture position, mask == records == naive(pseudo-ligungi)', () => {
  const positions = positionRecords(loadOracleFixture());
  let checked = 0;
  for (const rec of positions) {
    const board = boardFromFen(rec.komugiFen);
    if (board.draftW || board.draftB) continue;
    check(board, rec.id);
    checked++;
  }
  assert.ok(checked > 0, 'expected at least one non-draft fixture position');
});

const RANDPOS = 2500;
const RANDPOS_CHECKED_FLOOR = 2000;

test('review fuzz: seeded random adversarial positions', () => {
  const r = rng(1128);
  let checked = 0;
  let checks = 0;
  for (let i = 0; i < RANDPOS; i++) {
    const b = randomAdversarial(r);
    if (b.draftW || b.draftB) continue;
    checked++;
    if (inCheck(b, b.turn, 'ligungi')) checks++;
    check(b, `randpos#${i}`);
  }
  assert.ok(checked >= RANDPOS_CHECKED_FLOOR, `expected >= ${RANDPOS_CHECKED_FLOOR} non-draft positions, got ${checked}`);
  assert.ok(checks > 0, 'expected at least one in-check random position');
});

const GAMES_FLOOR = 50;
const GAMES_MAX = 400;
const PLIES_PER_GAME = 200;
const IN_CHECK_PLIES_FLOOR = 500;

test('review fuzz: seeded short games from the ADVANCED start', () => {
  let games = 0;
  let plies = 0;
  let inCheckPlies = 0;
  while (games < GAMES_MAX && inCheckPlies < IN_CHECK_PLIES_FLOOR * 2) {
    const r = rng(1128 + games);
    const b = boardFromFen(
      '9/9/9/9/9/9/9/9/9 M1G1I1J2W2N3R2S2F2D4C1A2K1T1/m1g1i1j2w2n3r2s2f2d4c1a2k1t1 w 3 wb 1',
    );
    for (let p = 0; p < PLIES_PER_GAME; p++) {
      if (isTerminalNonmate(b, 1)) break;
      const recs = generateRecords(b, 'ligungi');
      if (recs.length === 0) break;
      if (!(b.draftW || b.draftB)) {
        check(b, `game${games} ply${p}`);
        if (inCheck(b, b.turn, 'ligungi')) inCheckPlies++;
      }
      const pick = recs[Math.floor(r() * recs.length)];
      applyMove(b, pick.struct);
      plies++;
    }
    games++;
    if (games >= GAMES_FLOOR && inCheckPlies >= IN_CHECK_PLIES_FLOOR) break;
  }
  assert.ok(games >= GAMES_FLOOR, `expected >= ${GAMES_FLOOR} games, got ${games}`);
  assert.ok(
    inCheckPlies >= IN_CHECK_PLIES_FLOOR,
    `expected >= ${IN_CHECK_PLIES_FLOOR} in-check plies, got ${inCheckPlies} over ${games} games / ${plies} plies`,
  );
});

// Targeted fixtures ported from the KOM-1139 review's scratch probe (itself
// a port of cead8c6c test_ligungi_filter.py's a, a', b, c, c', d, e1-e3, f,
// f+, g, m). Each `expectFiltered` fixture must have the naive oracle
// actually remove something from the pseudo-legal candidate set -- a bad
// port that filters nothing would stay green under a broken filter too.
test('review fuzz: targeted fixtures (Cython a..m), mask == records == naive, and each fixture filters something where expected', () => {
  const W = 0;
  const B = 1;
  const code = makeCode;
  const mk = (turn: number, towers: Array<[number, number, number[]]>, hand = '-'): Board => {
    const b = boardFromFen(`9/9/9/9/9/9/9/9/9 ${hand}/- w 3 - 1`);
    b.turn = turn;
    for (const [r, f, codes] of towers) {
      codes.forEach((c, t) => {
        b.cells[cellIndex(r - 1, f - 1, t)] = c;
      });
    }
    return b;
  };
  const G = GENERAL;
  const S_ = SOLDIER;
  const M = MARSHAL;
  const CA = CANNON;
  const AR = ARCHER;
  const MU = MUSKETEER;
  const SP = SPY;

  const fx: Array<[string, Board, boolean]> = [
    ['a reveal', mk(W, [[3, 5, [code(G, B), code(S_, W)]], [5, 5, [code(M, W)]]]), true],
    ["a' reveal in check", mk(W, [[5, 5, [code(M, W)]], [5, 1, [code(G, B)]], [4, 4, [code(G, B), code(S_, W)]], [1, 9, [code(M, B)]]]), true],
    ['b discovered slider', mk(W, [[5, 1, [code(G, B)]], [5, 5, [code(M, W)]], [5, 3, [code(S_, W)]]]), true],
    ['c leaper lane', mk(W, [[3, 5, [code(CA, B)]], [6, 5, [code(M, W)]], [5, 5, [code(S_, W), code(S_, W)]]]), true],
    ["c' archer height drop", mk(W, [[5, 5, [code(S_, B), code(AR, B)]], [8, 3, [code(M, W)]], [7, 3, [code(S_, W), code(S_, W), code(S_, W)]]]), true],
    ['d capture opens lane', mk(B, [[7, 5, [code(S_, W), code(MU, W)]], [5, 5, [code(M, B)]], [6, 5, [code(G, B), code(SP, W), code(SP, W)]], [6, 3, [code(S_, B), code(S_, B), code(G, B)]]]), true],
    ['e1 marshal into covered', mk(W, [[5, 5, [code(M, W)]], [3, 3, [code(G, B)]]]), true],
    ['e2 marshal tsuke height', mk(W, [[7, 5, [code(M, W)]], [6, 5, [code(S_, W)]], [5, 5, [code(S_, B)]]]), false],
    ['e3 marshal captures defended', mk(W, [[7, 5, [code(M, W)]], [6, 5, [code(S_, B)]], [4, 5, [code(G, B)]]]), true],
    ['f drop in check', mk(W, [[5, 5, [code(M, W)]], [1, 5, [code(G, B)]], [2, 9, [code(S_, W)]]], 'D1'), true],
    ['f+ drop vs leaper lane', mk(W, [[3, 5, [code(CA, B)]], [6, 5, [code(M, W)]], [5, 5, [code(S_, W)]], [1, 9, [code(S_, W)]]], 'D1'), true],
    ['g marshal capture exception', mk(W, [[5, 5, [code(M, W)]], [1, 5, [code(G, B)]], [5, 9, [code(G, W)]], [5, 7, [code(M, B)]]]), true],
    ['m marshal on enemy', mk(W, [[5, 5, [code(G, B), code(M, W)]]]), true],
  ];

  for (const [label, b, expectFiltered] of fx) {
    check(b, label);
    if (expectFiltered) {
      const pseudoN = pseudoRecords(b.copy()).length;
      const naiveN = naiveIds(b.copy()).size;
      assert.ok(naiveN < pseudoN, `${label}: fixture filters nothing (bad port): naive=${naiveN} pseudo=${pseudoN}`);
    }
  }

  // g: the marshal capture itself must be legal.
  const g = fx[11][1];
  const cap = generateRecords(g, 'ligungi').find(
    (r) => r.struct[0] === MT_CAPTURE && r.struct[6] === 5 - 1 && r.struct[7] === 7 - 1,
  );
  assert.ok(cap !== undefined, 'g: marshal capture must be generated');
  const mask = maskIds(g.copy());
  assert.ok(mask.has(cap!.id), 'g: marshal capture must be legal');
});
