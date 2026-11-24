// KOM-1106 (SPLIT 2026-11-14): ligungi-only rule (2) archer tier-2/3 blocking
// and rule (3) all-or-nothing tactician betrayal. CLEAN ROOM: rule text comes
// from tickets/KOM-1106.md's prose only, never from ligungi source.
//
// "Red without its fix" verification (per the worker brief): every targeted
// test below was additionally run against a disposable copy of this tree
// under a tempfile dir, with web/engine/src/movegen_board.ts, terminal.ts and
// repetition.ts restored to their pre-fix (git HEAD) contents and
// rules_ligungi.ts removed -- confirming each test fails on that copy. See
// the final report for the exact commands; this file only encodes the
// green-with-fix assertions.

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { Board, cellIndex, colorOf, makeCode, typeIndex } from '../src/board.ts';
import {
  ACTION_SIZE,
  ARCHER,
  CANNON,
  MARSHAL,
  MODE_ADVANCED,
  MODE_INTERMEDIATE,
  MT_CAPTURE,
  SOLDIER,
  TACTICIAN,
  WHITE,
  BLACK,
} from '../src/constants.ts';
import { Game } from '../src/game.ts';
import { generateRecords, fillLegalMask } from '../src/movegen.ts';
import { RepetitionRule } from '../src/repetition.ts';
import type { RulesProfile } from '../src/repetition.ts';
import { RULES_PRESETS } from '../src/rules_config.ts';
import type { RulesConfig } from '../src/rules_config.ts';
import { archerElbowBlocked } from '../src/rules_ligungi.ts';
import { inCheck } from '../src/terminal.ts';
import {
  loadOracleFixture,
  positionRecords,
  sanForCanonical,
} from './oracle_helpers.ts';
import { boardFromFen } from '../src/fen.ts';

const PROFILES: readonly RulesProfile[] = ['komugi', 'gungijs', 'ligungi'];

const fixture = loadOracleFixture();
const positions = positionRecords(fixture);

function emptyBoard(): Board {
  const b = new Board();
  b.mode = MODE_ADVANCED;
  // Two marshals, out of the way of every scenario below -- Board itself
  // does not require this, but it keeps every board here "real" rather than
  // marshal-less.
  b.cells[cellIndex(0, 0, 0)] = makeCode(MARSHAL, WHITE);
  b.cells[cellIndex(8, 8, 0)] = makeCode(MARSHAL, BLACK);
  return b;
}

// 1-based (rank, file), 1-based tier -- matches movegen_board.ts's own
// coordinate convention throughout this file.
function place(board: Board, rank: number, file: number, ti: number, color: number, tier: number): void {
  board.cells[cellIndex(rank - 1, file - 1, tier - 1)] = makeCode(ti, color);
}

function maskIds(mask: Uint8Array): number[] {
  const out: number[] = [];
  for (let i = 0; i < mask.length; i++) if (mask[i]) out.push(i);
  return out;
}

// ---- rule (2): archer tier-2/3 blocking --------------------------------

// Direct unit tests of the helper (see rules_ligungi.ts's own comment): the
// elbow at step `s` (0-based) is always (lateral = s+1, forward = s+1), i.e.
// (py + (s+1)*dy, px + (s+1)*dx). A tier-3 mover can never actually be
// height-blocked on a real board (TIERS caps every tower at 3, and the
// existing "tier > originTier" test the engine already applies everywhere
// is therefore vacuous at originTier === 3) -- these two tests exercise the
// formula directly, with an originTier the real step-2 scan never uses, to
// pin which square each step checks independently of that pre-existing
// engine-wide limitation (reported in full in the final message).
test('archerElbowBlocked: step 1 checks the (lateral 2, forward 2) elbow', () => {
  const board = emptyBoard();
  // dy=-1 (forward), dx=+1 (lateral): elbow at step 1 is (py-2, px+2).
  place(board, 7, 7, SOLDIER, BLACK, 3); // taller than originTier=2
  assert.equal(archerElbowBlocked(board, 9, 5, -1, 1, 1, 2), true);
  // Nothing at the OTHER step's elbow (6, 8) -- step 2 alone unaffected.
  assert.equal(archerElbowBlocked(board, 9, 5, -1, 1, 2, 2), false);
});

// HELPER-LEVEL ONLY (unreachable in real play: design doc
// training-rules-ligungi.md 1.1). A real step-2 scan is only ever reached
// when length >= 3, i.e. pcTier + pcarry - 1 >= 3; for ARCHER (pcarry === 1
// in every direction, constants.ts's own PROBE_PCARRY row) that means
// pcTier === 3, never 2. originTier=2 here is therefore synthetic -- no call
// site in movegen_board.ts/terminal.ts ever reaches archerElbowBlocked with
// step=2 and originTier=2 together -- it exists purely to pin the k=3 elbow
// formula ((lateral 3, forward 3) = (py+3*dy, px+3*dx)) in isolation from
// step 1's.
test('archerElbowBlocked (helper-level, synthetic originTier): step 2 checks the (lateral 3, forward 3) elbow, independently of step 1', () => {
  const board = emptyBoard();
  // elbow at step 2 is (py-3, px+3) = (6, 8); step 1's own elbow (7,7) left
  // clear, demonstrating the two checks are independent (ticket: "blocked
  // separately by (2,2) and by (3,3)").
  assert.equal(archerElbowBlocked(board, 9, 5, -1, 1, 1, 2), false); // (7,7) clear
  place(board, 6, 8, CANNON, BLACK, 3);
  assert.equal(archerElbowBlocked(board, 9, 5, -1, 1, 2, 2), true);
});

// ---- F4 vacuity pins (training-rules-ligungi.md 1.1 / KOM-1125 AC3) ------
// These are guards, not reproductions: all four are already true today: they
// pin invariants the design doc calls out so a future change that breaks
// them is caught here rather than discovered in play.

test('archerElbowBlocked: a tier-3 archer is never elbow-blocked (board height caps at 3, so "height > originTier" can never hold at originTier=3)', () => {
  const board = emptyBoard();
  place(board, 7, 7, SOLDIER, BLACK, 3); // (lateral 2, forward 2) elbow, at the board's own height ceiling
  place(board, 6, 8, CANNON, BLACK, 3); // (lateral 3, forward 3) elbow, likewise
  assert.equal(archerElbowBlocked(board, 9, 5, -1, 1, 1, 3), false, 'step 1 elbow');
  assert.equal(archerElbowBlocked(board, 9, 5, -1, 1, 2, 3), false, 'step 2 elbow');
});

test('archerElbowBlocked: a height-2 piece at the (2,2) elbow does not block a tier-2 archer (equal height, not taller)', () => {
  const board = emptyBoard();
  place(board, 7, 7, SOLDIER, BLACK, 2);
  assert.equal(archerElbowBlocked(board, 9, 5, -1, 1, 1, 2), false);
});

test('archerElbowBlocked: a tier-1 archer is unaffected -- step 0 (its own (lateral 1, forward 2) jump) is never elbow-checked under any profile', () => {
  const board = emptyBoard();
  board.turn = WHITE;
  place(board, 9, 5, ARCHER, WHITE, 1);
  place(board, 7, 6, SOLDIER, BLACK, 1); // the step-0 target itself (py-2, px+1) -- a capture, present under every profile since step 0 is never elbow-checked
  for (const rules of PROFILES) {
    const reaches = generateRecords(board, rules).some((r) => r.struct[6] === 7 - 1 && r.struct[7] === 6 - 1);
    assert.equal(reaches, true, `${rules}: expected the tier-1 archer's (lateral 1, forward 2) capture to be generated`);
  }
});

test('archerElbowBlocked: under a height-2 mode (maxTier=2, movegen_board.ts), no tower can ever be taller than a tier-2 archer, so its elbow check is always false', () => {
  const board = emptyBoard();
  board.mode = MODE_INTERMEDIATE; // forEachBoardMove/fillBoardMask: maxTier = 2 under every mode but MODE_ADVANCED
  place(board, 7, 7, SOLDIER, BLACK, 2); // the tallest a tower can ever legally become under this mode
  assert.equal(archerElbowBlocked(board, 9, 5, -1, 1, 1, 2), false);
});

// End-to-end: a tier-2 archer's (lateral 2, forward 3) target, blocked only
// under ligungi by a tall piece at the (2,2) elbow -- off the archer's own
// staircase path, so komugi/gungijs (unchanged) still reach it.
function buildTier2ArcherBoard(): { board: Board; fromSq: number; toSq: number } {
  const board = emptyBoard();
  board.turn = WHITE;
  place(board, 9, 5, ARCHER, WHITE, 2); // origin, tier 2 (stacked on a filler below it)
  place(board, 8, 5, SOLDIER, WHITE, 1); // the piece the archer is stacked on
  place(board, 7, 7, SOLDIER, BLACK, 3); // elbow (lateral2,forward2), taller than tier 2
  // fromSq/toSq per movegen_board.ts's sqIndex(r,f) = r*9 + (8-f), 0-based.
  const fromSq = (9 - 1) * 9 + (8 - (5 - 1));
  const toSq = (6 - 1) * 9 + (8 - (7 - 1));
  return { board, fromSq, toSq };
}

test('archer tier-2 (+-2,3) target: reachable under komugi/gungijs, blocked under ligungi by the (2,2) elbow', () => {
  for (const rules of ['komugi', 'gungijs'] as const) {
    const { board } = buildTier2ArcherBoard();
    const ids = generateRecords(board, rules).map((r) => r.struct);
    const reaches = ids.some((s) => s[6] === 6 - 1 && s[7] === 7 - 1);
    assert.equal(reaches, true, `${rules}: expected the archer to reach (6,7)`);
  }
  const { board } = buildTier2ArcherBoard();
  const ligungiReaches = generateRecords(board, 'ligungi').some((r) => r.struct[6] === 5 && r.struct[7] === 6);
  assert.equal(ligungiReaches, false, 'ligungi: expected the (2,2) elbow to block (6,7)');
});

// KOM-1203 knob isolation: from the komugi preset, flip ONLY
// archerElbowBlocking to true (tacticianBetrayalMode stays komugi's own
// 'any-subset') -- the elbow block must fire exactly like ligungi's, and the
// betrayal behaviour (an unrelated knob) must stay at komugi's any-subset.
const ARCHER_ELBOW_ONLY: RulesConfig = { ...RULES_PRESETS.komugi, archerElbowBlocking: true };

test('knob isolation (archerElbowBlocking only): blocks the (2,2) elbow like ligungi, unlike komugi/gungijs', () => {
  const { board } = buildTier2ArcherBoard();
  const reaches = generateRecords(board, ARCHER_ELBOW_ONLY).some((r) => r.struct[6] === 6 - 1 && r.struct[7] === 7 - 1);
  assert.equal(reaches, false, 'expected the (2,2) elbow to block the jump');
});

test('knob isolation (archerElbowBlocking only): betrayal mode (an unrelated knob) stays at komugi\'s any-subset', () => {
  const { board, fromSq, toSq } = buildBetrayalBoard();
  assert.deepEqual(betraySels(board, ARCHER_ELBOW_ONLY, fromSq, toSq), [0, 1, 2, 3]);
});

test('archer tier-2 (+-2,3) target: fillLegalMask mask bit follows the same profile (mask path, not just records)', () => {
  const { board, fromSq, toSq } = buildTier2ArcherBoard();
  // A0 (plain move/capture) block id for this from/to pair.
  const id = fromSq * 81 + toSq;

  const komugiMask = new Uint8Array(ACTION_SIZE);
  fillLegalMask(board, komugiMask, 'komugi');
  assert.equal(komugiMask[id], 1, 'komugi: mask should allow the jump');

  const { board: board2 } = buildTier2ArcherBoard();
  const ligungiMask = new Uint8Array(ACTION_SIZE);
  fillLegalMask(board2, ligungiMask, 'ligungi');
  assert.equal(ligungiMask[id], 0, 'ligungi: mask should block the jump');
});

// F4 end-to-end: archer blocking via Game.validMoves (the real play path,
// not movegen internals), from the oracle fixture directly -- both colours
// and both lateral signs. Each id below carries only the `archer_blocking`
// class (no self_exposure mixed in), and its canonical string is the sole
// komugi/gungijs-only extra move oracle_legal.test.ts's F2 test already
// flags for that record (komugiFen is a position record's own start FEN).
const TARGETED_ARCHER_E2E = [
  { id: 'pos:targeted-archer/00', canonical: 'M:8-7>5-5:move', color: 'white', lateralSign: -1 },
  { id: 'pos:targeted-archer/05', canonical: 'M:8-5>5-7:move', color: 'white', lateralSign: 1 },
  { id: 'pos:targeted-archer/10', canonical: 'M:2-7>5-5:move', color: 'black', lateralSign: -1 },
  { id: 'pos:targeted-archer/15', canonical: 'M:2-5>5-7:move', color: 'black', lateralSign: 1 },
] as const;

for (const { id, canonical, color, lateralSign } of TARGETED_ARCHER_E2E) {
  test(`end-to-end (Game.validMoves): ${id} (${color}, lateral ${lateralSign > 0 ? '+1' : '-1'}) -- komugi/gungijs allow the archer jump, ligungi blocks it`, () => {
    const record = positions.find((r) => r.id === id);
    if (!record) throw new Error(`fixture record missing: ${id}`);
    for (const rules of ['komugi', 'gungijs'] as const) {
      const game = new Game(boardFromFen(record.komugiFen), RepetitionRule.PLACEMENT, 1, rules);
      const { id: actionId } = sanForCanonical(game.board, canonical);
      assert.equal(game.validMoves()[actionId], 1, `${rules}: expected the jump allowed`);
    }
    const game = new Game(boardFromFen(record.komugiFen), RepetitionRule.PLACEMENT, 1, 'ligungi');
    const { id: actionId } = sanForCanonical(game.board, canonical);
    assert.equal(game.validMoves()[actionId], 0, 'ligungi: expected the jump blocked');
  });
}

// inCheck: an archer's attack reaching through the (2,2) elbow is seen under
// komugi/gungijs and NOT seen under ligungi (terminal.ts's own copy of the
// scan, independent of movegen_board.ts).
function buildArcherAttackBoard(): Board {
  const board = emptyBoard();
  place(board, 9, 5, ARCHER, WHITE, 2);
  place(board, 8, 5, SOLDIER, WHITE, 1);
  place(board, 6, 7, MARSHAL, BLACK, 1); // the attacked square
  // Remove the default far-corner black marshal placed by emptyBoard() so
  // there is exactly one (the one being attacked).
  board.cells[cellIndex(8, 8, 0)] = 0;
  return board;
}

test('inCheck: an archer attack through the (2,2) elbow is seen under komugi/gungijs, not under ligungi', () => {
  for (const rules of ['komugi', 'gungijs'] as const) {
    const board = buildArcherAttackBoard();
    assert.equal(inCheck(board, BLACK, rules), true, `${rules}: expected black's marshal in check`);
  }
  const clear = buildArcherAttackBoard();
  assert.equal(inCheck(clear, BLACK, 'ligungi'), true, 'ligungi (no blocker): expected check');

  const blocked = buildArcherAttackBoard();
  // BLACK, not WHITE: height-blocking is color-independent, and a same-color
  // (WHITE) piece here would itself attack (6,7) by its own normal move,
  // invalidating the test.
  place(blocked, 7, 7, SOLDIER, BLACK, 3); // the (2,2) elbow, taller than tier 2
  assert.equal(inCheck(blocked, BLACK, 'komugi'), true, 'komugi: unaffected by the elbow');
  assert.equal(inCheck(blocked, BLACK, 'ligungi'), false, 'ligungi: elbow blocks the attack');
});

// ---- rule (3): all-or-nothing tactician betrayal -----------------------

// A tactician TSUKE onto a two-enemy stack (soldier at tier 1, cannon at
// tier 2): komugi/gungijs offer all three non-empty subsets (sel 1, 2, 3);
// ligungi offers only the full subset (sel 3), alongside the plain tsuke
// (sel 0, unaffected by the betrayal rule).
function buildBetrayalBoard(): { board: Board; fromSq: number; toSq: number } {
  const board = emptyBoard();
  board.turn = WHITE;
  // Tactician stacked to tier 2 (on a friendly filler): a tier-1 tactician
  // cannot even reach a height-2 target (reachability's own "taller than
  // the mover" test, movegen_board.ts:getAvailableSquares, applies to the
  // destination too, not just intermediate squares).
  place(board, 5, 5, SOLDIER, WHITE, 1);
  place(board, 5, 5, TACTICIAN, WHITE, 2);
  place(board, 4, 6, SOLDIER, BLACK, 1);
  place(board, 4, 6, CANNON, BLACK, 2);
  board.hand[SOLDIER * 2 + WHITE] = 2;
  board.hand[CANNON * 2 + WHITE] = 2;
  const fromSq = (5 - 1) * 9 + (8 - (5 - 1));
  const toSq = (4 - 1) * 9 + (8 - (6 - 1));
  return { board, fromSq, toSq };
}

function betraySels(board: Board, rules: RulesProfile | RulesConfig, fromSq: number, toSq: number): number[] {
  const sels: number[] = [];
  for (const { struct } of generateRecords(board, rules)) {
    const tr = struct[6];
    const tf = struct[7];
    const toSqGot = tr * 9 + (8 - tf);
    const fr = struct[3];
    const ff = struct[4];
    if (fr < 0) continue; // arata, not a board move
    const fromSqGot = fr * 9 + (8 - ff);
    if (fromSqGot === fromSq && toSqGot === toSq && struct[0] !== MT_CAPTURE) sels.push(struct[10]);
  }
  return sels.sort((a, b) => a - b);
}

test('tactician betrayal (board path, records): komugi/gungijs offer every subset, ligungi offers only the full one', () => {
  for (const rules of ['komugi', 'gungijs'] as const) {
    const { board, fromSq, toSq } = buildBetrayalBoard();
    assert.deepEqual(betraySels(board, rules, fromSq, toSq), [0, 1, 2, 3], rules);
  }
  const { board, fromSq, toSq } = buildBetrayalBoard();
  assert.deepEqual(betraySels(board, 'ligungi', fromSq, toSq), [0, 3], 'ligungi');
});

test('tactician betrayal (board path, mask): komugi/gungijs set sel 0-3, ligungi sets only sel 0 and 3', () => {
  const A1_START = 6561;
  const { board, fromSq, toSq } = buildBetrayalBoard();
  const komugiMask = new Uint8Array(ACTION_SIZE);
  fillLegalMask(board, komugiMask, 'komugi');
  for (const sel of [0, 1, 2, 3]) {
    assert.equal(komugiMask[A1_START + (fromSq * 4 + sel) * 81 + toSq], 1, `komugi sel=${sel}`);
  }

  const { board: board2 } = buildBetrayalBoard();
  const ligungiMask = new Uint8Array(ACTION_SIZE);
  fillLegalMask(board2, ligungiMask, 'ligungi');
  assert.equal(ligungiMask[A1_START + (fromSq * 4 + 0) * 81 + toSq], 1, 'ligungi sel=0');
  assert.equal(ligungiMask[A1_START + (fromSq * 4 + 1) * 81 + toSq], 0, 'ligungi sel=1');
  assert.equal(ligungiMask[A1_START + (fromSq * 4 + 2) * 81 + toSq], 0, 'ligungi sel=2');
  assert.equal(ligungiMask[A1_START + (fromSq * 4 + 3) * 81 + toSq], 1, 'ligungi sel=3');
});

// KOM-1203 knob isolation: from the komugi preset, flip ONLY
// tacticianBetrayalMode to 'all-or-nothing' (every other knob, including
// selfExposureFilter and archerElbowBlocking, stays at komugi's own `false`
// value) -- the betrayal-combo behaviour must match ligungi's own
// all-or-nothing result exactly, and the archer-elbow behaviour (an
// unrelated knob) must stay at komugi's (unblocked), proving the two knobs
// are independently settable, not coupled to one profile string.
const TACTICIAN_BETRAYAL_ONLY: RulesConfig = { ...RULES_PRESETS.komugi, tacticianBetrayalMode: 'all-or-nothing' };

test('knob isolation (tacticianBetrayalMode only): matches ligungi\'s all-or-nothing betrayal, unlike komugi/gungijs', () => {
  const { board, fromSq, toSq } = buildBetrayalBoard();
  assert.deepEqual(betraySels(board, TACTICIAN_BETRAYAL_ONLY, fromSq, toSq), [0, 3]);
});

test('knob isolation (tacticianBetrayalMode only): archer-elbow blocking (an unrelated knob) stays at komugi\'s unblocked behaviour', () => {
  const { board } = buildTier2ArcherBoard();
  const reaches = generateRecords(board, TACTICIAN_BETRAYAL_ONLY).some((r) => r.struct[6] === 6 - 1 && r.struct[7] === 7 - 1);
  assert.equal(reaches, true, 'expected the archer jump to still be reachable (archerElbowBlocking unaffected by the betrayal-mode flip)');
});

// R1 (review 2026-11-14, docs/reviews/2026-11-14-kom1106-ts-archer-betrayal-review.md:233/250):
// a SYNTHETIC partial-qualifying tower -- one enemy type (soldier) passes
// the hand test, the other (cannon) does not (zero in hand) -- so
// getBetrayalCombos/qualifyingTierMask's own betrayalOptions/qualifying mask
// can only ever offer the soldier alone, never the full tower. komugi/
// gungijs therefore offer that one partial subset (sel 1); ligungi's
// all-or-nothing rule (fullBetrayalCombo) requires the FULL tower to
// qualify, so a partial-only qualifying set collapses to no betrayal sel at
// all (only the plain stack, sel 0) -- not even the one partial subset
// komugi/gungijs allow. Mirrored for white and black (DIRS sign flip).
function buildPartialQualifyBoard(color: number): { board: Board; fromSq: number; toSq: number } {
  const board = emptyBoard();
  board.turn = color;
  if (color === WHITE) {
    place(board, 5, 5, SOLDIER, WHITE, 1);
    place(board, 5, 5, TACTICIAN, WHITE, 2);
    place(board, 4, 6, SOLDIER, BLACK, 1);
    place(board, 4, 6, CANNON, BLACK, 2);
  } else {
    place(board, 5, 5, SOLDIER, BLACK, 1);
    place(board, 5, 5, TACTICIAN, BLACK, 2);
    place(board, 6, 4, SOLDIER, WHITE, 1);
    place(board, 6, 4, CANNON, WHITE, 2);
  }
  board.hand[SOLDIER * 2 + color] = 2; // qualifies: enough hand for the tier-1 enemy
  board.hand[CANNON * 2 + color] = 0; // does not qualify: zero hand for the tier-2 enemy
  const fromSq = (5 - 1) * 9 + (8 - (5 - 1));
  const toR = color === WHITE ? 4 : 6;
  const toF = color === WHITE ? 6 : 4;
  const toSq = (toR - 1) * 9 + (8 - (toF - 1));
  return { board, fromSq, toSq };
}

for (const [label, color] of [['white', WHITE], ['black', BLACK]] as const) {
  test(`R1 partial-qualify tower (${label}, records): komugi/gungijs offer only the one qualifying subset, ligungi offers none (only the plain stack)`, () => {
    for (const rules of ['komugi', 'gungijs'] as const) {
      const { board, fromSq, toSq } = buildPartialQualifyBoard(color);
      assert.deepEqual(betraySels(board, rules, fromSq, toSq), [0, 1], `${rules} (${label})`);
    }
    const { board, fromSq, toSq } = buildPartialQualifyBoard(color);
    assert.deepEqual(betraySels(board, 'ligungi', fromSq, toSq), [0], `ligungi (${label})`);
  });

  test(`R1 partial-qualify tower (${label}, mask): komugi/gungijs set sel 0 and 1, ligungi sets only sel 0`, () => {
    const A1_START = 6561;
    const { board, fromSq, toSq } = buildPartialQualifyBoard(color);
    const komugiMask = new Uint8Array(ACTION_SIZE);
    fillLegalMask(board, komugiMask, 'komugi');
    assert.equal(komugiMask[A1_START + (fromSq * 4 + 0) * 81 + toSq], 1, `komugi sel=0 (${label})`);
    assert.equal(komugiMask[A1_START + (fromSq * 4 + 1) * 81 + toSq], 1, `komugi sel=1 (${label})`);
    assert.equal(komugiMask[A1_START + (fromSq * 4 + 3) * 81 + toSq], 0, `komugi sel=3 (${label})`);

    const { board: board2 } = buildPartialQualifyBoard(color);
    const ligungiMask = new Uint8Array(ACTION_SIZE);
    fillLegalMask(board2, ligungiMask, 'ligungi');
    assert.equal(ligungiMask[A1_START + (fromSq * 4 + 0) * 81 + toSq], 1, `ligungi sel=0 (${label})`);
    assert.equal(ligungiMask[A1_START + (fromSq * 4 + 1) * 81 + toSq], 0, `ligungi sel=1 (${label})`);
    assert.equal(ligungiMask[A1_START + (fromSq * 4 + 3) * 81 + toSq], 0, `ligungi sel=3 (${label})`);
  });
}

// Drop (arata) path: a legal drop target's top tier is always friendly (or
// the square is empty), so any enemy a tactician's drop could convert can
// only ever be the single tier directly beneath that friendly top -- never
// two distinct enemy tiers at once. Betrayal there is already all-or-nothing
// by construction (movegen_arata.ts's own comment), under every profile;
// this is a sanity check of that invariant, not a fix (nothing in
// movegen_arata.ts changed for this ticket -- see the final report).
test('tactician betrayal (drop path): already all-or-nothing under every profile (no partial subset exists to offer)', () => {
  const board = emptyBoard();
  board.turn = WHITE;
  place(board, 5, 5, SOLDIER, BLACK, 1);
  place(board, 5, 5, SOLDIER, WHITE, 2); // friendly top, room for one more (tier 3)
  board.hand[TACTICIAN * 2 + WHITE] = 1;
  board.hand[SOLDIER * 2 + WHITE] = 1;
  board.handOrder[0] = TACTICIAN * 2 + WHITE;
  board.handOrderLen = 1;

  for (const rules of PROFILES) {
    const sels = new Set<number>();
    for (const { struct } of generateRecords(board, rules)) {
      if (struct[3] >= 0) continue; // board move, not arata
      if (struct[6] === 4 && struct[7] === 4) sels.add(struct[10]);
    }
    // Not just "no multi-bit partial subset" (0/1 only) -- the drop is
    // actually offered both ways (plain stack and the single betrayal),
    // under every profile, so an empty set would also have to fail here.
    assert.deepEqual([...sels].sort((a, b) => a - b), [0, 1], rules);
  }
});

// ---- twin-agreement, over the oracle fixture ---------------------------

function marshalSquare(board: Board, color: number): { rank: number; file: number } | null {
  for (let r = 0; r < 9; r++) {
    for (let f = 0; f < 9; f++) {
      const top = board.top(r, f);
      if (top !== 0 && typeIndex(top) === MARSHAL && colorOf(top) === color) {
        return { rank: r + 1, file: f + 1 };
      }
    }
  }
  return null;
}

test('twin agreement: fillLegalMask ids equal generateRecords ids, over every fixture position, every profile', () => {
  for (const record of positions) {
    for (const rules of PROFILES) {
      const board = boardFromFen(record.komugiFen);
      const mask = new Uint8Array(ACTION_SIZE);
      fillLegalMask(board, mask, rules);
      const recordIds = new Set(generateRecords(board, rules).map((r) => r.id));
      const maskIdSet = new Set(maskIds(mask));
      assert.deepEqual(maskIdSet, recordIds, `${record.id} (${rules})`);
    }
  }
});

test('twin agreement: inCheck equals "an enemy capture onto the marshal is generated", over every fixture position, every profile', () => {
  for (const record of positions) {
    // Board moves (so board captures) never exist during the draft
    // (movegen_board.ts:forEachBoardMove's own early return); inCheck is
    // purely structural and does not gate on draft state, so this
    // equivalence only holds once the draft is over.
    const draftBoard = boardFromFen(record.komugiFen);
    if (draftBoard.draftW !== 0 || draftBoard.draftB !== 0) continue;
    for (const rules of PROFILES) {
      for (const color of [WHITE, BLACK] as const) {
        const board = boardFromFen(record.komugiFen);
        const got = inCheck(board, color, rules);

        const square = marshalSquare(board, color);
        let wantCapture = false;
        if (square !== null) {
          board.turn = 1 - color;
          wantCapture = generateRecords(board, rules).some(
            (r) =>
              r.struct[0] === MT_CAPTURE &&
              r.struct[6] === square.rank - 1 &&
              r.struct[7] === square.file - 1,
          );
        }
        assert.equal(got, wantCapture, `${record.id} (${rules}) color=${color}`);
      }
    }
  }
});
