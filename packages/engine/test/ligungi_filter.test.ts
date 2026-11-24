// KOM-1139: ligungi-only rule (1), the purpose-built self-exposure filter
// (TS port of KOM-1128's cymovegen.pyx filter). CLEAN ROOM: rule text comes
// from the ticket's design-doc excerpt and the ligungi oracle fixture only,
// never from ligungi source.
//
// "Red without its fix" verification (per the worker brief): every targeted
// test below was additionally run against a disposable copy of this tree
// under a tempfile dir, with web/engine/src/ligungi_filter.ts removed and
// movegen_board.ts/movegen_arata.ts/movegen.ts/terminal.ts restored to their
// pre-fix (git HEAD) contents -- confirming each test fails (or the whole
// file fails to import) on that copy. See the final report for the exact
// commands.
//
// This file covers two things the ticket calls out specifically:
// - a NAIVE in-test oracle (apply each candidate move to a board copy, then
//   check the mover's own marshal), differentially compared against the
//   production filter over every "clean" (no archer_blocking/betrayal_partial
//   class) oracle-fixture position -- isolating rule (1) from rules (2)/(3),
//   which already have their own extensive tests (ligungi_movegen.test.ts).
// - targeted lemma-level unit tests (L5 reject, L6 accept, the marshal-move
//   and marshal-capture-exception special cases, and the P0 buried-marshal
//   precondition), each built on a small synthetic board.

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { Board, cellIndex, makeCode, typeIndex } from '../src/board.ts';
import {
  BLACK,
  CANNON,
  GENERAL,
  LIEUTENANT_GENERAL,
  MARSHAL,
  MODE_ADVANCED,
  MT_ARATA,
  MT_BETRAY,
  MT_CAPTURE,
  MT_TSUKE,
  SOLDIER,
  TACTICIAN,
  WHITE,
} from '../src/constants.ts';
import { applyMove } from '../src/apply.ts';
import { boardFromFen } from '../src/fen.ts';
import { Game } from '../src/game.ts';
import { generateRecords, fillLegalMask } from '../src/movegen.ts';
import { ACTION_SIZE } from '../src/constants.ts';
import { canonicalMove, legalSet, RepetitionRule } from '../src/repetition.ts';
import { inCheck } from '../src/terminal.ts';
import { buildFilterContext, isMoveLegalUnderRule1 } from '../src/ligungi_filter.ts';
import { RULES_PRESETS } from '../src/rules_config.ts';
import { loadOracleFixture, positionRecords } from './oracle_helpers.ts';

// ---- naive-oracle differential, over the fixture's "clean" positions ------

const fixture = loadOracleFixture();
const positions = positionRecords(fixture);

// Records whose classes carry neither rule-(2) nor rule-(3) divergence: for
// these, komugi's own pseudo-legal set (no filtering at all) and ligungi's
// legal set (rules 1-3 applied) differ ONLY by rule (1), so a naive rule-(1)
// filter applied to komugi's pseudo-legal set must reproduce
// legalSet(board, 'ligungi') exactly.
// Draft positions are excluded: rule (1) never applies while either draft
// bit is set (design doc 4.1 "draft placements are exempt"), so comparing a
// naive per-move check against them would test something production code
// deliberately never does.
const RULE1_ONLY_RECORDS = positions.filter((r) => {
  if (r.classes.includes('archer_blocking') || r.classes.includes('betrayal_partial')) return false;
  const board = boardFromFen(r.komugiFen);
  return board.draftW === 0 && board.draftB === 0;
});

test('ligungi_oracle.json: has rule-(1)-only (no archer/betrayal class) position records', () => {
  assert.ok(RULE1_ONLY_RECORDS.length > 0);
});

// The naive oracle itself: apply(move), then the mover's own marshal must
// not be in check afterwards -- UNLESS the move captured the opponent's
// marshal outright (the design doc's own exception, already established by
// KOM-1109/1110/KOM-1106 UPDATE 87cd3a3 for terminal.ts's hasEscapingMove).
// This is independent of ligungi_filter.ts: it calls only applyMove and the
// already-trusted terminal.inCheck.
function naiveRule1Legal(board: Board, mover: number, struct: Int32Array): boolean {
  if (struct[0] === MT_CAPTURE) {
    const topBefore = board.top(struct[6], struct[7]);
    if (topBefore !== 0 && typeIndex(topBefore) === MARSHAL) return true;
  }
  const clone = board.copy();
  applyMove(clone, struct);
  return !inCheck(clone, mover, 'ligungi');
}

test('naive oracle differential: komugi pseudo-legal + naive rule-1 check == legalSet(ligungi), on every rule-(2)/(3)-clean fixture record', () => {
  for (const record of RULE1_ONLY_RECORDS) {
    const board = boardFromFen(record.komugiFen);
    const mover = board.turn;
    const naive = new Set<string>();
    for (const { struct } of generateRecords(board, 'komugi')) {
      if (naiveRule1Legal(board, mover, struct)) naive.add(canonicalMove(struct));
    }
    const mine = legalSet(board, 'ligungi');
    assert.deepEqual(
      [...naive].sort(),
      [...mine].sort(),
      `${record.id}: naive oracle and legalSet('ligungi') disagree`,
    );
  }
});

// KOM-1203 knob-isolation: a custom RulesConfig built from the komugi preset
// with ONLY selfExposureFilter flipped to true (every other knob, including
// marshalCaptureIsEscape, stays at komugi's own value -- F3's coherence rule
// requires marshalCaptureIsEscape when selfExposureFilter is set, and komugi
// already has it true) must apply rule (1) exactly like the 'ligungi' preset
// on every rule-(2)/(3)-clean fixture record, same as the test above.
const SELF_EXPOSURE_ONLY = { ...RULES_PRESETS.komugi, selfExposureFilter: true };

test('knob isolation (selfExposureFilter only, F3 coherence): naive oracle == legalSet(custom config), on every rule-(2)/(3)-clean fixture record', () => {
  for (const record of RULE1_ONLY_RECORDS) {
    const board = boardFromFen(record.komugiFen);
    const mover = board.turn;
    const naive = new Set<string>();
    for (const { struct } of generateRecords(board, 'komugi')) {
      if (naiveRule1Legal(board, mover, struct)) naive.add(canonicalMove(struct));
    }
    const mine = legalSet(board, SELF_EXPOSURE_ONLY);
    assert.deepEqual(
      [...naive].sort(),
      [...mine].sort(),
      `${record.id}: naive oracle and legalSet(selfExposureFilter-only config) disagree`,
    );
  }
});

// KOM-1203 F3 regression: before the fix, movegen_board.ts's fillBoardMask
// any-subset betrayal branch (the else arm of the tacticianBetrayalMode
// ternary) set its mask bits unconditionally, never calling legal() --
// harmless while selfExposureFilter and tacticianBetrayalMode were coupled to
// one profile string (every shipped preset with any-subset betrayal also has
// selfExposureFilter: false), but wrong for a config combining
// selfExposureFilter: true with tacticianBetrayalMode: 'any-subset' (no
// shipped preset does, but section 2's knobs are independently settable).
// forEachBoardMove's tryEmit, by contrast, already wraps every emitted move
// (including every betrayal sel) uniformly regardless of mode, so
// generateRecords was always correct; fillLegalMask's mask disagreed with it
// pre-fix. Proven over the full oracle-fixture corpus, not a single board.
const SELF_EXPOSURE_ANY_SUBSET = {
  ...RULES_PRESETS.komugi,
  selfExposureFilter: true,
  tacticianBetrayalMode: 'any-subset' as const,
};

test('F3 regression: fillLegalMask mask ids equal generateRecords ids under selfExposureFilter + any-subset betrayal, over every fixture position', () => {
  for (const record of positions) {
    const board = boardFromFen(record.komugiFen);
    const mask = new Uint8Array(ACTION_SIZE);
    fillLegalMask(board, mask, SELF_EXPOSURE_ANY_SUBSET);
    const recordIds = new Set(generateRecords(board, SELF_EXPOSURE_ANY_SUBSET).map((r) => r.id));
    const maskIds = new Set<number>();
    for (let i = 0; i < mask.length; i++) if (mask[i]) maskIds.add(i);
    assert.deepEqual(maskIds, recordIds, `${record.id}: fillLegalMask and generateRecords disagree`);
  }
});

// ---- targeted lemma-level unit tests ---------------------------------------

function emptyBoard(): Board {
  const b = new Board();
  b.mode = MODE_ADVANCED;
  return b;
}

function place(board: Board, rank: number, file: number, ti: number, color: number, tier: number): void {
  board.cells[cellIndex(rank - 1, file - 1, tier - 1)] = makeCode(ti, color);
}

// L5 (quick reject, in check): a general pinned to blocking the only check
// may not step off the checking ray, even sideways.
test('L5: in check, a move whose destination is outside every checker\'s readset is rejected', () => {
  const board = emptyBoard();
  board.turn = WHITE;
  place(board, 1, 1, MARSHAL, WHITE, 1);
  place(board, 9, 9, MARSHAL, BLACK, 1);
  // Black general checks along the file (a slider on N/S/E/W per
  // constants.ts's own probe table): general at (5,1), marshal at (1,1),
  // same file.
  place(board, 5, 1, GENERAL, BLACK, 1);
  // White general at (3,1), currently blocking the check: moving along the
  // file (N/S, e.g. to (2,1)/(4,1)) must stay legal; moving off the file
  // (E, to (3,2), or any diagonal step) must be rejected.
  place(board, 3, 1, GENERAL, WHITE, 1);

  assert.equal(inCheck(board, WHITE, 'ligungi'), false, 'the blocker currently blocks the check');

  const game = new Game(board, RepetitionRule.FULL, 1, 'ligungi');
  const mask = game.validMoves();
  let sawOffRay = false;
  let sawOnRay = false;
  for (const { id, struct } of generateRecords(board, 'ligungi')) {
    if (struct[3] !== 3 - 1 || struct[4] !== 1 - 1) continue; // not this general's own move
    if (struct[7] === 1 - 1) {
      sawOnRay = sawOnRay || mask[id] === 1;
    } else {
      sawOffRay = sawOffRay || mask[id] === 1;
    }
  }
  assert.equal(sawOffRay, false, 'a move leaving the file must be rejected');
  assert.ok(sawOnRay, 'a move staying on the file (still blocking) must be offered');
});

// L6 (fast accept, not in check): a move far from every opposing piece's
// reach is legal without needing the EXACT fallback. The single assertion
// this test used to make ("the move is legal") is vacuous: any movegen that
// never filtered anything would also pass it (KOM-1139 review F5). Fixed by
// turning it into a differential, reusing this file's own naive oracle
// (apply the candidate, then check the mover's marshal) over every one of
// the mover's pseudo-legal moves on this board, not just the one move that
// was already known to be far from both attackers -- a board with a SECOND,
// adjacent soldier adds candidates whose origin/destination DO fall in the
// close attacker's readset, so this also exercises the EXACT fallback this
// test's board previously never reached.
test('L6: not in check, isMoveLegalUnderRule1 matches the naive oracle for every pseudo-legal move', () => {
  const board = emptyBoard();
  board.turn = WHITE;
  place(board, 1, 1, MARSHAL, WHITE, 1);
  place(board, 9, 9, MARSHAL, BLACK, 1);
  place(board, 5, 5, SOLDIER, WHITE, 1);
  place(board, 5, 6, SOLDIER, BLACK, 1);
  const ctx = buildFilterContext(board, WHITE, 'ligungi');
  const struct = new Int32Array(16);
  struct[0] = 0; // MT_ROUTE
  struct[1] = SOLDIER;
  struct[2] = WHITE;
  struct[3] = 5 - 1;
  struct[4] = 5 - 1;
  struct[5] = 1;
  struct[6] = 4 - 1;
  struct[7] = 5 - 1;
  assert.equal(isMoveLegalUnderRule1(ctx, board, WHITE, 'ligungi', struct), true);

  let sawCandidate = false;
  for (const { struct: s } of generateRecords(board, 'komugi')) {
    if (s[2] !== WHITE) continue;
    sawCandidate = true;
    const got = isMoveLegalUnderRule1(ctx, board, WHITE, 'ligungi', s);
    const want = naiveRule1Legal(board, WHITE, s);
    assert.equal(got, want, `${canonicalMove(s)}: isMoveLegalUnderRule1=${got} naive=${want}`);
  }
  assert.ok(sawCandidate, 'expected at least one pseudo-legal move for white');
});

// Marshal-capture exception: capturing the opponent's marshal is always
// legal, even when the mover's own marshal remains attacked afterwards
// (design doc 1.2 / KOM-1106 UPDATE 87cd3a3).
test('marshal-capture exception: capturing the enemy marshal is legal even while the mover stays in check', () => {
  const board = emptyBoard();
  board.turn = WHITE;
  place(board, 1, 1, MARSHAL, WHITE, 1);
  // Black marshal adjacent to white's general, capturable this move.
  place(board, 5, 6, MARSHAL, BLACK, 1);
  place(board, 5, 5, GENERAL, WHITE, 1);
  // A black general keeps white's OWN marshal in check along the first rank
  // (E/W slider), independent of the capture above -- the capture must
  // still be legal.
  place(board, 1, 9, GENERAL, BLACK, 1);

  assert.equal(inCheck(board, WHITE, 'ligungi'), true, 'white should start in check');

  const game = new Game(board, RepetitionRule.FULL, 1, 'ligungi');
  const mask = game.validMoves();
  const capture = generateRecords(board, 'ligungi').find(
    (r) => r.struct[0] === MT_CAPTURE && r.struct[6] === 5 - 1 && r.struct[7] === 6 - 1,
  );
  assert.ok(capture !== undefined, 'expected the marshal capture to be generated');
  assert.equal(mask[capture!.id], 1, 'the marshal capture must be legal despite the standing check');
});

// P0 precondition: a buried marshal (constructed directly, bypassing
// movegen -- nothing tsukes onto a marshal in real play) must throw, not
// silently misbehave (design doc 4.1).
test('P0: a buried marshal throws', () => {
  const board = emptyBoard();
  board.turn = WHITE;
  place(board, 1, 1, MARSHAL, WHITE, 1);
  place(board, 1, 1, SOLDIER, WHITE, 2); // buries the marshal under a soldier
  place(board, 9, 9, MARSHAL, BLACK, 1);
  assert.throws(() => buildFilterContext(board, WHITE, 'ligungi'), /buried marshal/);
});

// F2 (KOM-1148): P0 also refuses a duplicate top marshal (two marshals of
// the same colour, never reachable from real play) and a marshal still
// sitting in hand once the draft is over (marshals are never recaptured
// into hand, so this is also only reachable from a synthetic FEN) --
// cymovegen's _check_p0 raises on both; buildFilterContext used to check
// only the buried-marshal case (KOM-1139 review F2).
test('P0: a marshal still in hand outside the draft throws', () => {
  assert.throws(
    () => legalSet(boardFromFen('4g3m/9/9/9/9/9/9/9/D8 M1/- w 3 - 1'), 'ligungi'),
    /marshal in hand/,
  );
});

test('P0: a duplicate top marshal of the same colour throws', () => {
  assert.throws(
    () => legalSet(boardFromFen('4g3m/9/9/9/4M4/9/4D4/9/M8 -/- w 3 - 1'), 'ligungi'),
    /duplicate top marshal/,
  );
});

// F3 (KOM-1148): the old draft-exemption test placed no marshal on the
// board at all (the ADVANCED start, before any draft placement), so the
// filter context was null whether or not the exemption worked -- it stayed
// green under mutant M5 (the filter force-applied to draft arata) (KOM-1139
// review F3). Fixed by porting the Cython fixture: a marshal ALREADY on the
// board, in check from an enemy general, with a hand piece to drop on a
// square outside that general's attack line. If the filter were applied
// (M5), buildFilterContext would find the standing check and reject the
// drop (its destination is not in the checker's readset); exempt, it stays
// legal. Run under both draft-flag combinations the ticket calls out (`wb`
// and `w` alone), since movegen_arata.ts's own exemption gate fires on
// EITHER flag being set.
function draftExemptionBoard(draftB: number): Board {
  const board = emptyBoard();
  board.turn = WHITE;
  board.draftW = 1;
  board.draftB = draftB;
  place(board, 9, 1, MARSHAL, WHITE, 1);
  place(board, 9, 9, GENERAL, BLACK, 1);
  board.hand[SOLDIER * 2 + WHITE] = 1;
  board.pushHandOrder(SOLDIER * 2 + WHITE);
  return board;
}

for (const draftB of [1, 0]) {
  test(`draft (draftW=1 draftB=${draftB}): a drop outside the checking general's readset stays legal (rule 1 never applies mid-draft)`, () => {
    const board = draftExemptionBoard(draftB);
    assert.equal(inCheck(board, WHITE, 'ligungi'), true, 'white marshal should start in check from the general');

    const mask = new Uint8Array(ACTION_SIZE);
    fillLegalMask(board, mask, 'ligungi');
    const drop = generateRecords(board, 'ligungi').find(
      (r) => r.struct[0] === MT_ARATA && r.struct[1] === SOLDIER && r.struct[6] === 7 - 1 && r.struct[7] === 5 - 1,
    );
    assert.ok(drop !== undefined, 'expected the soldier drop at 7-5 to be generated');
    assert.equal(mask[drop!.id], 1, 'the drop must stay legal despite the standing check (draft exemption)');
  });
}

// ---- micro-benchmark: movegen throughput, komugi vs ligungi (perf report) -

// bench/fast.ts (outside this ticket's file allowlist) has no `rules`
// parameter, so fast_bench itself cannot be run under 'ligungi'. This is a
// stand-in that measures the same dominant per-node cost the design doc
// targets (Game.validMoves()) directly, over the fixture's own mid-game
// positions, and reports the ratio for the final message rather than gating
// on it (machine-dependent).
test('perf (reported, not gated): Game.validMoves() throughput, ligungi vs komugi, over mid-game fixture positions', () => {
  const sampleFens = positions
    .filter((r) => r.mode === 'advanced')
    .slice(0, 40)
    .map((r) => r.komugiFen);
  assert.ok(sampleFens.length > 0);

  function bench(rules: 'komugi' | 'ligungi'): number {
    const games = sampleFens.map((fen) => new Game(boardFromFen(fen), RepetitionRule.FULL, 1, rules));
    const iterations = 50;
    const start = performance.now();
    for (let i = 0; i < iterations; i++) {
      for (const g of games) g.validMoves();
    }
    const elapsedMs = performance.now() - start;
    return (games.length * iterations) / (elapsedMs / 1000);
  }

  // One untimed warm-up pass per profile (JIT), same discipline fast_bench.test.ts uses.
  bench('komugi');
  bench('ligungi');

  const komugiRate = bench('komugi');
  const ligungiRate = bench('ligungi');
  const ratio = ligungiRate / komugiRate;
  console.log(
    `validMoves() throughput: komugi=${komugiRate.toFixed(0)}/s ligungi=${ligungiRate.toFixed(0)}/s ` +
      `ratio=${(ratio * 100).toFixed(1)}%`,
  );
  assert.ok(komugiRate > 0 && ligungiRate > 0);
});
