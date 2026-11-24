// U8: root reconstruction (design doc section 2.5 "Root reconstruction",
// section 6 U8 acceptance). No fixture needed -- every case is a hand-built
// FEN/history pair, the way test/game.test.ts's rules-cases tests hand-build
// FENs for the branches random games never hit.

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { Board, cellIndex, makeCode } from '../src/board.ts';
import { ADVANCED_POSITION, ARCHER, GENERAL, MARSHAL, MODE_ADVANCED, SOLDIER, WHITE, BLACK } from '../src/constants.ts';
import { boardFromFen, boardToFen, canonicalFen } from '../src/fen.ts';
import { Game } from '../src/game.ts';
import { generateRecords } from '../src/movegen.ts';
import { describeMove } from '../src/move_io.ts';
import { RepetitionRule } from '../src/repetition.ts';
import { RULES_PRESETS } from '../src/rules_config.ts';
import type { RulesConfig } from '../src/rules_config.ts';
import { resolveSan, sanRecords } from '../src/san.ts';
import {
  HISTORY_ABSENT,
  HISTORY_MISMATCH,
  HISTORY_PRESENT,
  gameFromHistory,
  historyKey,
  replayHistory,
  rootGame,
} from '../src/root.ts';

// The advanced start's first enumerated legal move (any legal SAN does: the
// tests below only need one real move whose replay is checkable).
function firstLegalSan(): string {
  const board = boardFromFen(ADVANCED_POSITION);
  const records = sanRecords(board);
  assert.ok(records.length > 0, 'no legal move at the advanced start');
  return records[0].san;
}

// Splits the FIRST multi-digit empty-run in a FEN's placement field into two
// adjacent digits summing to the same count -- a different spelling of the
// identical position (ADVANCED_POSITION's ranks are all "9", so this always
// finds one). Exercises the "non-canonical spacing" comparison path without
// needing a history.
function nonCanonicalSpelling(fen: string): string {
  const [placement, ...rest] = fen.split(' ');
  const splitRank = placement.replace(/[2-9]/, (d) => `1${Number(d) - 1}`);
  assert.notEqual(splitRank, placement, `no splittable empty-run digit in ${placement}`);
  return [splitRank, ...rest].join(' ');
}

test('no history: the FEN path under FULL, turns = 0, stamp absent', () => {
  const { game, stamp } = rootGame(ADVANCED_POSITION, undefined);
  assert.equal(stamp, HISTORY_ABSENT);
  assert.equal(game.rule, RepetitionRule.FULL);
  assert.equal(game.turns, 0);
  assert.equal(game.lastMove, null);
  assert.equal(canonicalFen(boardToFen(game.board)), canonicalFen(ADVANCED_POSITION));
});

test('null history behaves exactly like an absent key', () => {
  const { stamp } = rootGame(ADVANCED_POSITION, null);
  assert.equal(stamp, HISTORY_ABSENT);
});

test('a good history: present under PLACEMENT, keys and turns carried', () => {
  const san = firstLegalSan();
  const replayed = gameFromHistory([san]);
  const reachedFen = boardToFen(replayed.board);

  const { game, stamp } = rootGame(reachedFen, [san]);
  assert.equal(stamp, HISTORY_PRESENT);
  assert.equal(game.rule, RepetitionRule.PLACEMENT);
  assert.equal(game.turns, 1);
  assert.notEqual(game.lastMove, null);
  assert.equal(canonicalFen(boardToFen(game.board)), canonicalFen(reachedFen));
});

test('present, with the caller\'s FEN in non-canonical spacing', () => {
  const san = firstLegalSan();
  const replayed = gameFromHistory([san]);
  const reachedFen = boardToFen(replayed.board);
  const spelledDifferently = nonCanonicalSpelling(reachedFen);
  assert.notEqual(spelledDifferently, reachedFen);

  const { stamp } = rootGame(spelledDifferently, [san]);
  assert.equal(stamp, HISTORY_PRESENT);
});

test('a non-canonical site FEN with no history is still accepted on the absent path', () => {
  const spelledDifferently = nonCanonicalSpelling(ADVANCED_POSITION);
  const { stamp, game } = rootGame(spelledDifferently, undefined);
  assert.equal(stamp, HISTORY_ABSENT);
  assert.equal(canonicalFen(boardToFen(game.board)), canonicalFen(ADVANCED_POSITION));
});

test('an illegal SAN in the history: mismatch, falls back to the FEN path', () => {
  const { game, stamp } = rootGame(ADVANCED_POSITION, ['this-is-not-a-legal-san']);
  assert.equal(stamp, HISTORY_MISMATCH);
  assert.equal(game.rule, RepetitionRule.FULL);
  assert.equal(game.turns, 0);
});

test('a history that replays legally but not to the claimed FEN: mismatch', () => {
  const san = firstLegalSan();
  // The move necessarily changes the position, so claiming it reaches the
  // UNCHANGED start is always a mismatch.
  const result = rootGame(ADVANCED_POSITION, [san]);
  assert.equal(result.stamp, HISTORY_MISMATCH);
  assert.equal(result.game.rule, RepetitionRule.FULL);
  assert.equal(result.game.turns, 0);
});

test('history must be an array: a non-array value is a mismatch, not a throw', () => {
  const { stamp } = rootGame(ADVANCED_POSITION, 'not-an-array');
  assert.equal(stamp, HISTORY_MISMATCH);
});

test('D9/U16: a history resolveSan cannot replay stamps mismatch, not a throw', () => {
  // Whatever the site's own laxer rules (e.g. gungi.js's one-debit
  // betrayal, design D9) would have accepted, this port's single engine
  // either resolves a SAN the same way or falls back to mismatch -- it
  // never throws out of rootGame.
  const result = rootGame(ADVANCED_POSITION, ['返', '新']);
  assert.equal(result.stamp, HISTORY_MISMATCH);
});

test('sanity: resolveSan itself throws on an illegal SAN (the catch in rootGame is doing the work)', () => {
  assert.throws(() => resolveSan(boardFromFen(ADVANCED_POSITION), 'nonsense'));
});

// ---- KOM-1198 unit 3: historyKey / replayHistory (design section 5, 6) ----

test('historyKey: an all-string history returns the raw strings unchanged, non-canonical spelling included', () => {
  const san = firstLegalSan();
  // resolveSan strips a trailing '='/'#' (san.ts:75), so this is a
  // non-canonical spelling of `san` that resolveSan would normalize away --
  // historyKey must NOT canonicalize (today's key stays byte-identical).
  const spelled = `${san}=`;
  assert.deepEqual(historyKey([spelled]), [spelled]);
  assert.notEqual(spelled, san);
});

test('historyKey: a descriptor history returns its canonical SAN', () => {
  const board = boardFromFen(ADVANCED_POSITION);
  const first = sanRecords(board)[0]!;
  const descriptor = describeMove(board, first.id);
  assert.deepEqual(historyKey([descriptor]), [first.san]);
});

test('historyKey: a malformed descriptor history returns null, never throws', () => {
  assert.equal(historyKey([{ bogus: true }]), null);
  assert.equal(historyKey(['not-an-array-check-but-a-string', { bogus: true }]), null);
});

test('replayHistory: mixed SAN and descriptor entries resolve against the position reached so far', () => {
  const board = boardFromFen(ADVANCED_POSITION);
  const sanEntry = sanRecords(board)[0]!.san;
  const afterFirst = replayHistory([sanEntry]).game.board;
  const secondRecord = sanRecords(afterFirst)[0]!;
  const descriptorEntry = describeMove(afterFirst, secondRecord.id);

  const mixed = replayHistory([sanEntry, descriptorEntry]);
  const allSan = replayHistory([sanEntry, secondRecord.san]);
  assert.equal(canonicalFen(boardToFen(mixed.game.board)), canonicalFen(boardToFen(allSan.game.board)));
  assert.deepEqual(mixed.sanHistory, allSan.sanHistory);
});

test('rootGame: a malformed descriptor entry stamps mismatch, never throws', () => {
  const { stamp, game } = rootGame(ADVANCED_POSITION, [{ kind: 'bogus' }]);
  assert.equal(stamp, HISTORY_MISMATCH);
  assert.equal(game.turns, 0);
});

function emptyBoard(): Board {
  const b = new Board();
  b.mode = MODE_ADVANCED;
  return b;
}

function place(board: Board, rank: number, file: number, ti: number, color: number, tier: number): void {
  board.cells[cellIndex(rank - 1, file - 1, tier - 1)] = makeCode(ti, color);
}

test('replayHistory/rootGame under ligungi: a descriptor move outside the legal set is rejected, same as SAN', () => {
  // Same self-exposure shape as ligungi_filter.test.ts's L5: a white general
  // at (3,1) blocks a black general's file check on white's marshal at
  // (1,1); stepping that general OFF the file is pseudo-legal under
  // 'komugi' movegen but outside legalSet(board, 'ligungi') (rule 1).
  const board = emptyBoard();
  board.turn = WHITE;
  place(board, 1, 1, MARSHAL, WHITE, 1);
  place(board, 9, 9, MARSHAL, BLACK, 1);
  place(board, 5, 1, GENERAL, BLACK, 1);
  place(board, 3, 1, GENERAL, WHITE, 1);
  const startFen = boardToFen(board);

  const offRay = generateRecords(board, 'komugi').find(
    (r) => r.struct[3] === 3 - 1 && r.struct[4] === 1 - 1 && r.struct[7] !== 1 - 1,
  );
  assert.ok(offRay !== undefined, 'expected an off-ray move for the blocking general');
  const descriptor = describeMove(board, offRay!.id);

  assert.throws(
    () => gameFromHistory([descriptor], startFen, 'ligungi'),
    /outside ligungi legal set/,
  );

  const { stamp } = rootGame(startFen, [descriptor], startFen, 'ligungi');
  assert.equal(stamp, HISTORY_MISMATCH);
});

// Hand-built position 2 (design section 6): betrayal selections 31332 (sel
// 1) and 31413 (sel 2) both print the SAN `謀(9-5-2)(8-6-3)返兵`, and
// resolveSan always resolves that string to the FIRST match, 31332
// (san.ts:74-80). A descriptor names either id exactly.
const HAND_BUILT_POSITION_2 = '4m4/9/9/9/9/9/9/3|d:d|5/4|D:T|4 D2/- w 3 - 20';
const SEL_1_ID = 31332;
const SEL_2_ID = 31413;

test('SAN is not injective: a descriptor history naming sel 2 (31413) gets present; its SAN history gets mismatch', () => {
  const board = boardFromFen(HAND_BUILT_POSITION_2);
  const sel1San = sanRecords(board).find((r) => r.id === SEL_1_ID)?.san;
  const sel2San = sanRecords(board).find((r) => r.id === SEL_2_ID)?.san;
  assert.ok(sel1San !== undefined && sel2San !== undefined, 'expected both betrayal selections to be legal here');
  assert.equal(sel1San, sel2San, 'sel 1 and sel 2 must share one SAN string (section 3)');
  assert.equal(resolveSan(board, sel1San!), SEL_1_ID, 'resolveSan must resolve to the FIRST match, sel 1');

  const descriptor = describeMove(board, SEL_2_ID);
  const sel2Replay = replayHistory([descriptor], HAND_BUILT_POSITION_2);
  const sel2Fen = boardToFen(sel2Replay.game.board);

  const descriptorResult = rootGame(sel2Fen, [descriptor], HAND_BUILT_POSITION_2);
  assert.equal(descriptorResult.stamp, HISTORY_PRESENT, 'the descriptor names sel 2 exactly');

  const sanResult = rootGame(sel2Fen, sel2Replay.sanHistory, HAND_BUILT_POSITION_2);
  assert.equal(sanResult.stamp, HISTORY_MISMATCH, 'the SAN history resolves to sel 1 instead, landing elsewhere');
});

// ---- KOM-1203 unit 4: game.ts/root.ts knob reads --------------------------

test('knob isolation (repetitionKey only): gameFromHistory keys under FULL, not PLACEMENT', () => {
  const san = firstLegalSan();
  const REPETITION_FULL_ONLY: RulesConfig = { ...RULES_PRESETS.komugi, repetitionKey: 'full' };

  const placementGame = gameFromHistory([san], undefined, RULES_PRESETS.komugi);
  assert.equal(placementGame.rule, RepetitionRule.PLACEMENT);

  const fullGame = gameFromHistory([san], undefined, REPETITION_FULL_ONLY);
  assert.equal(fullGame.rule, RepetitionRule.FULL);
});

// Same shape as ligungi_movegen.test.ts's buildTier2ArcherBoard geometry: a
// tier-2 archer whose (lateral 2, forward 3) target is reachable under
// komugi movegen but blocked, under ligungi, only by the (2,2) elbow. Unlike
// that function, every tower here is filled from tier 1 up (a real FEN,
// round-tripped through boardFromFen by replayEntries, cannot represent a
// piece floating above empty tiers the way a bare in-memory Board can).
function buildTier2ArcherBoard(): Board {
  const board = emptyBoard();
  board.turn = WHITE;
  place(board, 9, 5, SOLDIER, WHITE, 1); // filler beneath the archer
  place(board, 9, 5, ARCHER, WHITE, 2); // origin, tier 2
  place(board, 7, 7, SOLDIER, BLACK, 1); // filler
  place(board, 7, 7, SOLDIER, BLACK, 2); // filler
  place(board, 7, 7, SOLDIER, BLACK, 3); // elbow (lateral2,forward2), taller than tier 2
  place(board, 1, 1, MARSHAL, WHITE, 1);
  place(board, 9, 9, MARSHAL, BLACK, 1);
  return board;
}

test('replayHistory/rootGame under a RulesConfig literal (archerElbowBlocking only): rejected the same as under ligungi (F2(a))', () => {
  // Same shape as the self-exposure case above (L180-206), but the knob
  // flipped is archerElbowBlocking alone, from a komugi-based literal, not
  // the whole 'ligungi' profile string.
  const ARCHER_ELBOW_ONLY: RulesConfig = { ...RULES_PRESETS.komugi, archerElbowBlocking: true };

  const board = buildTier2ArcherBoard();
  const startFen = boardToFen(board);

  const jump = generateRecords(board, 'komugi').find(
    (r) => r.struct[6] === 6 - 1 && r.struct[7] === 7 - 1,
  );
  assert.ok(jump !== undefined, 'expected the archer jump to (6,7) to be pseudo-legal under komugi movegen');
  const descriptor = describeMove(board, jump!.id);

  assert.throws(
    () => gameFromHistory([descriptor], startFen, ARCHER_ELBOW_ONLY),
    /outside ligungi legal set/,
  );

  const { stamp } = rootGame(startFen, [descriptor], startFen, ARCHER_ELBOW_ONLY);
  assert.equal(stamp, HISTORY_MISMATCH);
});

const PRE_STALEMATE_FEN = '5g1|d:s|M/7d|d:s|/9/9/9/9/9/9/m8 -/- b 3 - 1';

test('knob isolation (stalemateResult only): a stalemate position scores a loss, not a draw', () => {
  const STALEMATE_LOSS_ONLY: RulesConfig = { ...RULES_PRESETS.komugi, stalemateResult: 'loss-for-mover' };

  // Black's only winning move under rule (4) leaves White with no legal
  // move and not in check (ligungi_terminal.test.ts's own fixture position).
  function blackPlaysItsOnlyWinningMove(rules: RulesConfig): Game {
    const g = new Game(boardFromFen(PRE_STALEMATE_FEN), RepetitionRule.FULL, 1, rules);
    const rec = sanRecords(g.board).find((r) => r.san === '兵(2-2-1)(3-2-1)');
    assert.ok(rec !== undefined, 'expected the stalemating move to be found by SAN');
    g.play(rec!.id);
    return g;
  }

  const komugiAfter = blackPlaysItsOnlyWinningMove(RULES_PRESETS.komugi);
  assert.deepEqual(komugiAfter.winState(), [0, 0, 1], "komugi (stalemateResult: 'draw'): a stalemate is a draw");

  const flippedAfter = blackPlaysItsOnlyWinningMove(STALEMATE_LOSS_ONLY);
  assert.deepEqual(
    flippedAfter.winState(),
    [0, 1, 0],
    "stalemateResult: 'loss-for-mover' must score it a loss for White, the side to move",
  );
});

function buildNonTerminalBoard(): Board {
  const board = emptyBoard();
  board.turn = WHITE;
  place(board, 1, 1, MARSHAL, WHITE, 1);
  place(board, 9, 9, MARSHAL, BLACK, 1);
  place(board, 5, 5, SOLDIER, WHITE, 1);
  place(board, 5, 4, SOLDIER, BLACK, 1);
  return board;
}

test('knob isolation (plyCap only): the ply cap fires under komugi\'s 512 but not under plyCap: null', () => {
  const PLY_CAP_NONE: RulesConfig = { ...RULES_PRESETS.komugi, plyCap: null };

  const capped = new Game(buildNonTerminalBoard(), RepetitionRule.FULL, 1, RULES_PRESETS.komugi);
  capped.turns = 512;
  assert.deepEqual(capped.winState(), [0, 0, 1], 'komugi: turns >= 512 is a draw by the ply cap');

  const uncapped = new Game(buildNonTerminalBoard(), RepetitionRule.FULL, 1, PLY_CAP_NONE);
  uncapped.turns = 512;
  assert.deepEqual(uncapped.winState(), [0, 0, 0], 'plyCap: null must never trigger the ply-cap draw');
});
