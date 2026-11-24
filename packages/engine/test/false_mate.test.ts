// KOM-1110/KOM-1124: port of the KOM-1109 false-mate fix (capturing the
// opposing marshal is always an escape) to terminal.ts's hasEscapingMove,
// PROFILE DEPENDENT: 'komugi' and 'ligungi' both get it -- 'komugi' matches
// the fixed cyengine o12-for-o12 directly, and 'ligungi' gets the same
// escape per KOM-1106's UPDATE 2026-11-13 (87cd3a3): capturing the
// opponent's marshal is ALWAYS legal under ligungi and wins, so a side
// whose only reply captures the enemy marshal is not mated there either.
// 'gungijs' alone keeps the pre-fix gungi.js 3.4.11 behaviour (a false mate
// there still scores as mate, until jwyce/gungi.js#53 lands).

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { Board, cellIndex, colorOf, typeIndex } from '../src/board.ts';
import { applyMove } from '../src/apply.ts';
import { MARSHAL } from '../src/constants.ts';
import { boardFromFen } from '../src/fen.ts';
import { Game } from '../src/game.ts';
import { RepetitionRule } from '../src/repetition.ts';
import { sanRecords, type SanRecord } from '../src/san.ts';
import { isCheckmate } from '../src/terminal.ts';
import { gameForPosition, loadOracleFixture, positionRecords, verdictOf, verdictsDisagree } from './oracle_helpers.ts';

// ---- shared helper: every pseudo-legal reply that captures the opponent's
// marshal (mirrors tests/engine/test_false_mate.py's own
// _marshal_capturing_replies, reconstructed from public board state since
// isMarshalCaptured/the scratch board inside terminal.ts are private). -----

function marshalCountOf(board: Board, color: number): number {
  let count = 0;
  for (let r = 0; r < 9; r++) {
    for (let f = 0; f < 9; f++) {
      for (let t = 0; t < 3; t++) {
        const code = board.cells[cellIndex(r, f, t)];
        if (code !== 0 && typeIndex(code) === MARSHAL && colorOf(code) === color) count++;
      }
    }
  }
  return count;
}

function marshalCapturingReplies(board: Board): SanRecord[] {
  const opponent = 1 - board.turn;
  const before = marshalCountOf(board, opponent);
  const out: SanRecord[] = [];
  for (const record of sanRecords(board)) {
    const copy = board.copy();
    applyMove(copy, record.struct);
    if (marshalCountOf(copy, opponent) < before) out.push(record);
  }
  return out;
}

// ---- (I) the oracle fixture's false_checkmate class, all 30 records -------
// (KOM-1191: 35 -> 30 once the bot-match-sourced records were filtered out
// of the fixture.)

const fixture = loadOracleFixture();
const falseCheckmateRecords = positionRecords(fixture).filter((r) => r.classes.includes('false_checkmate'));

test('ligungi_oracle.json: has false_checkmate position records', () => {
  assert.equal(falseCheckmateRecords.length, 30);
});

test('komugi: every false_checkmate record is NOT terminal, and a marshal-capturing reply is legal', () => {
  for (const record of falseCheckmateRecords) {
    const game = gameForPosition(record, 'komugi');
    assert.equal(verdictOf(game), null, `${record.id}: expected not-terminal under komugi`);
    assert.equal(
      verdictsDisagree(verdictOf(game), record.ligungi.terminal),
      false,
      `${record.id}: komugi's verdict must agree with record.ligungi.terminal`,
    );

    const replies = marshalCapturingReplies(game.board);
    assert.ok(replies.length > 0, `${record.id}: expected a marshal-capturing reply`);

    const mask = game.validMoves();
    for (const reply of replies) {
      assert.equal(mask[reply.id], 1, `${record.id}: ${reply.san} must be legal under komugi`);
    }
  }
});

test('komugi: playing the marshal-capturing reply ends the game for the capturer', () => {
  for (const record of falseCheckmateRecords) {
    const game = gameForPosition(record, 'komugi');
    const [reply] = marshalCapturingReplies(game.board);
    assert.ok(reply !== undefined, `${record.id}: expected a marshal-capturing reply`);

    const capturer = game.board.turn;
    game.play(reply.id);

    const [whiteWins, blackWins, draw] = game.winState();
    assert.equal(draw, 0, `${record.id}: capture must not be scored a draw`);
    assert.equal([whiteWins, blackWins][capturer], 1, `${record.id}: the capturer must win`);
  }
});

test('gungijs: every false_checkmate record stays (wrongly) checkmate -- the documented, deliberate divergence', () => {
  for (const record of falseCheckmateRecords) {
    const game = gameForPosition(record, 'gungijs');
    assert.deepEqual(verdictOf(game), record.komugi.terminal, `${record.id}: gungijs must match the old-engine verdict`);
  }
});

// KOM-1124 (F2): Game.validMoves() must agree with the (wrongly) terminal
// gungijs verdict above -- an empty mask, not merely a non-empty one the
// search layer would flag as an inconsistent terminal state (search.ts's own
// "terminal state that reports N legal actions" guard).
test('gungijs: Game.validMoves() is empty at every false_checkmate record', () => {
  for (const record of falseCheckmateRecords) {
    const game = gameForPosition(record, 'gungijs');
    const mask = game.validMoves();
    assert.ok(mask.every((bit) => bit === 0), `${record.id}: expected an all-zero mask under gungijs`);

    const [whiteWins, blackWins, draw] = game.winState();
    assert.ok(whiteWins || blackWins || draw, `${record.id}: expected gungijs to score this position terminal`);
  }
});

// ---- (II) the KOM-1108/KOM-1109 worked example, ported verbatim from
// tests/engine/test_false_mate.py (same FEN, same two SANs). -----------------

const FALSE_MATE_FEN =
  'r1|s:n|m2r|s:g|1/4k|a:J|j2/|n:K:F|1ww1f3/1nt6/2c6/a8/1|D:D:G||S:d|6' + '/1|N:N||S:M|6/9 I1R1F1C1A1T1/d2 w 3 - 33';
const MATING_LOOKING_MOVE = '小(2-4-2)(2-5-2)付';
const CAPTURING_REPLY = '兵(7-7-2)取(8-7-1)#';

function resolveAndPlay(game: Game, san: string): void {
  const normalized = san.replace(/[=#]+$/, '');
  for (const record of sanRecords(game.board)) {
    if (record.san === san || record.san === normalized) {
      game.play(record.id);
      return;
    }
  }
  throw new Error(`move not found: ${san}`);
}

test('KOM-1108 worked example: the mating-looking move is not checkmate under komugi, and the capture is legal', () => {
  const game = new Game(boardFromFen(FALSE_MATE_FEN), RepetitionRule.FULL, 1, 'komugi');
  resolveAndPlay(game, MATING_LOOKING_MOVE);

  assert.equal(isCheckmate(game.board, 'komugi'), false);

  const replies = marshalCapturingReplies(game.board);
  assert.ok(
    replies.some((r) => r.san === CAPTURING_REPLY || r.san === CAPTURING_REPLY.replace(/[=#]+$/, '')),
    `expected ${CAPTURING_REPLY} among the marshal-capturing replies, got ${JSON.stringify(replies.map((r) => r.san))}`,
  );
});

test('KOM-1108 worked example: playing the capture ends the game for the capturer (komugi)', () => {
  const game = new Game(boardFromFen(FALSE_MATE_FEN), RepetitionRule.FULL, 1, 'komugi');
  resolveAndPlay(game, MATING_LOOKING_MOVE);
  const capturer = game.board.turn;
  resolveAndPlay(game, CAPTURING_REPLY);

  const [whiteWins, blackWins, draw] = game.winState();
  assert.equal(draw, 0);
  assert.equal([whiteWins, blackWins][capturer], 1);
});

test('KOM-1108 worked example: gungijs still (wrongly) claims checkmate -- documented divergence', () => {
  const game = new Game(boardFromFen(FALSE_MATE_FEN), RepetitionRule.FULL, 1, 'gungijs');
  resolveAndPlay(game, MATING_LOOKING_MOVE);
  assert.equal(isCheckmate(game.board, 'gungijs'), true);
});

// ---- (III) a genuine checkmate (KOM-142's MATE_FEN) stays mate under all
// three profiles -- no marshal-capturing reply exists, so the fix never
// fires regardless of `rules`. ----------------------------------------------

const GENUINE_CHECKMATE_FEN = '8m/7D1/6I1G/9/9/9/9/9/4M4 -/- b 1 - 1';

test('a genuine checkmate has no marshal-capturing reply and stays mate under komugi, gungijs and ligungi', () => {
  const board = boardFromFen(GENUINE_CHECKMATE_FEN);
  assert.deepEqual(marshalCapturingReplies(board), []);

  assert.equal(isCheckmate(board, 'komugi'), true);
  assert.equal(isCheckmate(board, 'gungijs'), true);
  assert.equal(isCheckmate(board, 'ligungi'), true);
});
