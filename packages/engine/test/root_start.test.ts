// KOM-1104: root.ts/protocol.ts/worker.ts/ponder.ts grow an optional
// `start` FEN that `history` replays from, instead of always
// ADVANCED_POSITION. No fixture needed -- every case here is a hand-built
// FEN/history pair, the same style test/root.test.ts already uses.

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { cellIndex } from '../src/board.ts';
import { ADVANCED_POSITION, BEGINNER_POSITION, MODE_BEGINNER, MT_ROUTE, MT_TSUKE } from '../src/constants.ts';
import { boardFromFen, boardToFen, canonicalFen } from '../src/fen.ts';
import { Game } from '../src/game.ts';
import { ponderKey } from '../src/ponder.ts';
import { RepetitionRule } from '../src/repetition.ts';
import { resolveSan, sanRecords } from '../src/san.ts';
import type { SanRecord } from '../src/san.ts';
import { HISTORY_MISMATCH, HISTORY_PRESENT, gameFromHistory, rootGame } from '../src/root.ts';

// The advanced start's own FEN still has both sides drafting ("wb"): no
// piece is on the board yet, so there is no route move to shuffle until the
// draft finishes. Plays the FIRST enumerated legal move, repeatedly, until
// the draft is over -- turns ADVANCED_POSITION into a `start` FEN already in
// ordinary play, the same shape BEGINNER_POSITION's own FEN already is
// (draft field "-"). Whatever mode/hands the played-out draft leaves is all
// this file needs: a legal post-draft `start` at mode 3, not literally the
// pre-draft constant.
function postDraftStart(fen: string): string {
  const game = new Game(boardFromFen(fen), RepetitionRule.PLACEMENT);
  while (game.board.draftW !== 0 || game.board.draftB !== 0) {
    const records = sanRecords(game.board);
    assert.ok(records.length > 0, `no legal draft move from ${boardToFen(game.board)}`);
    game.play(records[0].id);
  }
  return boardToFen(game.board);
}

function routeMoves(board: ReturnType<typeof boardFromFen>): SanRecord[] {
  return sanRecords(board).filter((rec) => rec.struct[0] === MT_ROUTE);
}

// A record whose move lands exactly where `rec` started, same square/tier
// both ways -- the one-ply "undo" of `rec`, found by searching the NEXT
// position's own legal moves rather than assuming any particular piece can
// always step backward.
function reverseOf(records: SanRecord[], rec: SanRecord): SanRecord | undefined {
  return records.find(
    (r) =>
      r.struct[0] === MT_ROUTE &&
      r.struct[3] === rec.struct[6] &&
      r.struct[4] === rec.struct[7] &&
      r.struct[5] === rec.struct[8] &&
      r.struct[6] === rec.struct[3] &&
      r.struct[7] === rec.struct[4] &&
      r.struct[8] === rec.struct[5],
  );
}

function cellsEqual(a: Uint8Array, b: Uint8Array): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

// Finds a 4-ply "white moves a piece out, black moves a piece out, white
// undoes its own move, black undoes its own move" cycle from `startFen`
// (already past any draft): a SAN loop whose end position is cell-for-cell
// identical to its start, so replaying it under PLACEMENT (repetition.ts's
// cells-only key) repeats the same key once per cycle, regardless of turn/
// move-number bookkeeping. Generic over `start` -- it never assumes a
// particular piece type is the one that can shuffle.
function findShuffleCycle(startFen: string): string[] {
  const start = new Game(boardFromFen(startFen), RepetitionRule.PLACEMENT);
  for (const whiteOut of routeMoves(start.board)) {
    const afterWhiteOut = start.clone();
    afterWhiteOut.play(whiteOut.id);
    for (const blackOut of routeMoves(afterWhiteOut.board)) {
      const afterBlackOut = afterWhiteOut.clone();
      afterBlackOut.play(blackOut.id);
      const whiteBack = reverseOf(routeMoves(afterBlackOut.board), whiteOut);
      if (whiteBack === undefined) continue;
      const afterWhiteBack = afterBlackOut.clone();
      afterWhiteBack.play(whiteBack.id);
      const blackBack = reverseOf(routeMoves(afterWhiteBack.board), blackOut);
      if (blackBack === undefined) continue;
      const afterBlackBack = afterWhiteBack.clone();
      afterBlackBack.play(blackBack.id);
      if (cellsEqual(afterBlackBack.board.cells, start.board.cells)) {
        return [whiteOut.san, blackOut.san, whiteBack.san, blackBack.san];
      }
    }
  }
  throw new Error(`no reversible 4-ply shuffle cycle found from ${startFen}`);
}

// Replays `cycle` three times from `startFen` (reaching the fourth
// occurrence of the starting PLACEMENT key, repetition.ts's own
// REPETITION_THRESHOLD) and checks the repetition count climbs 2, 3, 4 and
// that a draw fires exactly once it does -- both directly (Game.play) and
// through the production path (rootGame with `start`).
function assertCycleDraws(startFen: string, cycle: readonly string[]): void {
  const game = new Game(boardFromFen(startFen), RepetitionRule.PLACEMENT);
  const history: string[] = [];
  const counts: number[] = [];
  for (let rep = 0; rep < 3; rep++) {
    for (const san of cycle) {
      const action = resolveSan(game.board, san);
      game.play(action);
      history.push(san);
    }
    counts.push(game.repetitionCount());
  }
  assert.deepEqual(counts, [2, 3, 4], 'repetition count should climb by one per cycle, to the fourth occurrence');
  assert.equal(game.winState()[2], 1, 'the fourth occurrence should be a draw');

  const finalFen = boardToFen(game.board);
  const result = rootGame(finalFen, history, startFen);
  assert.equal(result.stamp, HISTORY_PRESENT);
  assert.equal(result.game.repetitionCount(), 4);
  assert.equal(result.game.winState()[2], 1);
}

test('Beginner start: a looping history increments the repetition count and draws, via rootGame', () => {
  const cycle = findShuffleCycle(BEGINNER_POSITION);
  assertCycleDraws(BEGINNER_POSITION, cycle);
});

test('Advanced (post-draft) start: the same mechanism, for parity with the Beginner-start case above', () => {
  const start = postDraftStart(ADVANCED_POSITION);
  const cycle = findShuffleCycle(start);
  assertCycleDraws(start, cycle);
});

// KOM-1105 (KOM-1104 verifier finding: this test was vacuous -- the fresh
// Beginner start has no towers at all yet, so "no tier-3 tsuke" held
// trivially regardless of whether mode 1's cap was enforced). Fixed by
// actually building a tier-2 tower first (via a real history replayed
// through the production `rootGame` path, `start` included) and THEN
// checking no tsuke onto it reaches tier 3.
test('Beginner start: no tier-3 tsuke is ever generated onto an actual tier-2 tower (mode 1 caps the tower at 2)', () => {
  const board = boardFromFen(BEGINNER_POSITION);
  assert.equal(board.mode, MODE_BEGINNER);

  const firstTowerMove = sanRecords(board).find((rec) => rec.struct[0] === MT_TSUKE && rec.struct[8] === 2);
  assert.ok(firstTowerMove !== undefined, 'expected at least one legal tier-2 tsuke from the Beginner start');
  if (firstTowerMove === undefined) return;
  const history = [firstTowerMove.san];

  const { game, stamp } = rootGame(boardToFen(gameFromHistory(history, BEGINNER_POSITION).board), history, BEGINNER_POSITION);
  assert.equal(stamp, HISTORY_PRESENT);

  // The tower actually has height 2 at the tsuke's destination square now --
  // this is what makes the assertion below falsifiable, not vacuous.
  // struct[6]/struct[7] are already 0-indexed (rank-1/file-1, the same
  // convention cellIndex itself takes -- movegen_board.ts's `tr`/`tf`), so
  // this must NOT subtract 1 again.
  const [, , , , , , toRank, toFile, toTier] = firstTowerMove.struct;
  assert.equal(toTier, 2);
  assert.notEqual(
    game.board.cells[cellIndex(toRank, toFile, toTier - 1)],
    0,
    'expected a real piece on the second tier of the tower',
  );

  // struct's tier fields are 1-indexed (movegen_board.ts's own `maxTier`
  // convention: mode 1's cap of 2 means tiers 1 and 2 only, never 3).
  const tier3Tsuke = sanRecords(game.board).filter(
    (rec) => rec.struct[0] === MT_TSUKE && rec.struct[6] === toRank && rec.struct[7] === toFile && rec.struct[8] === 3,
  );
  assert.deepEqual(tier3Tsuke, [], 'a tier-3 tsuke onto the real tower should never be legal under mode 1');
});

test('Beginner start: the replay with stamp present has the right hands', () => {
  const { game, stamp } = rootGame(BEGINNER_POSITION, [], BEGINNER_POSITION);
  assert.equal(stamp, HISTORY_PRESENT);
  assert.equal(game.board.mode, MODE_BEGINNER);
  assert.deepEqual(Array.from(game.board.hand), Array.from(boardFromFen(BEGINNER_POSITION).hand));
  assert.notDeepEqual(Array.from(game.board.hand), Array.from(boardFromFen(ADVANCED_POSITION).hand));
});

test('no start: gameFromHistory/rootGame behave exactly as before (ADVANCED_POSITION default)', () => {
  const draftMove = sanRecords(boardFromFen(ADVANCED_POSITION))[0].san;
  const withDefault = gameFromHistory([draftMove]);
  const withExplicitAdvanced = gameFromHistory([draftMove], ADVANCED_POSITION);
  assert.equal(boardToFen(withDefault.board), boardToFen(withExplicitAdvanced.board));

  const reachedFen = boardToFen(withDefault.board);
  const a = rootGame(reachedFen, [draftMove]);
  const b = rootGame(reachedFen, [draftMove], undefined);
  assert.equal(a.stamp, HISTORY_PRESENT);
  assert.equal(a.stamp, b.stamp);
  assert.equal(boardToFen(a.game.board), boardToFen(b.game.board));
  assert.equal(a.game.turns, b.game.turns);
});

// KOM-1105 (KOM-1104 verifier finding: this test was vacuous -- it passed
// `fen` as the UNCHANGED ADVANCED_POSITION, which a replay of a one-move
// history could never reproduce regardless of whether `start: 'not-a-fen'`
// was correctly rejected or silently defaulted to ADVANCED_POSITION, so it
// could not actually distinguish the two). Fixed by making `fen` the exact
// board a silent default WOULD have produced: `draftMove` replayed from
// ADVANCED_POSITION (the default `gameFromHistory`/`rootGame` fall back to).
// A buggy silent-default implementation would then replay cleanly and
// report `present`; only correctly rejecting the invalid `start` yields
// `mismatch` here.
test('an invalid start falls back exactly like a bad history: stamp mismatch (not a silent default)', () => {
  const draftMove = sanRecords(boardFromFen(ADVANCED_POSITION))[0].san;
  const fenFromDefaultStart = boardToFen(gameFromHistory([draftMove]).board);
  const { game, stamp } = rootGame(fenFromDefaultStart, [draftMove], 'not-a-fen');
  assert.equal(stamp, HISTORY_MISMATCH);
  assert.equal(game.rule, RepetitionRule.FULL);
  assert.equal(game.turns, 0);
  assert.equal(canonicalFen(boardToFen(game.board)), canonicalFen(fenFromDefaultStart));
});

test('C2: the ponder key differs when only `start` differs, and an omitted start normalizes to ADVANCED_POSITION', () => {
  const common = {
    history: ['m1', 'm2'],
    fen: 'shared-fen',
    sims: 16,
    batch: 4,
    configIdentity: 'id',
    backend: null,
    modelSha256: null,
    maxMs: null,
    mainTableEpoch: 0,
  };
  const omitted = ponderKey(common);
  const explicitAdvanced = ponderKey({ ...common, start: ADVANCED_POSITION });
  const explicitBeginner = ponderKey({ ...common, start: BEGINNER_POSITION });
  assert.equal(omitted, explicitAdvanced);
  assert.notEqual(explicitAdvanced, explicitBeginner);
});
