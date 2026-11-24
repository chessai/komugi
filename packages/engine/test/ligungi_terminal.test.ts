// KOM-1107 (terminal scoring and repetition under 'ligungi'). CLEAN ROOM:
// rule text comes from tickets/KOM-1107.md's prose only, never from ligungi
// source.
//
// "Red without its fix" verification (per the worker brief): every targeted
// test below was additionally run against a disposable copy of this tree
// under a tempfile dir, with web/engine/src/game.ts, worker.ts, root.ts and
// web/engine/test/oracle_helpers.ts restored to their pre-fix (git HEAD)
// contents. Not every test goes red on that copy: the rule (6) regression
// test was already true before this ticket (insufficient material's
// adjacency exception is not profile-dependent, nothing to make red); the
// "Beginner start" smoke test is plumbing only, not a rule reproduction; the
// fullKey/placementKey tests exercise pre-existing key functions untouched
// by this ticket's production edits; and the KOM-1107 adversarial review
// (docs/reviews/2026-11-14-kom1107-ts-terminal-review.md, F1-F4) found that
// the worker-512 and legalSet-mismatch tests below also stayed green
// without their corresponding fix -- see the F1-F4 tests, which close that
// gap and are red on the review's mutants (m2, m4, m5, m6).

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';

import { Board, cellIndex, makeCode } from '../src/board.ts';
import {
  ADVANCED_POSITION,
  ARCHER,
  BEGINNER_POSITION,
  BLACK,
  CANNON,
  GENERAL,
  MARSHAL,
  MODE_ADVANCED,
  MT_CAPTURE,
  SOLDIER,
  SPY,
  TACTICIAN,
  WHITE,
} from '../src/constants.ts';
import { boardFromFen, boardToFen } from '../src/fen.ts';
import { Game, MaxTurns } from '../src/game.ts';
import { generateRecords } from '../src/movegen.ts';
import { canonicalMove, fullKey, legalSet, placementKey, RepetitionRule } from '../src/repetition.ts';
import type { RulesProfile } from '../src/repetition.ts';
import { RULES_PRESETS } from '../src/rules_config.ts';
import type { RulesConfig } from '../src/rules_config.ts';
import { gameFromHistory, HISTORY_MISMATCH, HISTORY_PRESENT, rootGame } from '../src/root.ts';
import { sanRecords } from '../src/san.ts';
import { MatchGameState } from '../src/search/state.ts';
import { inCheck, isCheckmate, isInsufficientMaterial } from '../src/terminal.ts';
import { WorkerEngine } from '../src/engine.ts';
import type { WorkerRequest } from '../src/protocol.ts';
import { loadOracleFixture, replaySequence, sequenceRecords, sequenceStart } from './oracle_helpers.ts';

const PROFILES: readonly RulesProfile[] = ['komugi', 'gungijs', 'ligungi'];

// 1-based (rank, file), 1-based tier -- matches movegen_board.ts's/
// ligungi_movegen.test.ts's own coordinate convention.
function place(board: Board, rank: number, file: number, ti: number, color: number, tier: number): void {
  board.cells[cellIndex(rank - 1, file - 1, tier - 1)] = makeCode(ti, color);
}

// ---- rule (4): no legal move while not in check is a LOSS under ligungi ---

// A synthetic no-legal-move, not-in-check position, built by FEN (empty
// hands, every piece blocked): White's marshal sits in the (1,1) corner, so
// only 3 of its 8 single-step directions are even on the board -- (1,2),
// (2,1), (2,2) -- and each of those is topped by a height-2 tower (any
// piece atop a tower taller than the marshal's own tier-1 blocks the move
// entirely, per terminal.ts's squareInAvailableSquares, regardless of
// color). White has no other piece and an empty hand, so this is White's
// only possible move-source, pseudo-legal or otherwise -- zero legal moves.
// The top of each tower is chosen (SPY at (1,2)/(2,1), SOLDIER at (2,2)) so
// that none of them attacks (1,1): SPY's attack table is inactive to its
// own West/North, SOLDIER's is inactive to its own North-West -- so White's
// marshal is NOT in check (a stalemate, not a checkmate).
function buildNoLegalMoveBoard(): Board {
  const board = new Board();
  board.mode = MODE_ADVANCED;
  board.turn = WHITE;

  place(board, 1, 1, MARSHAL, WHITE, 1);
  place(board, 9, 9, MARSHAL, BLACK, 1);

  place(board, 1, 2, SOLDIER, BLACK, 1);
  place(board, 1, 2, SPY, BLACK, 2);

  place(board, 2, 1, SOLDIER, BLACK, 1);
  place(board, 2, 1, SPY, BLACK, 2);

  place(board, 2, 2, SOLDIER, BLACK, 1);
  place(board, 2, 2, SOLDIER, BLACK, 2);

  return board;
}

function gameFromSyntheticFen(board: Board, rules: RulesProfile): Game {
  // "built by FEN": round-trip the programmatically-built board through
  // boardToFen/boardFromFen, exactly as a real request's `fen` would arrive.
  const fen = boardToFen(board);
  return new Game(boardFromFen(fen), RepetitionRule.PLACEMENT, 1, rules);
}

test('no-legal-move synthetic position: White has zero legal moves and is not in check, under every profile', () => {
  for (const rules of PROFILES) {
    const game = gameFromSyntheticFen(buildNoLegalMoveBoard(), rules);
    const mask = game.validMoves();
    assert.ok(mask.every((bit) => bit === 0), `${rules}: expected no legal moves`);
  }
});

test('komugi/gungijs: no legal move while not in check is a DRAW (unchanged)', () => {
  for (const rules of ['komugi', 'gungijs'] as const) {
    const game = gameFromSyntheticFen(buildNoLegalMoveBoard(), rules);
    assert.deepEqual(game.winState(), [0, 0, 1], rules);
  }
});

test('ligungi: no legal move while not in check is a LOSS for the side to move (rule 4)', () => {
  const game = gameFromSyntheticFen(buildNoLegalMoveBoard(), 'ligungi');
  // White (board.turn) has no legal move -- White loses, Black wins.
  assert.deepEqual(game.winState(), [0, 1, 0]);
});

// F4 (review 2026-11-14, mutant m6: search/state.ts terminalWdl scores a
// ligungi stalemate as a draw). The tests above only check Game.winState();
// the SCOPE NOTE also requires rule (4) proven at the search's own terminal
// values (the root-win scan and terminalWdl), which no test here reached.
// Black to move; 兵(2-2-1)(3-2-1) is Black's ONLY move that wins under
// ligungi, and it wins only by rule (4): it leaves White with no legal move
// and not in check (komugi scores the result a draw, so no Black move wins
// under komugi).
const PRE_STALEMATE_FEN = '5g1|d:s|M/7d|d:s|/9/9/9/9/9/9/m8 -/- b 3 - 1';
test('ligungi: search terminal values follow rule (4) (root-win scan + terminalWdl)', () => {
  const g = new Game(boardFromFen(PRE_STALEMATE_FEN), RepetitionRule.FULL, 1, 'ligungi');
  const rec = sanRecords(g.board).find((r) => r.san === '兵(2-2-1)(3-2-1)');
  assert.ok(rec !== undefined);
  const after = new MatchGameState(g.clone());
  after.play(rec!.id);
  assert.deepEqual(Array.from(after.terminalWdl()), [0, 0, 1], 'White (to move) loses under ligungi');

  const r = new WorkerEngine().handle({ id: 1, fen: PRE_STALEMATE_FEN, sims: 1, discovery: true, rules: 'ligungi' });
  assert.ok(!('err' in r));
  if (!('err' in r)) assert.equal(r.stats.tactical?.rootWitness, rec!.id, 'ligungi root-win scan');

  const k = new WorkerEngine().handle({ id: 1, fen: PRE_STALEMATE_FEN, sims: 1, discovery: true, rules: 'komugi' });
  assert.ok(!('err' in k));
  if (!('err' in k)) assert.equal(k.stats.tactical?.rootWitness, null, 'komugi: a stalemate is a draw, not a witness');
});

// ---- rule (5): repetition stays RepetitionRule.FULL under ligungi ---------

test('repetition key: FULL differs on hands alone; PLACEMENT does not (rule 5)', () => {
  const board1 = boardFromFen(ADVANCED_POSITION);
  const board2 = boardFromFen(ADVANCED_POSITION);
  board2.hand[SOLDIER * 2 + WHITE] += 1;

  assert.notDeepEqual(Array.from(fullKey(board1)), Array.from(fullKey(board2)));
  assert.deepEqual(Array.from(placementKey(board1)), Array.from(placementKey(board2)));
});

test('repetition key: FULL never collides a drafting position with the same cells post-draft (draft bits)', () => {
  const drafting = boardFromFen(ADVANCED_POSITION);
  assert.ok(drafting.draftW !== 0 || drafting.draftB !== 0, 'expected the advanced start to still be drafting');

  const postDraft = boardFromFen(ADVANCED_POSITION);
  postDraft.draftW = 0;
  postDraft.draftB = 0;

  assert.notDeepEqual(Array.from(fullKey(drafting)), Array.from(fullKey(postDraft)));
});

test('ligungi: gameFromHistory replays under RepetitionRule.FULL from a Beginner start (KOM-1104 plumbing)', () => {
  const history: string[] = [];
  let board = gameFromHistory(history, BEGINNER_POSITION, 'ligungi').board;
  for (let i = 0; i < 3; i++) {
    const records = sanRecords(board);
    assert.ok(records.length > 0, `no legal move at ply ${i}`);
    history.push(records[0].san);
    board = gameFromHistory(history, BEGINNER_POSITION, 'ligungi').board;
  }
  const fen = boardToFen(board);
  const { game, stamp } = rootGame(fen, history, BEGINNER_POSITION, 'ligungi');
  assert.equal(stamp, HISTORY_PRESENT);
  assert.equal(game.rules, 'ligungi');
});

// F1 (review 2026-11-14, mutant m4: root.ts gameFromHistory keeps PLACEMENT
// under ligungi). replaySequence (oracle_helpers.ts) repeats the FULL/
// PLACEMENT choice inside the test helper rather than calling production
// code, so every terminal-ply assertion built on it measures the helper,
// not rootGame/gameFromHistory. Here the PRODUCTION rootGame under
// 'ligungi', given a repetition_key record's history only up to komugi's
// terminal ply, must return a FULL-keyed game that is NOT yet terminal
// (ligungi's own end is later).
test('ligungi: rootGame keys the replay FULL (a repetition_key record is not terminal at the komugi ply) (rule 5)', () => {
  const recs = sequenceRecords(loadOracleFixture()).filter((r) => r.classes.length === 1 && r.classes[0] === 'repetition_key');
  assert.equal(recs.length, 31);
  for (const record of recs) {
    const k = replaySequence(record, 'komugi');
    assert.ok(k.terminalPly !== null && k.terminalPly < (record.ligungi.terminalPly as number));
    const { game, stamp } = rootGame(k.finalFen, k.sans, sequenceStart(record), 'ligungi');
    assert.equal(stamp, HISTORY_PRESENT, record.id);
    assert.equal(game.rule, RepetitionRule.FULL, `${record.id}: rule`);
    assert.deepEqual(game.winState(), [0, 0, 0], `${record.id}: must not be terminal at komugi ply ${k.terminalPly}`);
  }
});

// ---- rule (6) regression: adjacent lone marshals are not a draw -----------

test('insufficient material: adjacent lone marshals are not a draw, under every profile (regression)', () => {
  for (const rules of PROFILES) {
    const board = new Board();
    board.mode = MODE_ADVANCED;
    board.turn = WHITE;
    place(board, 5, 5, MARSHAL, WHITE, 1);
    place(board, 5, 6, MARSHAL, BLACK, 1); // adjacent

    assert.equal(isInsufficientMaterial(board), false, rules);
    const game = gameFromSyntheticFen(board, rules);
    const [, , draw] = game.winState();
    assert.equal(draw, 0, `${rules}: adjacent lone marshals must not be scored a draw`);
  }
});

// ---- rule (7): NO ply cap under ligungi ------------------------------------

test('komugi/gungijs: the ply cap still fires at turns >= 512 (regression guard)', () => {
  for (const rules of ['komugi', 'gungijs'] as const) {
    const game = new Game(boardFromFen(ADVANCED_POSITION), RepetitionRule.PLACEMENT, 1, rules);
    game.turns = MaxTurns;
    assert.deepEqual(game.winState(), [0, 0, 1], `${rules}: turns=512`);
  }
});

test('ligungi: no ply cap at turns 512 or 600 (rule 7)', () => {
  for (const turns of [512, 600]) {
    const game = new Game(boardFromFen(ADVANCED_POSITION), RepetitionRule.PLACEMENT, 1, 'ligungi');
    game.turns = turns;
    // The fresh advanced start is not otherwise terminal, so the only way
    // winState() could report anything here is a ply-cap short-circuit --
    // which must not happen under ligungi.
    assert.deepEqual(game.winState(), [0, 0, 0], `turns=${turns}`);
  }
});

// F3 (review 2026-11-14, mutant m2: game.ts ply cap re-enabled under
// ligungi). The previous version of this test picked sanRecords(...)[0]
// every ply, whose history is a 37-fold repetition terminal by ligungi's
// own FULL key from ply 112 on -- so the engine's reply ("root position is
// terminal ...: no move to search") passed the `err !== 'turn_cap'`
// assertion for the wrong reason, and stayed green with the game.ts cap
// re-enabled. Here the history is built from validMoves() (cap-independent:
// it never reads turns) while staying clear of repetition, so the root is
// genuinely non-terminal and must be searched at both 512 and 600 plies.
function nonRepeatingHistory(plies: number): { fen: string; history: string[] } {
  const game = new Game(boardFromFen(ADVANCED_POSITION), RepetitionRule.FULL, 1, 'ligungi');
  const history: string[] = [];
  for (let i = 0; i < plies; i++) {
    const mask = game.validMoves();
    let chosen: { id: number; san: string } | null = null;
    for (const rec of sanRecords(game.board)) {
      if (!mask[rec.id]) continue;
      game.play(rec.id);
      const ok = game.repetitionCount() <= 2 && game.validMoves().some((x) => x !== 0);
      game.undo();
      if (ok) {
        chosen = rec;
        break;
      }
    }
    assert.ok(chosen !== null, `stuck at ply ${i}`);
    history.push(chosen!.san);
    game.play(chosen!.id);
  }
  return { fen: boardToFen(game.board), history };
}

test('ligungi: a non-terminal history of 512 / 600 plies is searched (no cap, no refusal) (rule 7, worker level)', () => {
  const engine = new WorkerEngine();
  for (const plies of [MaxTurns, 600]) {
    const { fen, history } = nonRepeatingHistory(plies);
    const request: WorkerRequest = { id: 1, fen, history, sims: 4, rules: 'ligungi' };
    const reply = engine.handle(request);
    assert.ok(!('err' in reply), `plies=${plies}: ${JSON.stringify(reply).slice(0, 200)}`);
    if (!('err' in reply)) assert.equal(reply.stamp, 'present');
  }
});

// ---- a history move outside legalSet under ligungi gives a stamp mismatch -

// A tactician TSUKE onto a two-enemy stack (soldier at tier 1, cannon at
// tier 2), copied from ligungi_movegen.test.ts's buildBetrayalBoard: under
// the default (komugi-shape) movegen that resolveSan/sanRecords always use,
// every non-empty subset of the two enemies is offered (sel 1, 2, 3)
// alongside the plain stack (sel 0); ligungi's own legalSet only ever
// offers sel 0 and the FULL subset (sel 3) -- so the sel=1 (partial) SAN
// resolves and plays under the default movegen but is not in
// legalSet(board, 'ligungi').
function buildPartialBetrayalBoard(): Board {
  const board = new Board();
  board.mode = MODE_ADVANCED;
  board.turn = WHITE;
  place(board, 1, 1, MARSHAL, WHITE, 1);
  place(board, 9, 9, MARSHAL, BLACK, 1);
  place(board, 5, 5, SOLDIER, WHITE, 1);
  place(board, 5, 5, TACTICIAN, WHITE, 2);
  place(board, 4, 6, SOLDIER, BLACK, 1);
  place(board, 4, 6, CANNON, BLACK, 2);
  board.hand[SOLDIER * 2 + WHITE] = 2;
  board.hand[CANNON * 2 + WHITE] = 2;
  return board;
}

test('ligungi: a partial-betray history move is outside legalSet (sanity: sel 1 absent from legalSet, present in default movegen)', () => {
  const board = buildPartialBetrayalBoard();
  const partial = sanRecords(board).find((r) => r.struct[10] === 1 && r.struct[0] !== MT_CAPTURE);
  assert.ok(partial !== undefined, 'expected a sel=1 partial-betray SAN under default movegen');
  if (!partial) return;
  const ligungiLegal = legalSet(board, 'ligungi');
  assert.equal(ligungiLegal.has(canonicalMove(partial.struct)), false, 'sel=1 must not be in ligungi legalSet');
});

// F2 (review 2026-11-14, mutant m5: the legalSet check in gameFromHistory
// removed). The previous version of this test passed the START fen as
// `fen`, so the replay (one move, landing elsewhere) was already a mismatch
// against `fen` whether or not the legalSet check ran -- m5 left it green.
// Here `fen` is the position the partial-betray move really reaches, as the
// komugi twin below does, so the only thing that can produce a mismatch is
// the legalSet check itself.
test('ligungi: a history move outside legalSet gives a stamp mismatch even when the replay lands on fen (rule, root.ts gameFromHistory)', () => {
  const board = buildPartialBetrayalBoard();
  const startFen = boardToFen(board);
  const partial = sanRecords(board).find((r) => r.struct[10] === 1 && r.struct[0] !== MT_CAPTURE);
  assert.ok(partial !== undefined, 'expected a sel=1 partial-betray SAN under default movegen');
  if (!partial) return;

  const reached = new Game(boardFromFen(startFen), RepetitionRule.PLACEMENT, 1, 'komugi');
  reached.play(partial.id);
  const reachedFen = boardToFen(reached.board);
  assert.equal(rootGame(reachedFen, [partial.san], startFen, 'komugi').stamp, HISTORY_PRESENT);
  assert.equal(rootGame(reachedFen, [partial.san], startFen, 'ligungi').stamp, HISTORY_MISMATCH);
});

// Twin check: the SAME history move, under 'komugi', replays fine (stamp
// present) -- the mismatch above is specific to ligungi's own legal set,
// not a general resolveSan/gameFromHistory breakage.
test('komugi: the same partial-betray history move replays fine (stamp present)', () => {
  const board = buildPartialBetrayalBoard();
  const startFen = boardToFen(board);
  const partial = sanRecords(board).find((r) => r.struct[10] === 1 && r.struct[0] !== MT_CAPTURE);
  assert.ok(partial !== undefined);
  if (!partial) return;
  const reached = gameFromHistory([partial.san], startFen, 'komugi');
  const reachedFen = boardToFen(reached.board);
  const { stamp } = rootGame(reachedFen, [partial.san], startFen, 'komugi');
  assert.equal(stamp, HISTORY_PRESENT);
});

// Keep generateRecords imported for the sanity check above's indirect use
// via sanRecords -- referenced directly here so a future refactor of
// sanRecords (which currently calls generateRecords with no `rules`) is
// still pinned against the same default-profile enumeration this file
// relies on throughout.
test('sanity: default-profile sanRecords enumeration matches generateRecords(board) with no rules', () => {
  const board = buildPartialBetrayalBoard();
  assert.deepEqual(
    sanRecords(board).map((r) => r.id),
    generateRecords(board).map((r) => r.id),
  );
});

// ---- KOM-1139 F6: SAN/history replay validates rule (1) under the profile -

// A self-exposing move (legal under the default-profile movegen sanRecords/
// resolveSan use, illegal under ligungi's own legalSet): white general at
// (3,1) currently blocks a file-wide check from black's general at (5,1);
// stepping it to (3,2) exposes the marshal. Same shape as
// ligungi_filter.test.ts's L5 test, reused here for root.ts's history path
// instead of Game.validMoves() directly.
function buildSelfExposingBoard(): Board {
  const board = new Board();
  board.mode = MODE_ADVANCED;
  board.turn = WHITE;
  place(board, 1, 1, MARSHAL, WHITE, 1);
  place(board, 9, 9, MARSHAL, BLACK, 1);
  place(board, 5, 1, GENERAL, BLACK, 1);
  place(board, 3, 1, GENERAL, WHITE, 1);
  return board;
}

test('ligungi: a self-exposing history move is outside legalSet (sanity: present under default movegen)', () => {
  const board = buildSelfExposingBoard();
  const exposing = sanRecords(board).find(
    (r) => r.struct[3] === 3 - 1 && r.struct[4] === 1 - 1 && r.struct[7] === 2 - 1,
  );
  assert.ok(exposing !== undefined, 'expected the self-exposing move under default movegen');
  if (!exposing) return;
  const ligungiLegal = legalSet(board, 'ligungi');
  assert.equal(ligungiLegal.has(canonicalMove(exposing.struct)), false, 'self-exposing move must not be in ligungi legalSet');
});

// F6 (review 2026-11-14): root.ts's gameFromHistory already validates each
// replayed move against legalSet(board, rules) under ligungi (KOM-1107) --
// this proves that check ALSO rejects a rule-(1) self-exposure, not just
// KOM-1107's rule-(3) partial betrayal, mirroring the partial-betray test
// above exactly.
test('ligungi: a self-exposing history move gives a stamp mismatch even when the replay lands on fen (rule, root.ts gameFromHistory)', () => {
  const board = buildSelfExposingBoard();
  const startFen = boardToFen(board);
  const exposing = sanRecords(board).find(
    (r) => r.struct[3] === 3 - 1 && r.struct[4] === 1 - 1 && r.struct[7] === 2 - 1,
  );
  assert.ok(exposing !== undefined);
  if (!exposing) return;

  const reached = new Game(boardFromFen(startFen), RepetitionRule.PLACEMENT, 1, 'komugi');
  reached.play(exposing.id);
  const reachedFen = boardToFen(reached.board);
  assert.equal(rootGame(reachedFen, [exposing.san], startFen, 'komugi').stamp, HISTORY_PRESENT);
  assert.equal(rootGame(reachedFen, [exposing.san], startFen, 'ligungi').stamp, HISTORY_MISMATCH);
});

test('komugi: the same self-exposing history move replays fine (stamp present)', () => {
  const board = buildSelfExposingBoard();
  const startFen = boardToFen(board);
  const exposing = sanRecords(board).find(
    (r) => r.struct[3] === 3 - 1 && r.struct[4] === 1 - 1 && r.struct[7] === 2 - 1,
  );
  assert.ok(exposing !== undefined);
  if (!exposing) return;
  const reached = gameFromHistory([exposing.san], startFen, 'komugi');
  const reachedFen = boardToFen(reached.board);
  const { stamp } = rootGame(reachedFen, [exposing.san], startFen, 'komugi');
  assert.equal(stamp, HISTORY_PRESENT);
});

// ---- KOM-1191: seeded no-bot replay games -----------------------------------
//
// Replaces the KOM-1139 two-game replay section: those games were played
// against a built-in opponent, which the user has ruled out of every public
// artifact (tickets/KOM-1191.md). The replacement fixture
// (web/engine/test/fixtures/ligungi_seeded_games.json) is seeded random play
// from the ADVANCED start, each move taken from real ligungi's legal set
// intersected with Komugi's, refereed by real ligungi -- no bot is involved.

const SEEDED_GAMES_FIXTURE_PATH = fileURLToPath(new URL('./fixtures/ligungi_seeded_games.json', import.meta.url));

type SeededGame = {
  plies: string[];
  finalFenKomugi: string;
  finalFenLigungi: string;
  ligungiTerminal: { winner: 'white' | 'black' | null; reason: string };
  komugiTerminalAtGeneration: { winner: 'white' | 'black' | null; reason: string };
  exposingPlies: number;
};

type SeededGamesFixture = {
  schema: string;
  method: string;
  games: Record<string, SeededGame>;
};

function loadSeededGamesFixture(): SeededGamesFixture {
  return JSON.parse(readFileSync(SEEDED_GAMES_FIXTURE_PATH, 'utf8')) as SeededGamesFixture;
}

// winState()'s [whiteWins, blackWins, draw] mapped to the fixture's own
// winner vocabulary (white/black/null for a draw).
function winnerOfWinState([whiteWins, blackWins, draw]: readonly [number, number, number]): 'white' | 'black' | null {
  if (whiteWins) return 'white';
  if (blackWins) return 'black';
  assert.ok(draw, 'expected winState() to report one of white/black/draw');
  return null;
}

test('ligungi: seeded games end where ligungi ends them (final position)', () => {
  const fixture = loadSeededGamesFixture();
  for (const [key, g] of Object.entries(fixture.games)) {
    const game = new Game(boardFromFen(ADVANCED_POSITION), RepetitionRule.FULL, 1, 'ligungi');
    for (const want of g.plies) {
      const recs = generateRecords(game.board, 'ligungi');
      const hit = recs.find((r) => canonicalMove(r.struct) === want);
      assert.ok(hit !== undefined, `game ${key}: ${want} not in the ligungi legal set`);
      game.play(hit!.id);
    }

    assert.equal(winnerOfWinState(game.winState()), g.ligungiTerminal.winner, `game ${key}: ligungi winner`);
    if (g.ligungiTerminal.reason === 'mate') {
      assert.equal(legalSet(game.board, 'ligungi').size, 0, `game ${key}: expected an empty ligungi legal set at mate`);
    }
  }
});

test('ligungi: seeded games replay in full under ligungi and komugi (every ply legal, open until the end, oracle outcome)', () => {
  const fixture = loadSeededGamesFixture();
  for (const [key, g] of Object.entries(fixture.games)) {
    assert.ok(g.plies.length > 0, `game ${key}: expected a non-empty ply list`);
    let exposingPlies = 0;

    for (const rules of ['ligungi', 'komugi'] as const) {
      const rule = rules === 'ligungi' ? RepetitionRule.FULL : RepetitionRule.PLACEMENT;
      const game = new Game(boardFromFen(ADVANCED_POSITION), rule, 1, rules);

      for (let i = 0; i < g.plies.length; i++) {
        const want = g.plies[i];
        const recs = generateRecords(game.board, rules);
        const hit = recs.find((r) => canonicalMove(r.struct) === want);
        assert.ok(hit !== undefined, `game ${key} (${rules}) ply ${i}: ${want} not in the legal set`);
        const mask = game.validMoves();
        assert.equal(mask[hit!.id], 1, `game ${key} (${rules}) ply ${i}: ${want} not in validMoves()`);

        if (i < g.plies.length - 1) {
          const [w, b, d] = game.winState();
          assert.ok(!w && !b && !d, `game ${key} (${rules}) ply ${i}: should not be terminal before the last ply`);
        }

        if (rules === 'ligungi') {
          const ligungiLegal = legalSet(game.board, 'ligungi');
          const komugiLegal = legalSet(game.board, 'komugi');
          if ([...komugiLegal].some((m) => !ligungiLegal.has(m))) exposingPlies += 1;
        }

        game.play(hit!.id);
      }

      assert.equal(
        boardToFen(game.board),
        g.finalFenKomugi,
        `game ${key} (${rules}): final position must match the fixture's finalFenKomugi`,
      );

      const winner = winnerOfWinState(game.winState());
      if (rules === 'ligungi') {
        assert.equal(winner, g.ligungiTerminal.winner, `game ${key} (ligungi): final winner`);
      } else {
        assert.equal(winner, g.komugiTerminalAtGeneration.winner, `game ${key} (komugi): final winner`);
      }
    }

    assert.ok(exposingPlies >= 3, `game ${key}: expected at least 3 plies where komugi's legal set exceeds ligungi's`);
  }
});

// ---- KOM-1203 unit 3: terminal.ts's own knob-isolation tests -------------
// (movegen_board.ts's archer-elbow effect on reachability is already proven
// knob-isolated in ligungi_movegen.test.ts; the two tests below prove the
// SAME two knobs' effect on terminal.ts's own predicates -- check detection
// and checkmate scoring -- using a RulesConfig literal built from
// RULES_PRESETS.komugi with exactly one field flipped, never a profile
// string, so the claim is about the knob's value, not profile identity.)

// Mirrors ligungi_movegen.test.ts's buildTier2ArcherBoard geometry (a tier-2
// archer whose tier-2 staircase target is blocked only by the (2,2) elbow),
// but with a marshal standing on the target square so the same geometry
// exercises inCheck/isCheckmate instead of movegen reachability.
function buildArcherElbowCheckBoard(): Board {
  const board = new Board();
  board.turn = WHITE;
  place(board, 9, 5, ARCHER, WHITE, 2); // origin, tier 2 (stacked on a filler below it)
  place(board, 8, 5, SOLDIER, WHITE, 1); // the piece the archer is stacked on
  place(board, 7, 7, SOLDIER, BLACK, 3); // elbow (lateral2,forward2), taller than tier 2
  place(board, 6, 7, MARSHAL, BLACK, 1); // the attack target, in place of movegen's empty square
  place(board, 1, 1, MARSHAL, WHITE, 1);
  return board;
}

const ARCHER_ELBOW_ONLY: RulesConfig = { ...RULES_PRESETS.komugi, archerElbowBlocking: true };

test('knob isolation (archerElbowBlocking only): check detection follows the SAME (2,2) elbow as movegen', () => {
  assert.equal(
    inCheck(buildArcherElbowCheckBoard(), BLACK, RULES_PRESETS.komugi),
    true,
    "komugi (archerElbowBlocking: false): Black's marshal is in check",
  );
  assert.equal(
    inCheck(buildArcherElbowCheckBoard(), BLACK, ARCHER_ELBOW_ONLY),
    false,
    'archerElbowBlocking only: the (2,2) elbow must block the check',
  );
});

// Mirrors false_mate.test.ts's own KOM-1108 worked example (same FEN, same
// mating-looking move), but drives the divergence from a RulesConfig literal
// that flips ONLY marshalCaptureIsEscape off komugi's baseline, rather than
// the whole 'gungijs' profile string -- proving the knob, not the profile
// identity, is what the fix reads.
const FALSE_MATE_FEN =
  'r1|s:n|m2r|s:g|1/4k|a:J|j2/|n:K:F|1ww1f3/1nt6/2c6/a8/1|D:D:G||S:d|6' + '/1|N:N||S:M|6/9 I1R1F1C1A1T1/d2 w 3 - 33';
const MATING_LOOKING_MOVE = '小(2-4-2)(2-5-2)付';

const MARSHAL_CAPTURE_ESCAPE_OFF: RulesConfig = { ...RULES_PRESETS.komugi, marshalCaptureIsEscape: false };

test('knob isolation (marshalCaptureIsEscape only): a false mate scores as mate exactly like gungijs, unlike komugi', () => {
  const komugiGame = new Game(boardFromFen(FALSE_MATE_FEN), RepetitionRule.FULL, 1, 'komugi');
  const rec = sanRecords(komugiGame.board).find((r) => r.san === MATING_LOOKING_MOVE);
  assert.ok(rec !== undefined, 'expected the mating-looking move to be found by SAN');
  komugiGame.play(rec!.id);
  assert.equal(isCheckmate(komugiGame.board, RULES_PRESETS.komugi), false, "komugi: not actually checkmate");

  const flippedGame = new Game(boardFromFen(FALSE_MATE_FEN), RepetitionRule.FULL, 1, 'komugi');
  const rec2 = sanRecords(flippedGame.board).find((r) => r.san === MATING_LOOKING_MOVE);
  flippedGame.play(rec2!.id);
  assert.equal(
    isCheckmate(flippedGame.board, MARSHAL_CAPTURE_ESCAPE_OFF),
    true,
    'marshalCaptureIsEscape: false must score the false mate as mate, matching gungijs',
  );
});
