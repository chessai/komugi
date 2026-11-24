// U6: fixture-game replay bench and the byte-exact undo checker
// (docs/design/clientside-engine-port.md section 3 and unit U6, M9).
//
// A *transition* is play(action) followed by isTerminal(), the unit the
// Python survey counts (src/komugi/search/match/search.py:326-329).

import { ADVANCED_POSITION } from '../src/constants.ts';
import { boardFromFen } from '../src/fen.ts';
import { Game } from '../src/game.ts';
import { RepetitionRule } from '../src/repetition.ts';
import { resolveSan } from '../src/san.ts';

export type BenchGame = { name: string; actions: Int32Array };

type HistoryRecord = { id: string; history: string[] };

// isTerminal = winState (section 3 "Terminal checks").
export function isTerminal(game: Game): boolean {
  const w = game.winState();
  return w[0] !== 0 || w[1] !== 0 || w[2] !== 0;
}

export function newGame(rule: RepetitionRule): Game {
  return new Game(boardFromFen(ADVANCED_POSITION), rule);
}

// One game per fixture game index (record id `g<i>p<ply>`, from the recorded parity fixture):
// the longest history recorded for it, resolved to action ids once, outside
// any timed loop.
export function gamesFromRecords(records: HistoryRecord[]): BenchGame[] {
  const longest = new Map<string, string[]>();
  for (const rec of records) {
    const name = rec.id.split('p')[0];
    const prev = longest.get(name);
    if (!prev || rec.history.length > prev.length) longest.set(name, rec.history);
  }
  const games: BenchGame[] = [];
  for (const [name, history] of longest) {
    const game = newGame(RepetitionRule.PLACEMENT);
    const actions = new Int32Array(history.length);
    for (let i = 0; i < history.length; i++) {
      actions[i] = resolveSan(game.board, history[i]);
      game.play(actions[i]);
    }
    games.push({ name, actions });
  }
  return games;
}

// Plays every game from the start once; returns the transition count.
export function replayOnce(games: BenchGame[], rule: RepetitionRule): number {
  let transitions = 0;
  for (const g of games) {
    const game = newGame(rule);
    for (let i = 0; i < g.actions.length; i++) {
      game.play(g.actions[i]);
      isTerminal(game);
      transitions++;
    }
  }
  return transitions;
}

export type ReplayRate = { rounds: number; transitions: number; ms: number; rate: number };

// Repeats replayOnce until both `minRounds` and `minMs` are reached, after
// one untimed warm-up round.
export function benchReplay(
  games: BenchGame[],
  rule: RepetitionRule,
  minMs = 1000,
  minRounds = 3,
): ReplayRate {
  replayOnce(games, rule);
  let rounds = 0;
  let transitions = 0;
  const t0 = performance.now();
  let ms = 0;
  while (rounds < minRounds || ms < minMs) {
    transitions += replayOnce(games, rule);
    rounds++;
    ms = performance.now() - t0;
  }
  return { rounds, transitions, ms, rate: (transitions * 1000) / ms };
}

export type ReplaySplit = { playMs: number; terminalMs: number; transitions: number };

// Where a replay round's time goes: play() versus isTerminal(), timed
// separately over one round (diagnostic only, never gated).
export function splitReplay(games: BenchGame[], rule: RepetitionRule): ReplaySplit {
  let playMs = 0;
  let terminalMs = 0;
  let transitions = 0;
  for (const g of games) {
    const game = newGame(rule);
    for (let i = 0; i < g.actions.length; i++) {
      const t0 = performance.now();
      game.play(g.actions[i]);
      const t1 = performance.now();
      isTerminal(game);
      terminalMs += performance.now() - t1;
      playMs += t1 - t0;
      transitions++;
    }
  }
  return { playMs, terminalMs, transitions };
}

// Every piece of state play() may touch: board bytes and scalars, the game
// counters, the last move, and the whole repetition count table (Game's
// `counts` is TS-private, which type stripping leaves as a plain field).
// A table-wide view is needed because an undo that skips the key-count
// decrement leaves the *current* position's count unchanged.
export function captureState(game: Game): string {
  const b = game.board;
  const table = (game as unknown as { counts: { snapshot(): string } }).counts.snapshot();
  return JSON.stringify({
    cells: Array.from(b.cells),
    hand: Array.from(b.hand),
    handOrder: Array.from(b.handOrder.subarray(0, b.handOrderLen)),
    turn: b.turn,
    mode: b.mode,
    draftW: b.draftW,
    draftB: b.draftB,
    moveNumber: b.moveNumber,
    turns: game.turns,
    progressPlies: game.progressPlies,
    lastMove: game.lastMove,
    player: game.player,
    repetition: game.repetitionCount(),
    table,
  });
}

export type UndoCheck = { plies: number; mismatches: number; first: string };

// After every ply: undo must restore the pre-move state exactly, and
// replaying the same move must land on exactly the post-move state again
// (a public-API view that also catches a missed decrement: the redone
// position's repetitionCount would be one too high). At the end of each game
// the whole game is unwound back to the start position.
export function checkUndo(makeGame: () => Game, games: BenchGame[]): UndoCheck {
  let plies = 0;
  let mismatches = 0;
  let first = '';
  const miss = (label: string) => {
    mismatches++;
    if (!first) first = label;
  };
  for (const g of games) {
    const game = makeGame();
    const start = captureState(game);
    for (let i = 0; i < g.actions.length; i++) {
      plies++;
      const before = captureState(game);
      game.play(g.actions[i]);
      const after = captureState(game);
      game.undo();
      if (captureState(game) !== before) miss(`${g.name}#undo${i + 1}`);
      game.play(g.actions[i]);
      if (captureState(game) !== after) miss(`${g.name}#redo${i + 1}`);
    }
    for (let i = g.actions.length; i > 0; i--) game.undo();
    if (captureState(game) !== start) miss(`${g.name}#unwind`);
  }
  return { plies, mismatches, first };
}
