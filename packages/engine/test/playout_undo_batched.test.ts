// KOM-1192b1: beginPlayout (the batched/virtual-loss descent) now plays its
// moves on the caller's own MatchState/Game, undoing every ply before
// returning on every exit ('terminal', 'collision' and 'pending' alike),
// and clones ONLY the leaf position when a descent ends 'pending' -- the
// thing a whole-game clone used to buy for every descent, terminal and
// collision included.

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { makeFakeEvaluator } from '../src/fake_evaluator.ts';
import { boardFromFen, boardToFen } from '../src/fen.ts';
import { Game } from '../src/game.ts';
import { keyFor, RepetitionRule } from '../src/repetition.ts';
import type { MatchNode } from '../src/search/node.ts';
import { MatchGameState, type MatchState, type WdlEval } from '../src/search/state.ts';
import { MatchTree, type MatchConfig, runBatchGenSync } from '../src/search/tree.ts';
import { pendingOf, type Overlay } from '../src/search/virtual.ts';
import { loadSearchFixture } from './fixture.ts';

type SearchRecord = {
  config: MatchConfig & { discovery: boolean; draftmin: number; playouts: number; mlh: { lam: number } };
  position: { fen: string; stamp: string };
};

function loadRecords(): SearchRecord[] {
  const fixture = loadSearchFixture<{ searches: SearchRecord[] }>();
  assert.ok(fixture.searches.length > 0, 'no search records in the fixture');
  return fixture.searches;
}

function ruleOf(rec: SearchRecord): RepetitionRule {
  return rec.position.stamp === 'present' ? RepetitionRule.PLACEMENT : RepetitionRule.FULL;
}

type Snapshot = {
  cells: Int8Array;
  hand: Uint8Array;
  key: Uint8Array;
  rootCount: number;
  turns: number;
  progressPlies: number;
  lastMove: Game['lastMove'];
  undoDepth: number;
};

function snapshot(game: Game, rule: RepetitionRule): Snapshot {
  return {
    cells: game.board.cells.slice(),
    hand: game.board.hand.slice(),
    key: keyFor(game.board, rule),
    rootCount: game.repetitionCount(),
    turns: game.turns,
    progressPlies: game.progressPlies,
    lastMove: game.lastMove,
    undoDepth: game.undoDepth,
  };
}

function assertSnapshotEqual(label: string, before: Snapshot, after: Snapshot): void {
  assert.deepEqual(Array.from(after.cells), Array.from(before.cells), `${label}: board cells`);
  assert.deepEqual(Array.from(after.hand), Array.from(before.hand), `${label}: board hand`);
  assert.deepEqual(Array.from(after.key), Array.from(before.key), `${label}: repetition key bytes`);
  assert.equal(after.rootCount, before.rootCount, `${label}: root key repetition count`);
  assert.equal(after.turns, before.turns, `${label}: turns`);
  assert.equal(after.progressPlies, before.progressPlies, `${label}: progressPlies`);
  assert.deepEqual(after.lastMove, before.lastMove, `${label}: lastMove`);
  assert.equal(after.undoDepth, before.undoDepth, `${label}: undo-stack depth`);
}

function rootStateFor(rec: SearchRecord): { game: Game; rule: RepetitionRule; state: MatchGameState } {
  const board = boardFromFen(rec.position.fen);
  const rule = ruleOf(rec);
  const game = new Game(board, rule);
  return { game, rule, state: new MatchGameState(game, { minPlacements: rec.config.draftmin, isRoot: true }) };
}

// Fixture roots wide enough that the first descent off a freshly-expanded
// root actually reaches an unexpanded, non-terminal edge (so it pends
// rather than finishing the game in one ply) and has more than one legal
// action (so a batched search gets to hold more than one leaf at once).
function wideRecords(records: SearchRecord[]): SearchRecord[] {
  const out: SearchRecord[] = [];
  for (const rec of records) {
    if (rec.config.playouts <= 0) continue;
    const { state } = rootStateFor(rec);
    const tree = new MatchTree(rec.config, makeFakeEvaluator(rec.config.num_actions));
    tree.ensureRoot(state);
    if ((tree.root as MatchNode).width > 1) out.push(rec);
  }
  assert.ok(out.length >= 3, 'fewer than 3 wide fixture roots found');
  return out;
}

test('a batched search at batch 4 and 8 leaves the root game byte-identical (KOM-1192b1 AC3)', () => {
  const records = wideRecords(loadRecords());
  for (const rec of records) {
    for (const batch of [4, 8]) {
      const { game, rule, state } = rootStateFor(rec);
      const before = snapshot(game, rule);
      const tree = new MatchTree(rec.config, makeFakeEvaluator(rec.config.num_actions));
      tree.ensureRoot(state);
      runBatchGenSync(tree.playoutBatchGen(state, rec.config.playouts, batch));
      const after = snapshot(game, rule);
      assertSnapshotEqual(`${rec.position.fen} B${batch}`, before, after);
    }
  }
});

// Wraps a MatchState and counts clone() calls, so a descent that must not
// clone (terminal, collision) can be pinned directly rather than inferred
// from timing.
class CountingState implements MatchState {
  inner: MatchState;
  clones: { count: number };
  constructor(inner: MatchState, clones: { count: number }) {
    this.inner = inner;
    this.clones = clones;
  }
  legalMask(): Uint8Array {
    return this.inner.legalMask();
  }
  play(action: number): void {
    this.inner.play(action);
  }
  undo(): void {
    this.inner.undo();
  }
  clone(): CountingState {
    this.clones.count += 1;
    return new CountingState(this.inner.clone(), this.clones);
  }
  isTerminal(): boolean {
    return this.inner.isTerminal();
  }
  toPlay(): number {
    return this.inner.toPlay();
  }
  terminalWdl(): Float64Array {
    return this.inner.terminalWdl();
  }
  placementKeyBytes(): Uint8Array {
    return this.inner.placementKeyBytes();
  }
}

// One legal action, always; it ends the game the instant it's played.
// Root (path=[]) is non-terminal, so the first-ever descent plays exactly
// one ply and lands straight on expandGen's terminal branch.
class TerminalLineState implements MatchState {
  path: number[];
  constructor(path: number[] = []) {
    this.path = path;
  }
  legalMask(): Uint8Array {
    const mask = new Uint8Array(1);
    if (!this.isTerminal()) mask[0] = 1;
    return mask;
  }
  play(action: number): void {
    if (action !== 0) throw new Error(`terminal-line: illegal action ${action}`);
    this.path.push(0);
  }
  undo(): void {
    this.path.pop();
  }
  clone(): TerminalLineState {
    return new TerminalLineState(this.path.slice());
  }
  isTerminal(): boolean {
    return this.path.length >= 1;
  }
  toPlay(): number {
    return this.path.length % 2;
  }
  terminalWdl(): Float64Array {
    return Float64Array.from([0, 1, 0]);
  }
  placementKeyBytes(): Uint8Array {
    return Uint8Array.from(this.path);
  }
}

// One legal action, never terminal within any depth this test reaches --
// so the first descent off the root always lands at an unexpanded,
// non-terminal edge, which a pre-populated overlay turns into a collision.
class CollideLineState implements MatchState {
  path: number[];
  constructor(path: number[] = []) {
    this.path = path;
  }
  legalMask(): Uint8Array {
    const mask = new Uint8Array(1);
    if (!this.isTerminal()) mask[0] = 1;
    return mask;
  }
  play(action: number): void {
    if (action !== 0) throw new Error(`collide-line: illegal action ${action}`);
    this.path.push(0);
  }
  undo(): void {
    this.path.pop();
  }
  clone(): CollideLineState {
    return new CollideLineState(this.path.slice());
  }
  isTerminal(): boolean {
    return this.path.length >= 100;
  }
  toPlay(): number {
    return this.path.length % 2;
  }
  terminalWdl(): Float64Array {
    return Float64Array.from([0, 1, 0]);
  }
  placementKeyBytes(): Uint8Array {
    return Uint8Array.from(this.path);
  }
}

function lineEval(): WdlEval {
  return { policyLogits: Float32Array.from([0]), wdl: Float64Array.from([0.3, 0.4, 0.3]) };
}

function lineConfig(): MatchConfig {
  return { ...loadRecords()[0].config, num_actions: 1 };
}

test(`a 'terminal' descent leaves the state unchanged and allocates no clone`, () => {
  const config = lineConfig();
  const clones = { count: 0 };
  const inner = new TerminalLineState();
  const state = new CountingState(inner, clones);
  const tree = new MatchTree(config, lineEval);
  tree.ensureRoot(state);
  assert.equal(clones.count, 0, 'ensureRoot must not clone');

  const overlay: Overlay = new Map();
  const result = tree.beginPlayout(state, overlay);
  assert.equal(result.kind, 'terminal', 'expected a terminal descent');
  assert.equal(clones.count, 0, "a 'terminal' descent must not clone the state");
  assert.equal(inner.path.length, 0, 'a terminal descent must leave the state unplayed');
});

test(`a 'collision' descent leaves the state unchanged and allocates no clone`, () => {
  const config = lineConfig();
  const clones = { count: 0 };
  const inner = new CollideLineState();
  const state = new CountingState(inner, clones);
  const tree = new MatchTree(config, lineEval);
  tree.ensureRoot(state);
  assert.equal(clones.count, 0, 'ensureRoot must not clone');

  const overlay: Overlay = new Map();
  const root = tree.root as MatchNode;
  pendingOf(overlay, root)[0] = 1; // a held leaf already claims the tree's only edge

  const result = tree.beginPlayout(state, overlay);
  assert.equal(result.kind, 'collision', 'expected a collision descent');
  assert.equal(clones.count, 0, "a 'collision' descent must not clone the state");
  assert.equal(inner.path.length, 0, 'a collision descent must leave the state unplayed');
});

test(`a 'pending' descent's leaf matches a clone-and-play reference, caller state undone`, () => {
  const rec = wideRecords(loadRecords())[0];
  const { game, rule, state } = rootStateFor(rec);
  const before = snapshot(game, rule);

  const tree = new MatchTree(rec.config, makeFakeEvaluator(rec.config.num_actions));
  tree.ensureRoot(state);
  const overlay: Overlay = new Map();
  const result = tree.beginPlayout(state, overlay);
  assert.equal(result.kind, 'pending', 'expected the first descent on a fresh wide root to pend');

  const after = snapshot(game, rule);
  assertSnapshotEqual(`${rec.position.fen}: caller state after a pending descent`, before, after);

  if (result.kind !== 'pending') throw new Error('unreachable');
  const reference = rootStateFor(rec).state;
  for (const [node, index] of result.path) reference.play((node as MatchNode).actions[index]);

  const got = result.simulation as MatchGameState;
  assert.equal(boardToFen(got.game.board), boardToFen(reference.game.board), 'pending leaf FEN');
  assert.equal(got.game.repetitionCount(), reference.game.repetitionCount(), 'pending leaf repetition count');
});
