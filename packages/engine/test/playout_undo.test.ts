// KOM-1192a: playoutGen (the exact, non-batched descent) now plays its moves
// on the caller's own MatchState/Game and undoes them all before returning,
// instead of cloning the game once per playout. These tests pin the thing a
// clone used to buy for free: the caller's state (and any pre-existing
// history on it) comes back byte-identical, on every exit path -- a normal
// return, a descent that stops at an existing or freshly-built terminal
// node, and a generator abandoned mid-descent.

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { boardFromFen } from '../src/fen.ts';
import { Game } from '../src/game.ts';
import { keyFor, RepetitionRule } from '../src/repetition.ts';
import type { MatchNode } from '../src/search/node.ts';
import { search, type SearchConfig } from '../src/search/search.ts';
import { MatchGameState, type MatchState, type WdlEval } from '../src/search/state.ts';
import { MatchTree, type MatchConfig } from '../src/search/tree.ts';
import { loadSearchFixture } from './fixture.ts';

type SearchRecord = {
  config: SearchConfig & {
    discovery: boolean;
    draftmin: number;
    mlh: { lam: number; tau0: number; tau1: number; slope: number };
  };
  position: { fen: string; history: string[]; stamp: string };
  result: { playouts_used: number };
};

type SearchFixture = { searches: SearchRecord[] };

function loadRecords(): SearchRecord[] {
  const fixture = loadSearchFixture<SearchFixture>();
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
  undoDepth: number;
};

function snapshot(game: Game, rule: RepetitionRule): Snapshot {
  return {
    cells: game.board.cells.slice(),
    hand: game.board.hand.slice(),
    key: keyFor(game.board, rule),
    rootCount: game.repetitionCount(),
    undoDepth: game.undoDepth,
  };
}

function assertSnapshotEqual(label: string, before: Snapshot, after: Snapshot): void {
  assert.deepEqual(Array.from(after.cells), Array.from(before.cells), `${label}: board cells`);
  assert.deepEqual(Array.from(after.hand), Array.from(before.hand), `${label}: board hand`);
  assert.deepEqual(Array.from(after.key), Array.from(before.key), `${label}: repetition key bytes`);
  assert.equal(after.rootCount, before.rootCount, `${label}: root key repetition count`);
  assert.equal(after.undoDepth, before.undoDepth, `${label}: undo-stack depth`);
}

// Reads the repetition count AT an earlier real ply without leaving it
// disturbed: undo down to that ply, read, then replay the same (already
// legal, already played) actions back up to where `game` started. A pure
// query over the public make/undo API -- the same discipline playoutGen
// itself now uses.
function repetitionCountAtDepth(game: Game, actions: number[], targetDepth: number): number {
  const total = actions.length;
  for (let i = total; i > targetDepth; i--) game.undo();
  const count = game.repetitionCount();
  for (let i = targetDepth; i < total; i++) game.play(actions[i]);
  return count;
}

function runSearchAndCheck(label: string, game: Game, rule: RepetitionRule, config: SearchRecord['config']): void {
  const before = snapshot(game, rule);
  const state = new MatchGameState(game, { minPlacements: config.draftmin, isRoot: true });
  const tree = new MatchTree(config, makeDeterministicEvaluator(config.num_actions));
  search(tree, state, config, { discovery: config.discovery });
  const after = snapshot(game, rule);
  assertSnapshotEqual(label, before, after);
}

// A fake evaluator independent of fake_evaluator.ts (this file does not need
// the KOM-1060 fixture's evaluator_vectors pinning, only a valid WDL/logits
// pair from any MatchState's own placementKeyBytes()).
function makeDeterministicEvaluator(numActions: number): (state: MatchState) => WdlEval {
  return (state: MatchState): WdlEval => {
    const bytes = state.placementKeyBytes();
    let h = 2166136261;
    for (let i = 0; i < bytes.length; i++) {
      h ^= bytes[i];
      h = (h * 16777619) >>> 0;
    }
    const logits = new Float32Array(numActions);
    for (let a = 0; a < numActions; a++) logits[a] = ((h >>> (a % 24)) & 0xff) / 64 - 2;
    const w = (h & 0xff) / 1024;
    const d = ((h >>> 8) & 0xff) / 1024;
    const l = Math.max(0, 1 - w - d);
    const total = w + d + l;
    return { policyLogits: logits, wdl: Float64Array.from([w / total, d / total, l / total]) };
  };
}

// Picks the first legal action at every step, up to `maxPlies`, stopping
// early if the position runs out of legal moves. Real plies on a real
// board/rule, not a scripted toy -- this is what gives the "long history"
// case a non-empty undo stack and more than one distinct repetition key
// before the search under test ever starts.
function playGreedily(game: Game, maxPlies: number): number[] {
  const actions: number[] = [];
  for (let i = 0; i < maxPlies; i++) {
    const mask = game.validMoves();
    let action = -1;
    for (let a = 0; a < mask.length; a++) {
      if (mask[a]) {
        action = a;
        break;
      }
    }
    if (action === -1) break;
    game.play(action);
    actions.push(action);
  }
  return actions;
}

function widthAtCurrentPosition(game: Game): number {
  const mask = game.validMoves();
  let width = 0;
  for (let a = 0; a < mask.length; a++) if (mask[a]) width++;
  return width;
}

// A record whose own fixed search already took real playouts (so the root
// has width > 1 and config.playouts > 0), paired with its own FEN/rule --
// used for the "advanced start" (no history at all) and "play-phase" (post
// draft) cases, where the Game is built straight from the fixture's FEN.
function pickRecord(records: SearchRecord[], extra: (rec: SearchRecord) => boolean): SearchRecord {
  const rec = records.find((r) => r.result.playouts_used > 0 && extra(r));
  assert.ok(rec !== undefined, 'no fixture record matched (see test name for which case)');
  return rec as SearchRecord;
}

// Builds a position with real pre-existing history: walk a fixture's start
// position forward a handful of real plies (greedy, first legal action),
// stopping once the resulting position still has width > 1 so the search
// under test actually plays out (not just a zero-playout root-win/width<=1
// return) -- the scenario that actually exercises playoutGen's make/undo.
function buildLongHistoryCase(records: SearchRecord[]): {
  game: Game;
  rule: RepetitionRule;
  actions: number[];
  config: SearchRecord['config'];
} {
  for (const rec of records) {
    if (rec.config.playouts <= 0) continue;
    const rule = ruleOf(rec);
    const game = new Game(boardFromFen(rec.position.fen), rule);
    const actions = playGreedily(game, 12);
    if (actions.length < 4) continue;
    if (widthAtCurrentPosition(game) <= 1) continue;
    return { game, rule, actions, config: rec.config };
  }
  throw new Error('no fixture record produced a width>1 position after a handful of real plies');
}

test('a full exact search leaves the root game byte-identical (KOM-1192a AC3)', () => {
  const records = loadRecords();

  const freshRec = pickRecord(records, (r) => r.position.history.length === 0);
  const freshGame = new Game(boardFromFen(freshRec.position.fen), ruleOf(freshRec));
  runSearchAndCheck('advanced start (no history)', freshGame, ruleOf(freshRec), freshRec.config);

  const playRec = pickRecord(records, (r) => {
    const b = boardFromFen(r.position.fen);
    return b.draftW === 0 && b.draftB === 0;
  });
  const playGame = new Game(boardFromFen(playRec.position.fen), ruleOf(playRec));
  runSearchAndCheck('play-phase position', playGame, ruleOf(playRec), playRec.config);

  const long = buildLongHistoryCase(records);
  const before = snapshot(long.game, long.rule);
  const beforeAncestors = [
    repetitionCountAtDepth(long.game, long.actions, 0),
    repetitionCountAtDepth(long.game, long.actions, Math.floor(long.actions.length / 2)),
  ];
  runSearchAndCheck('long-history position', long.game, long.rule, long.config);
  const after = snapshot(long.game, long.rule);
  assertSnapshotEqual('long-history position (re-check)', before, after);
  const afterAncestors = [
    repetitionCountAtDepth(long.game, long.actions, 0),
    repetitionCountAtDepth(long.game, long.actions, Math.floor(long.actions.length / 2)),
  ];
  assert.deepEqual(afterAncestors, beforeAncestors, 'long-history position: ancestor ply repetition counts');
});

// A tiny scripted MatchState (not the engine's Game): root has two actions.
// Action 0 ends the game at once (a fresh expansion that lands straight on
// expandGen's terminal branch, zero further descent). Action 1 reaches a
// single-action non-terminal node (a fresh expansion into a live node),
// whose only action then ends the game one ply further down -- so a later
// playout through it exits the descent loop on an ALREADY-existing terminal
// child instead of building one. Three different "ends the descent" shapes,
// all of them reachable within a handful of playouts on one small tree.
const CHAIN_ACTIONS = 2;

class ChainState implements MatchState {
  path: number[];
  constructor(path: number[] = []) {
    this.path = path;
  }
  private legal(): number[] {
    const d = this.path.length;
    if (d === 0) return [0, 1];
    if (d === 1 && this.path[0] === 1) return [0];
    return [];
  }
  legalMask(): Uint8Array {
    const mask = new Uint8Array(CHAIN_ACTIONS);
    for (const a of this.legal()) mask[a] = 1;
    return mask;
  }
  play(action: number): void {
    if (!this.legal().includes(action)) throw new Error(`chain: illegal ${action} at ${this.path.join(',')}`);
    this.path.push(action);
  }
  undo(): void {
    this.path.pop();
  }
  clone(): ChainState {
    return new ChainState(this.path.slice());
  }
  isTerminal(): boolean {
    const d = this.path.length;
    return (d === 1 && this.path[0] === 0) || d === 2;
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

function chainEval(state: MatchState): WdlEval {
  const p = (state as ChainState).path;
  const logits = new Float32Array(CHAIN_ACTIONS);
  for (let a = 0; a < CHAIN_ACTIONS; a++) logits[a] = a + p.length;
  return { policyLogits: logits, wdl: Float64Array.from([0.3, 0.4, 0.3]) };
}

function chainTreeConfig(records: SearchRecord[]): MatchConfig {
  return { ...records[0].config, num_actions: CHAIN_ACTIONS };
}

test('a playout ending on a terminal node and one that expands both leave the state unchanged', () => {
  const config = chainTreeConfig(loadRecords());
  const state = new ChainState();
  const tree = new MatchTree(config, chainEval);
  tree.ensureRoot(state);
  assert.equal(state.path.length, 0, 'ensureRoot must not play any move');

  for (let i = 0; i < 8; i++) {
    tree.playout(state);
    assert.equal(state.path.length, 0, `playout ${i}: state left mutated`);
  }
  const root = tree.root as MatchNode;
  assert.equal(root.children.size, 2, 'both root actions (the terminal one and the expanding one) must have run');
});

test('an abandoned generator (return() after the first yield) leaves the state unchanged', async () => {
  const config = chainTreeConfig(loadRecords());
  const state = new ChainState();
  const asyncEval = (s: MatchState): Promise<WdlEval> => Promise.resolve(chainEval(s));
  const tree = new MatchTree(config, asyncEval);
  await tree.ensureRootAsync(state);
  assert.equal(state.path.length, 0, 'ensureRootAsync must not play any move');

  const gen = tree.playoutGen(state);
  const first = gen.next();
  assert.equal(first.done, false, 'the descent must suspend at the evaluator call before this test can abandon it');
  assert.ok(state.path.length > 0, 'the descent must have played at least one move before suspending');
  gen.return(undefined);
  assert.equal(state.path.length, 0, 'an abandoned generator must undo every move its descent played');
});

