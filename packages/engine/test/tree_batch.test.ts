// Fast-mode unit U2: the tree batch loop (beginPlayout / completePlayout /
// playoutBatchGen), docs/design/clientside-fast-mode.md sections 1.3-1.5 and
// 6.2 items 1-4 and 8, at batch 2/4/8/16 from the KOM-1060 search fixture's
// roots plus a scripted toy MatchState for collisions and terminals.
// KOM_SEARCH is required: every test FAILS (never skips) without it (B1).

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { makeFakeEvaluator } from '../src/fake_evaluator.ts';
import { boardFromFen } from '../src/fen.ts';
import { Game } from '../src/game.ts';
import { RepetitionRule } from '../src/repetition.ts';
import { MatchNode } from '../src/search/node.ts';
import { MatchGameState, type MatchState, type WdlEval } from '../src/search/state.ts';
import {
  MatchTree,
  type BatchPlayoutStats,
  type BatchWdlEvaluator,
  type MatchConfig,
  runBatchGenAsync,
  runBatchGenSync,
} from '../src/search/tree.ts';
import { type Overlay } from '../src/search/virtual.ts';
import { loadSearchFixture } from './fixture.ts';

const BATCHES = [2, 4, 8, 16];

type SearchRecord = {
  config: MatchConfig & { discovery: boolean; draftmin: number; playouts: number; mlh: { lam: number } };
  position: { fen: string; stamp: string };
};

function loadRecords(): SearchRecord[] {
  const fixture = loadSearchFixture<{ searches: SearchRecord[] }>();
  assert.ok(fixture.searches.length > 0, 'no search records in the fixture');
  return fixture.searches;
}

// Wraps a state and remembers the action path from the root, so an
// evaluator spy can key rows by (placementKeyBytes, path) (6.2 item 3).
class TracingState implements MatchState {
  inner: MatchState;
  path: number[];
  constructor(inner: MatchState, path: number[] = []) {
    this.inner = inner;
    this.path = path;
  }
  legalMask(): Uint8Array {
    return this.inner.legalMask();
  }
  play(action: number): void {
    this.inner.play(action);
    this.path.push(action);
  }
  undo(): void {
    this.inner.undo();
    this.path.pop();
  }
  clone(): TracingState {
    return new TracingState(this.inner.clone(), this.path.slice());
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

function rootStateFor(rec: SearchRecord): TracingState {
  const board = boardFromFen(rec.position.fen);
  const rule = rec.position.stamp === 'present' ? RepetitionRule.PLACEMENT : RepetitionRule.FULL;
  const game = new Game(board, rule);
  return new TracingState(new MatchGameState(game, { minPlacements: rec.config.draftmin, isRoot: true }));
}

type Spy = {
  calls: string[][];
  overlayRootSums: number[];
};

// The batch evaluator spy: records each row's (key bytes, path) and checks,
// at every call, that the overlay holds exactly this batch's leaves (each
// pending leaf passes through the root once) -- which pins that the previous
// batch's overlay was fully undone (6.2 item 1).
function spyBatch(
  inner: (state: MatchState) => WdlEval | Promise<WdlEval>,
  getTree: () => MatchTree,
  overlay: Overlay,
  spy: Spy,
): BatchWdlEvaluator {
  return (states: MatchState[]) => {
    const keys = states.map((s) => {
      const t = s as TracingState;
      return `${Buffer.from(t.placementKeyBytes()).toString('hex')}|${t.path.join(',')}`;
    });
    spy.calls.push(keys);
    const root = getTree().root as MatchNode;
    const pending = overlay.get(root);
    let sum = 0;
    if (pending !== undefined) for (let i = 0; i < pending.length; i++) sum += pending[i];
    spy.overlayRootSums.push(sum);
    const rows = states.map((s) => inner(s));
    if (rows.some((r) => r instanceof Promise)) return Promise.all(rows);
    return rows as WdlEval[];
  };
}

function walk(root: MatchNode): MatchNode[] {
  const out: MatchNode[] = [];
  const stack = [root];
  while (stack.length > 0) {
    const n = stack.pop()!;
    out.push(n);
    for (const c of n.children.values()) stack.push(c as MatchNode);
  }
  return out;
}

function sumVisits(node: MatchNode): number {
  let s = 0;
  for (let i = 0; i < node.width; i++) s += node.visits[i];
  return s;
}

// 6.2 items 1-4 (and 3's "no leaf evaluated twice in a batch") on one tree.
function checkInvariants(
  label: string,
  tree: MatchTree,
  playouts: number,
  batch: number,
  stats: BatchPlayoutStats,
  overlay: Overlay,
  spy: Spy,
  rootCalls: number,
): void {
  const root = tree.root as MatchNode;
  // item 1
  assert.equal(overlay.size, 0, `${label}: overlay map not empty`);
  tree.assertOverlayClear(overlay);
  for (let b = 0; b < spy.calls.length; b++) {
    assert.equal(spy.overlayRootSums[b], spy.calls[b].length, `${label}: batch ${b} overlay != held leaves`);
  }
  // item 2
  assert.equal(stats.playouts, playouts, `${label}: playouts`);
  assert.equal(sumVisits(root), playouts, `${label}: sum(root.visits)`);
  let terminalNodes = 0;
  for (const node of walk(root)) {
    if (node.isTerminal) terminalNodes++;
    for (const [index, c] of node.children) {
      const child = c as MatchNode;
      if (child.isTerminal) {
        assert.ok(node.visits[index] >= 1, `${label}: terminal edge never backed up`);
      } else {
        assert.equal(node.visits[index], 1 + sumVisits(child), `${label}: visit conservation at an edge`);
      }
    }
  }
  // item 3
  assert.equal(rootCalls, 1, `${label}: root evaluated once, through the single-leaf seam`);
  let rows = 0;
  for (const call of spy.calls) {
    assert.ok(call.length >= 1 && call.length <= batch, `${label}: batch size ${call.length} vs ${batch}`);
    assert.equal(new Set(call).size, call.length, `${label}: a leaf evaluated twice in one batch`);
    rows += call.length;
  }
  assert.equal(spy.calls.length, stats.batches, `${label}: one evaluator call per batch`);
  assert.equal(rows, stats.rows, `${label}: rows`);
  // item 4
  assert.equal(tree.evaluations, 1 + rows, `${label}: evaluations == 1 + non-terminal expansions`);
  assert.equal(tree.evaluations, tree.nodeCount - terminalNodes, `${label}: evaluations == nodes - terminal`);
  assert.ok(tree.evaluations <= playouts + 2, `${label}: evaluations above the worker ceiling`);
  // M-1 accounting
  assert.equal(stats.descents, stats.playouts + stats.collisions, `${label}: descents`);
  assert.equal(stats.playouts, stats.rows + stats.terminals, `${label}: playouts == rows + terminals`);
  assert.ok(stats.collisions <= stats.batches, `${label}: at most one (closing) collision per batch`);
}

function runBatched(rec: SearchRecord, batch: number) {
  const state = rootStateFor(rec);
  const single = makeFakeEvaluator(rec.config.num_actions);
  let rootCalls = 0;
  const overlay: Overlay = new Map();
  const spy: Spy = { calls: [], overlayRootSums: [] };
  const tree: MatchTree = new MatchTree(rec.config, (s) => {
    rootCalls++;
    return single(s);
  });
  tree.batchEvaluator = spyBatch(single, () => tree, overlay, spy);
  tree.ensureRoot(state);
  const playouts = rec.config.playouts;
  const stats = runBatchGenSync(tree.playoutBatchGen(state, playouts, batch, { overlay, checkOverlay: true }));
  return { tree, stats, overlay, spy, rootCalls, playouts };
}

function wideRecords(): SearchRecord[] {
  const out: SearchRecord[] = [];
  for (const rec of loadRecords()) {
    const tree = new MatchTree(rec.config, makeFakeEvaluator(rec.config.num_actions));
    tree.ensureRoot(rootStateFor(rec));
    if ((tree.root as MatchNode).width > 1 && rec.config.playouts > 0) out.push(rec);
  }
  assert.ok(out.length > 0, 'no fixture root with width > 1');
  return out;
}

test('6.2 items 1-4 hold at batch 2/4/8/16 from every wide fixture root', () => {
  const records = wideRecords();
  let checked = 0;
  let multiRow = 0;
  for (const rec of records) {
    for (const batch of BATCHES) {
      const r = runBatched(rec, batch);
      checkInvariants(`${rec.position.fen} B${batch}`, r.tree, r.playouts, batch, r.stats, r.overlay, r.spy, r.rootCalls);
      if (r.spy.calls.some((c) => c.length > 1)) multiRow++;
      checked++;
    }
  }
  console.log(`tree_batch: fixture runs checked=${checked} with a multi-row batch=${multiRow}`);
  assert.ok(multiRow > 0, 'no run ever held more than one leaf: the batched path was not exercised');
});

function biasDump(tree: MatchTree): string {
  const keys = Array.from(tree.bias.entries.keys()).sort((a, b) => a - b);
  return JSON.stringify(
    keys.map((k) => {
      const e = tree.bias.entries.get(k)!;
      return [k, Array.from(e.errSum), e.weightSum, e.gen];
    }),
  );
}

function rootDump(tree: MatchTree): string {
  const root = tree.root as MatchNode;
  return JSON.stringify([Array.from(root.visits), Array.from(root.weight), Array.from(root.wdlSum), Array.from(root.moment)]);
}

test('batch 1 reproduces the exact single-leaf playouts bit for bit (1.7 degeneracy)', () => {
  let checked = 0;
  for (const rec of wideRecords()) {
    const state = rootStateFor(rec);
    const exact = new MatchTree(rec.config, makeFakeEvaluator(rec.config.num_actions));
    exact.ensureRoot(state);
    for (let i = 0; i < rec.config.playouts; i++) exact.playout(state);
    const r = runBatched(rec, 1);
    assert.equal(rootDump(r.tree), rootDump(exact), `${rec.position.fen}: root statistics`);
    assert.equal(biasDump(r.tree), biasDump(exact), `${rec.position.fen}: bias table`);
    assert.equal(r.tree.evaluations, exact.evaluations, `${rec.position.fen}: evaluations`);
    assert.equal(r.tree.nodeCount, exact.nodeCount, `${rec.position.fen}: nodes`);
    assert.equal(r.stats.collisions, 0, `${rec.position.fen}: batch 1 cannot collide`);
    checked++;
  }
  assert.ok(checked > 0);
});

test('batched runs are deterministic and the async driver equals the sync one', async () => {
  const rec = wideRecords()[0];
  for (const batch of [4, 16]) {
    const a = runBatched(rec, batch);
    const b = runBatched(rec, batch);
    assert.equal(rootDump(a.tree), rootDump(b.tree), `B${batch}: root statistics differ between runs`);
    assert.equal(biasDump(a.tree), biasDump(b.tree), `B${batch}: bias tables differ between runs`);

    const state = rootStateFor(rec);
    const single = makeFakeEvaluator(rec.config.num_actions);
    const tree = new MatchTree(rec.config, single);
    let asyncCalls = 0;
    tree.batchEvaluator = (states) => {
      asyncCalls++;
      return Promise.resolve(states.map((s) => single(s)));
    };
    tree.ensureRoot(state);
    assert.throws(
      () => runBatchGenSync(tree.playoutBatchGen(state, rec.config.playouts, batch)),
      /async batch evaluation/,
    );
    const fresh = new MatchTree(rec.config, single);
    fresh.batchEvaluator = tree.batchEvaluator;
    fresh.ensureRoot(state);
    asyncCalls = 0;
    const stats = await runBatchGenAsync(fresh.playoutBatchGen(state, rec.config.playouts, batch));
    assert.equal(asyncCalls, stats.batches, 'one async evaluator call per batch');
    assert.equal(rootDump(fresh), rootDump(a.tree), `B${batch}: async root statistics`);
    assert.equal(biasDump(fresh), biasDump(a.tree), `B${batch}: async bias table`);
  }
});

// 6.2 item 8 (S-7): a scripted toy MatchState. Root (mover 0) has actions
// {0, 1, 2}; 1 and 2 reach terminal positions the opponent has won (an
// all-terminal subtree). Action 0 starts a forced width-1 chain (action 3)
// down to depth 4, then a wide region (actions {0, 1}) down to depth 8, where
// every line ends in a draw.
const TOY_ACTIONS = 4;

class ToyState implements MatchState {
  path: number[];
  constructor(path: number[] = []) {
    this.path = path;
  }
  private legal(): number[] {
    const d = this.path.length;
    if (d === 0) return [0, 1, 2];
    if (this.isTerminal()) return [];
    if (d < 4) return [3];
    return [0, 1];
  }
  legalMask(): Uint8Array {
    const mask = new Uint8Array(TOY_ACTIONS);
    for (const a of this.legal()) mask[a] = 1;
    return mask;
  }
  play(action: number): void {
    if (!this.legal().includes(action)) throw new Error(`toy: illegal ${action} at ${this.path.join(',')}`);
    this.path.push(action);
  }
  undo(): void {
    this.path.pop();
  }
  clone(): ToyState {
    return new ToyState(this.path.slice());
  }
  isTerminal(): boolean {
    const d = this.path.length;
    return (d >= 1 && this.path[0] !== 0) || d >= 8;
  }
  toPlay(): number {
    return this.path.length % 2;
  }
  terminalWdl(): Float64Array {
    if (this.path[0] !== 0) return Float64Array.from([1, 0, 0]); // the mover (opponent) has won
    return Float64Array.from([0, 1, 0]);
  }
  placementKeyBytes(): Uint8Array {
    return Uint8Array.from(this.path);
  }
}

function toyEval(state: MatchState): WdlEval {
  const p = (state as ToyState).path;
  let h = 7;
  for (const a of p) h = (h * 31 + a + 1) % 977;
  const logits = new Float32Array(TOY_ACTIONS);
  for (let a = 0; a < TOY_ACTIONS; a++) logits[a] = ((h * (a + 3)) % 17) / 8;
  logits[0] += 4; // the chain edge dominates the root's prior
  const w = h % 4;
  const d = (h >> 2) % (5 - w);
  return { policyLogits: logits, wdl: Float64Array.from([w / 8, d / 8, (8 - w - d) / 8]) };
}

function toyConfig(): MatchConfig {
  return { ...loadRecords()[0].config, num_actions: TOY_ACTIONS };
}

test('toy chain: a batch closes at its first collision; terminals take no slot (6.2 item 8, M-1)', () => {
  const config = toyConfig();
  for (const batch of BATCHES) {
    const state = new ToyState();
    const overlay: Overlay = new Map();
    const calls: string[][] = [];
    const tree: MatchTree = new MatchTree(config, toyEval);
    tree.batchEvaluator = (states) => {
      calls.push(states.map((s) => (s as ToyState).path.join(',')));
      return states.map((s) => toyEval(s));
    };
    tree.ensureRoot(state);
    assert.equal((tree.root as MatchNode).width, 3);

    // First batch by hand: descents == held + terminal + 1 (the collision).
    let descents = 0;
    let terminals = 0;
    const held = [];
    for (;;) {
      descents++;
      const r = tree.beginPlayout(state, overlay);
      if (r.kind === 'terminal') {
        terminals++;
        continue;
      }
      if (r.kind === 'collision') break;
      held.push(r);
      assert.ok(held.length <= batch, 'a collision must arrive before the batch is full');
    }
    assert.equal(held.length, 1, `B${batch}: one leaf held before the root edge collides`);
    assert.equal(descents, held.length + terminals + 1, `B${batch}: beginPlayout call count`);
    const rootPending = overlay.get(tree.root as MatchNode)!;
    assert.deepEqual(Array.from(rootPending), [1, 0, 0], `B${batch}: the collision left overlay residue`);
    for (const leaf of held) tree.completePlayout(leaf, toyEval(leaf.simulation), overlay);
    assert.equal(overlay.size, 0, `B${batch}: overlay not empty after completion`);
    const firstPlayouts = held.length + terminals;

    // Next batch reaches the width-1 chain node: its only edge collides.
    const before = tree.evaluations;
    const rest = 40;
    const stats = runBatchGenSync(tree.playoutBatchGen(state, rest, batch, { overlay, checkOverlay: true }));
    assert.equal(stats.playouts, rest);
    assert.ok(stats.collisions > 0, `B${batch}: the forced chain never collided`);
    assert.equal(stats.descents, stats.playouts + stats.collisions);
    assert.equal(tree.evaluations - before, stats.rows);
    assert.equal(sumVisits(tree.root as MatchNode), firstPlayouts + rest, `B${batch}: visit conservation`);
    for (const call of calls) assert.equal(new Set(call).size, call.length, `B${batch}: duplicate leaf in a batch`);
    assert.equal(overlay.size, 0);
  }
});

test('an all-terminal budget spends itself without one evaluator call (1.5)', () => {
  const config = toyConfig();
  const tree = new MatchTree(config, toyEval);
  let calls = 0;
  tree.batchEvaluator = (states) => {
    calls++;
    return states.map((s) => toyEval(s));
  };
  // Root at the end of the chain's wide region (depth 7): every child is a
  // terminal draw.
  const state = new ToyState([0, 3, 3, 3, 0, 1, 0]);
  tree.ensureRoot(state);
  const stats = runBatchGenSync(tree.playoutBatchGen(state, 9, 8, { checkOverlay: true }));
  assert.equal(calls, 0);
  assert.equal(stats.terminals, 9);
  assert.equal(stats.batches, 0);
  assert.equal(sumVisits(tree.root as MatchNode), 9);
  assert.equal(tree.evaluations, 1);
});
