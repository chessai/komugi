// Ports src/komugi/search/match/tree.py: the match tree -- ensure_root,
// one-evaluation-per-playout expansion, the WDL-swap backup, the design
// A2 subtree-value-bias wiring, and rehoming (docs/design/clientside-engine-port.md
// section 2.6, section 6 U7b1).

import { SubtreeValueBias } from './bias.ts';
import {
  MatchNode,
  NO_MOVER,
  NO_PREVIOUS_ACTION,
  WDL,
  biasKeyOf,
  legalHashEqual,
  legalHashOf,
  packOuter,
} from './node.ts';
import { selectChild, type MatchConfig as SelectConfig } from './select.ts';
import type { MatchState, WdlEval } from './state.ts';
import { type Overlay, pendingOf, selectChildVirtual } from './virtual.ts';

// Every MatchConfig field tree.ts reads beyond select.ts's own set
// (config.py's bias_* trio: select.ts only ever reads a bias table's
// already-divided `read()` result, never these raw constants).
export interface MatchConfig extends SelectConfig {
  bias_free_prop: number;
  bias_weight_floor: number;
  bias_weight_exponent: number;
}

// tree.py:75 -- how far an evaluator or terminal WDL may sit from summing
// to 1 before the tree refuses it (KOM-179/KOM-218).
export const WDL_SIMPLEX_TOL = 1e-6;

function validateWdlDistribution(value: Float64Array, source: string): void {
  if (value.length !== WDL) {
    throw new Error(`${source} WDL must have length ${WDL}, got ${value.length}`);
  }
  for (let i = 0; i < WDL; i++) {
    if (!Number.isFinite(value[i])) {
      throw new Error(`${source} returned a non-finite wdl: ${Array.from(value)}`);
    }
    if (value[i]! < 0.0) {
      throw new Error(`${source} returned a wdl with a negative entry: ${Array.from(value)}`);
    }
  }
  let total = 0.0;
  for (let i = 0; i < WDL; i++) total += value[i]!;
  if (Math.abs(total - 1.0) > WDL_SIMPLEX_TOL) {
    throw new Error(
      `${source} returned a wdl that is not a distribution: ${Array.from(value)} sums to ` +
        `${total}, which is further than ${WDL_SIMPLEX_TOL} from 1`,
    );
  }
}

// swap (tree.py:133-146): (w, d, l) seen from the other side -- a
// permutation, not a negation, so the draw mass never oscillates across an
// edge where the mover changes.
export function swap(wdl: Float64Array): Float64Array<ArrayBuffer> {
  return Float64Array.from([wdl[2], wdl[1], wdl[0]]);
}

// MatchTreeDesync (tree.py:105-131): a rehomed root does not describe the
// position the next search was handed.
export class MatchTreeDesync extends Error {
  ply: number;
  storedHash: { hi: number; lo: number };
  actualHash: { hi: number; lo: number };

  constructor(ply: number, storedHash: { hi: number; lo: number }, actualHash: { hi: number; lo: number }) {
    super(
      `the retained root's legal set does not match the position at ply ${ply}: stored legal_hash ` +
        `${JSON.stringify(storedHash)}, actual ${JSON.stringify(actualHash)}. The tree describes a ` +
        `different position than the one being searched`,
    );
    this.ply = ply;
    this.storedHash = storedHash;
    this.actualHash = actualHash;
  }
}

function legalActionsOf(mask: Uint8Array): Int32Array {
  let count = 0;
  for (let i = 0; i < mask.length; i++) if (mask[i]) count++;
  const out = new Int32Array(count);
  let k = 0;
  for (let i = 0; i < mask.length; i++) if (mask[i]) out[k++] = i;
  return out;
}

function arraysEqual(a: Int32Array, b: Int32Array): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

// expandGen's legal set, refused before any evaluation is asked for.
function expandableActions(state: MatchState): Int32Array {
  const actions = legalActionsOf(state.legalMask());
  if (actions.length === 0) {
    throw new Error(
      'a non-terminal position reported no legal action: the search has nothing to expand and ' +
        'no value to back up',
    );
  }
  return actions;
}

function leafEvaluation(node: MatchNode): WdlEval {
  return { policyLogits: Float32Array.from(node.logits), wdl: Float64Array.from(node.wdlNet) };
}

// KOM-1078: an evaluator may answer later rather than now (onnxruntime-web's
// InferenceSession.run has no synchronous form). `expand`/`playout` are
// written once, as generators, and yield exactly at the evaluator call;
// everything before and after that point runs in the same order and with
// the same operations regardless of who drives the generator. A sync
// evaluator never actually suspends (isPromiseLike is false, so the `yield`
// inside evaluateGen is never reached), so runGenSync's single `.next()`
// always sees `done: true` -- the pre-KOM-1078 control flow, byte for byte.
export type MaybeAsyncWdlEvaluator = (state: MatchState) => WdlEval | Promise<WdlEval>;

function isPromiseLike<T>(value: T | Promise<T>): value is Promise<T> {
  return typeof value === 'object' && value !== null && typeof (value as Promise<T>).then === 'function';
}

// Drives a generator that only ever yields a Promise<WdlEval>, synchronously.
// Throws if it actually suspends -- a sync evaluator by construction never
// does, so this throwing is the signal that an async evaluator was handed to
// the sync entry point by mistake.
export function runGenSync<T>(gen: Generator<Promise<WdlEval>, T, WdlEval>): T {
  const step = gen.next();
  if (!step.done) {
    throw new Error(
      'the synchronous search driver received an async evaluation: use the Async entry point ' +
        '(ensureRootAsync/playoutAsync/searchAsync) with this evaluator instead',
    );
  }
  return step.value;
}

// Drives the same generator asynchronously: awaits every yielded evaluation
// and resumes the generator with its result, in the order the generator
// asks for them -- one evaluation per expansion, exactly as the sync driver
// sees, just paced by `await` instead of returning immediately.
export async function runGenAsync<T>(gen: Generator<Promise<WdlEval>, T, WdlEval>): Promise<T> {
  let step = gen.next();
  while (!step.done) {
    const value = await step.value;
    step = gen.next(value);
  }
  return step.value;
}

// Fast-mode U2 (docs/design/clientside-fast-mode.md section 2.1): the batch
// seam. Same discipline as the single-leaf seam: a synchronous batch
// evaluator never suspends, so runBatchGenSync's single `.next()` sees
// `done: true`; the async driver awaits each yielded batch.
export type BatchWdlEvaluator = (states: MatchState[]) => WdlEval[] | Promise<WdlEval[]>;

export function runBatchGenSync<T>(gen: Generator<Promise<WdlEval[]>, T, WdlEval[]>): T {
  const step = gen.next();
  if (!step.done) {
    throw new Error(
      'the synchronous batched search driver received an async batch evaluation: use ' +
        'runBatchGenAsync with this batch evaluator instead',
    );
  }
  return step.value;
}

export async function runBatchGenAsync<T>(gen: Generator<Promise<WdlEval[]>, T, WdlEval[]>): Promise<T> {
  let step = gen.next();
  while (!step.done) {
    const value = await step.value;
    step = gen.next(value);
  }
  return step.value;
}

// The default batch seam over a single-leaf evaluator: the evaluator mapped
// over rows, in row order; a promise only if some row answered with one.
export function mapEvaluatorOverRows(evaluator: MaybeAsyncWdlEvaluator): BatchWdlEvaluator {
  return (states: MatchState[]) => {
    const rows = states.map((state) => evaluator(state));
    if (rows.some((row) => isPromiseLike(row))) return Promise.all(rows);
    return rows as WdlEval[];
  };
}

// design 1.3: a descent that stopped at an unexpanded non-terminal edge
// (`node`, `index`), its overlay increments held along `path`.
export type PendingPlayout = {
  kind: 'pending';
  path: Array<[MatchNode, number]>;
  simulation: MatchState;
  biasKey: number;
  actions: Int32Array;
};

export type BeginPlayoutResult = PendingPlayout | { kind: 'terminal' } | { kind: 'collision' };

// What one playoutBatchGen run did: `descents` = beginPlayout calls, so
// descents == playouts + collisions and playouts == held leaves + terminals.
export type BatchPlayoutStats = {
  playouts: number;
  descents: number;
  terminals: number;
  collisions: number;
  batches: number;
  rows: number;
};

export type PlayoutBatchOptions = {
  overlay?: Overlay;
  // design 1.3: "U2 checks this with a full tree walk in debug builds".
  checkOverlay?: boolean;
};

// MatchTree (tree.py:149-652): one retained search tree, its evaluator and
// its bias table.
export class MatchTree {
  private _config: MatchConfig;
  private _evaluator: MaybeAsyncWdlEvaluator;
  private _root: MatchNode | null = null;
  private _rootAction = NO_PREVIOUS_ACTION;
  // The mover at the parent of the edge that reached the current root, or
  // NO_MOVER when no such edge is known (tree.py:166-172).
  private _rootParentToPlay = NO_MOVER;
  private _evaluations = 0;
  private _bias: SubtreeValueBias;
  private _batchEvaluator: BatchWdlEvaluator;

  constructor(config: MatchConfig, evaluator: MaybeAsyncWdlEvaluator, batchEvaluator?: BatchWdlEvaluator) {
    this._config = config;
    this._evaluator = evaluator;
    this._batchEvaluator = batchEvaluator ?? mapEvaluatorOverRows((state) => this._evaluator(state));
    this._bias = new SubtreeValueBias({
      freeProp: config.bias_free_prop,
      weightFloor: config.bias_weight_floor,
    });
  }

  get config(): MatchConfig {
    return this._config;
  }

  // Rebindable: contempt is a property of the reader (design S2), so
  // installing a new config against the same tree is exactly a mid-game
  // contempt change, re-derived at the next read.
  set config(config: MatchConfig) {
    this._config = config;
  }

  get evaluator(): MaybeAsyncWdlEvaluator {
    return this._evaluator;
  }

  // Rebindable for the same reason as `config`: the worker keeps one tree
  // (and so one bias table) for its whole life, like EngineServer's one
  // MatchPlayer, while each request may carry its own MLH settings.
  set evaluator(evaluator: MaybeAsyncWdlEvaluator) {
    this._evaluator = evaluator;
  }

  // Fast-mode U2 (design 2.1): the batch seam playoutBatchGen asks, beside
  // `evaluator` and rebindable for the same reason. Root expansion still
  // goes through `evaluator` (design 6.3 S-1). Defaults to `evaluator`
  // mapped over rows (read at call time, so rebinding `evaluator` follows).
  get batchEvaluator(): BatchWdlEvaluator {
    return this._batchEvaluator;
  }

  set batchEvaluator(batchEvaluator: BatchWdlEvaluator) {
    this._batchEvaluator = batchEvaluator;
  }

  get bias(): SubtreeValueBias {
    return this._bias;
  }

  get root(): MatchNode | null {
    return this._root;
  }

  get rootAction(): number {
    return this._rootAction;
  }

  get evaluations(): number {
    return this._evaluations;
  }

  // node_count (tree.py:236-251): walked iteratively, not recursively.
  get nodeCount(): number {
    if (this._root === null) return 0;
    let seen = 0;
    const stack: MatchNode[] = [this._root];
    while (stack.length > 0) {
      const node = stack.pop()!;
      seen += 1;
      for (const child of node.children.values()) stack.push(child as MatchNode);
    }
    return seen;
  }

  // clear (tree.py:253-265): the bias table is deliberately NOT cleared --
  // an entry is keyed on (mover, move, previous move), not on any node.
  clear(): void {
    this._root = null;
    this._rootAction = NO_PREVIOUS_ACTION;
    this._rootParentToPlay = NO_MOVER;
  }

  // descend (tree.py:267-325): rehome onto the child reached by `action`.
  // Ported for completeness (tree_reuse is off by default, design B2);
  // unused by the U7b1/U7b2 worker.
  descend(action: number): void {
    this._bias.decayAll();
    const root = this._root;
    if (root === null) {
      this.clear();
      return;
    }
    const index = root.childIndex(action);
    const child = index === null ? undefined : (root.children.get(index) as MatchNode | undefined);
    if (child === undefined) {
      this.clear();
      this._rootAction = action;
      this._rootParentToPlay = root.toPlay;
      return;
    }
    child.isRoot = true;
    this._root = child;
    this._rootAction = action;
    this._rootParentToPlay = root.toPlay;
  }

  // check_consistency (tree.py:327-353): an equality of legal sets, checked
  // in both directions. Ported for completeness; unused by the U7b1/U7b2
  // worker (tree_reuse off).
  checkConsistency(state: MatchState, ply = -1): void {
    const root = this._root;
    if (root === null) return;
    const current = legalActionsOf(state.legalMask());
    const actual = legalHashOf(current);
    if (!legalHashEqual(root.legalHash, actual) || !arraysEqual(root.actions, current)) {
      throw new MatchTreeDesync(ply, root.legalHash, actual);
    }
  }

  // ensure_root (tree.py:355-366): one evaluation on an empty tree, zero on
  // a retained one. Sync entry point: throws if the evaluator answers async
  // (runGenSync's own check).
  ensureRoot(state: MatchState): void {
    runGenSync(this.ensureRootGen(state));
  }

  // KOM-1078: the async entry point, for a real per-leaf-async evaluator
  // (onnxruntime-web). Suspends at the one evaluation this can need and
  // resumes with its result; same statements as ensureRoot, same order.
  async ensureRootAsync(state: MatchState): Promise<void> {
    await runGenAsync(this.ensureRootGen(state));
  }

  *ensureRootGen(state: MatchState): Generator<Promise<WdlEval>, void, WdlEval> {
    if (this._root !== null) return;
    const { node } = yield* this.expandGen(state, this.rootBiasKey(state));
    node.isRoot = true;
    this._root = node;
  }

  // playout (tree.py:368-408): select down, expand at most one node, back
  // the leaf's value up the path it came down. `state` is mutated during
  // the descent (make/undo, KOM-1192a) but every move played is undone
  // before this returns, so the caller sees it byte-identical to before the
  // call -- including on an early return (terminal/expansion) or an
  // abandoned generator (try/finally covers both). Sync entry point.
  playout(state: MatchState): void {
    runGenSync(this.playoutGen(state));
  }

  // KOM-1078: async entry point, same statements as playout().
  async playoutAsync(state: MatchState): Promise<void> {
    await runGenAsync(this.playoutGen(state));
  }

  *playoutGen(state: MatchState): Generator<Promise<WdlEval>, void, WdlEval> {
    const root = this._root;
    if (root === null) throw new Error('playout on an empty tree: expand the root first');

    const path: Array<[MatchNode, number]> = [];
    let previousAction = this._rootAction;
    let node = root;
    let evaluation: WdlEval | null = null;
    let plies = 0;

    try {
      while (!node.isTerminal) {
        const index = selectChild(node, this._config, this._bias);
        const action = node.actions[index]!;
        path.push([node, index]);
        state.play(action);
        plies += 1;
        const existing = node.children.get(index) as MatchNode | undefined;
        if (existing === undefined) {
          const key = biasKeyOf(node.toPlay, action, previousAction, this._config.num_actions);
          const expanded = yield* this.expandGen(state, key);
          node.children.set(index, expanded.node);
          node = expanded.node;
          evaluation = expanded.evaluation;
          break;
        }
        previousAction = action;
        node = existing;
      }

      const finalEvaluation = evaluation ?? leafEvaluation(node);
      this.backup(path, node, finalEvaluation);
    } finally {
      for (let i = 0; i < plies; i++) state.undo();
    }
  }

  // Fast-mode U2 (design 1.3-1.5): playoutGen's descent with the virtual
  // scorer in place of selectChild (which it calls itself wherever the
  // overlay is all zero). Ends one of three ways:
  // - 'terminal': an existing terminal node was reached, or the edge's state
  //   is terminal (created and inserted now) -- backed up at once, exactly as
  //   playoutGen does, without touching the overlay;
  // - 'collision': the descent ended at an unexpanded edge a held pending leaf
  //   already ends at. An unexpanded edge carries pending > 0 only because a
  //   held descent ended there (any descent through it would have to stop at
  //   it), so the overlay itself is the (node, index) key of held leaves. The
  //   descent's overlay increments are never applied (equivalently: undone
  //   at once), so nothing is left behind;
  // - 'pending': `overlay` incremented by one along `path`. A descent never
  //   revisits a node, so its own increments could not have changed its own
  //   selections -- applying them at the end is the same as along the way.
  beginPlayout(state: MatchState, overlay: Overlay): BeginPlayoutResult {
    const root = this._root;
    if (root === null) throw new Error('playout on an empty tree: expand the root first');

    const path: Array<[MatchNode, number]> = [];
    let previousAction = this._rootAction;
    let node = root;
    let plies = 0;

    try {
      while (!node.isTerminal) {
        const index = selectChildVirtual(node, this._config, overlay, this._bias);
        const action = node.actions[index]!;
        path.push([node, index]);
        state.play(action);
        plies += 1;
        const existing = node.children.get(index) as MatchNode | undefined;
        if (existing === undefined) {
          const key = biasKeyOf(node.toPlay, action, previousAction, this._config.num_actions);
          if (state.isTerminal()) {
            const terminal = this.terminalNode(state, key);
            node.children.set(index, terminal);
            this.backup(path, terminal, leafEvaluation(terminal));
            return { kind: 'terminal' };
          }
          const held = overlay.get(node);
          if (held !== undefined && held[index]! > 0) return { kind: 'collision' };
          const actions = expandableActions(state);
          for (const [onPath, edge] of path) pendingOf(overlay, onPath)[edge]! += 1;
          return { kind: 'pending', path, simulation: state.clone(), biasKey: key, actions };
        }
        previousAction = action;
        node = existing;
      }

      this.backup(path, node, leafEvaluation(node));
      return { kind: 'terminal' };
    } finally {
      for (let i = 0; i < plies; i++) state.undo();
    }
  }

  // Fast-mode U2 (design 1.3): the non-terminal half of expandGen for a held
  // leaf -- buildNode (bias generation read now, at completion), insert the
  // child, release the leaf's overlay (entries that reach all-zero are
  // removed, so a fully completed batch leaves the map empty), then the
  // ordinary backup.
  completePlayout(pending: PendingPlayout, evaluation: WdlEval, overlay: Overlay): void {
    const [parent, index] = pending.path[pending.path.length - 1]!;
    if (parent.children.has(index)) {
      throw new Error('completePlayout: the pending edge was expanded while its leaf was in flight');
    }
    const built = this.buildNode(pending.simulation, pending.biasKey, pending.actions, evaluation);
    parent.children.set(index, built.node);
    for (const [onPath, edge] of pending.path) {
      const counts = overlay.get(onPath);
      if (counts === undefined || counts[edge]! <= 0) {
        throw new Error('completePlayout: the overlay does not hold this leaf');
      }
      counts[edge]! -= 1;
      let clear = true;
      for (let i = 0; i < counts.length; i++) if (counts[i] !== 0) clear = false;
      if (clear) overlay.delete(onPath);
    }
    this.backup(pending.path, built.node, built.evaluation);
  }

  // Fast-mode U2 (design 1.3 steps 1-4): `playouts` playouts in batches of
  // at most `batch` pending leaves. A batch closes when it holds `batch`
  // leaves, the budget is spent, or at the FIRST collision (M-1); terminal
  // descents count against the budget at once and take no slot. One batch
  // evaluator call per batch, rows in collection order, completed in
  // collection order. At batch 1 every descent sees an all-zero overlay, so
  // this runs playoutGen's selections exactly.
  *playoutBatchGen(
    state: MatchState,
    playouts: number,
    batch: number,
    options: PlayoutBatchOptions = {},
  ): Generator<Promise<WdlEval[]>, BatchPlayoutStats, WdlEval[]> {
    if (!Number.isInteger(batch) || batch < 1) throw new Error(`batch must be an integer >= 1, got ${batch}`);
    if (!Number.isInteger(playouts) || playouts < 0) {
      throw new Error(`playouts must be an integer >= 0, got ${playouts}`);
    }
    const overlay: Overlay = options.overlay ?? new Map();
    if (overlay.size !== 0) throw new Error('playoutBatchGen needs an empty overlay');
    const stats: BatchPlayoutStats = { playouts: 0, descents: 0, terminals: 0, collisions: 0, batches: 0, rows: 0 };

    while (stats.playouts < playouts) {
      const held: PendingPlayout[] = [];
      while (held.length < batch && stats.playouts + held.length < playouts) {
        stats.descents += 1;
        const result = this.beginPlayout(state, overlay);
        if (result.kind === 'terminal') {
          stats.terminals += 1;
          stats.playouts += 1;
          continue;
        }
        if (result.kind === 'collision') {
          stats.collisions += 1;
          break;
        }
        held.push(result);
      }
      if (held.length === 0) continue;

      this._evaluations += held.length;
      const states = held.map((leaf) => leaf.simulation);
      const maybe = this._batchEvaluator(states);
      const rows = isPromiseLike(maybe) ? yield maybe : maybe;
      if (!Array.isArray(rows) || rows.length !== held.length) {
        throw new Error(
          `the batch evaluator must return one row per state: asked ${held.length}, got ` +
            `${Array.isArray(rows) ? rows.length : typeof rows}`,
        );
      }
      for (let i = 0; i < held.length; i++) this.completePlayout(held[i]!, rows[i]!, overlay);
      stats.playouts += held.length;
      stats.rows += held.length;
      stats.batches += 1;
      if (options.checkOverlay) this.assertOverlayClear(overlay);
    }
    return stats;
  }

  // design 1.3's invariant after every batch: the collector's map is empty
  // and a whole-tree walk finds no node with a non-zero pending entry.
  assertOverlayClear(overlay: Overlay): void {
    for (const [, counts] of overlay) {
      for (let i = 0; i < counts.length; i++) {
        if (counts[i] !== 0) throw new Error('virtual-loss overlay not fully undone');
      }
    }
    if (overlay.size !== 0) throw new Error(`virtual-loss overlay map not empty: ${overlay.size} entries`);
    if (this._root === null) return;
    const stack: MatchNode[] = [this._root];
    while (stack.length > 0) {
      const node = stack.pop()!;
      if (overlay.has(node)) throw new Error('virtual-loss overlay still holds a tree node');
      for (const child of node.children.values()) stack.push(child as MatchNode);
    }
  }

  // _root_bias_key (tree.py:410-437): the mover slot is the mover at the
  // PARENT of the edge that reached the current root, remembered across a
  // rehome in `_rootParentToPlay`; with no edge at all it degenerates to the
  // state's own mover.
  private rootBiasKey(state: MatchState): number {
    const parentToPlay = this._rootParentToPlay === NO_MOVER ? state.toPlay() : this._rootParentToPlay;
    return biasKeyOf(parentToPlay, this._rootAction, NO_PREVIOUS_ACTION, this._config.num_actions);
  }

  // _expand (tree.py:439-525): create the node for `state`, asking the
  // evaluator unless it is terminal. A generator: yields exactly once, only
  // when the evaluator itself answers asynchronously (evaluateGen), and only
  // on the non-terminal branch -- the terminal branch never evaluates at all,
  // so a terminal expansion never suspends under either driver.
  private *expandGen(
    state: MatchState,
    biasKey: number,
  ): Generator<Promise<WdlEval>, { node: MatchNode; evaluation: WdlEval | null }, WdlEval> {
    if (state.isTerminal()) {
      return { node: this.terminalNode(state, biasKey), evaluation: null };
    }
    const actions = expandableActions(state);
    const evaluation = yield* this.evaluateGen(state);
    return this.buildNode(state, biasKey, actions, evaluation);
  }

  // expandGen's terminal branch (tree.py:439-525): never evaluates.
  private terminalNode(state: MatchState, biasKey: number): MatchNode {
    const result = Float64Array.from(state.terminalWdl());
    validateWdlDistribution(result, 'terminal ingress');
    return new MatchNode({
      toPlay: state.toPlay(),
      isTerminal: true,
      actions: new Int32Array(0),
      logits: new Float32Array(0),
      wdlNet: result,
      biasKey,
      biasGen: this._bias.generation(biasKey),
    });
  }

  // Fast-mode U2 (design 1.3): expandGen's non-terminal tail after the
  // evaluation, shared with completePlayout -- logits length check, simplex
  // validation, and the MatchNode built with the bias generation read now.
  private buildNode(
    state: MatchState,
    biasKey: number,
    actions: Int32Array,
    evaluation: WdlEval,
  ): { node: MatchNode; evaluation: WdlEval } {
    const config = this._config;
    const logits = evaluation.policyLogits;
    if (logits.length !== config.num_actions) {
      throw new Error(`policy_logits must have length ${config.num_actions}, got ${logits.length}`);
    }
    const value = Float64Array.from(evaluation.wdl);
    validateWdlDistribution(value, 'evaluator');
    const gatheredLogits = new Float32Array(actions.length);
    for (let i = 0; i < actions.length; i++) gatheredLogits[i] = logits[actions[i]!]!;
    const node = new MatchNode({
      toPlay: state.toPlay(),
      isTerminal: false,
      actions,
      logits: gatheredLogits,
      wdlNet: value,
      biasKey,
      biasGen: this._bias.generation(biasKey),
    });
    return { node, evaluation: { policyLogits: logits, wdl: Float64Array.from(value) } };
  }

  // KOM-1078: the one suspension point in the whole tree. `this._evaluator`
  // may return its WdlEval directly (the pre-KOM-1078 sync shape) or a
  // Promise of one; only the latter ever executes `yield`, so a sync
  // evaluator's generator runs to completion on the first `.next()` with
  // exactly the statements evaluate() used to run, in the same order.
  private *evaluateGen(state: MatchState): Generator<Promise<WdlEval>, WdlEval, WdlEval> {
    this._evaluations += 1;
    const maybe = this._evaluator(state);
    if (isPromiseLike(maybe)) {
      return yield maybe;
    }
    return maybe;
  }

  // _backup (tree.py:532-590): accumulate `leaf`'s WDL up `path`, swapping
  // at every mover change. The leaf weighter is UnitWeight (config.py's
  // default, the only one that ships): every backup weight is exactly 1.0,
  // so W(a) == float(N(a)) exactly, as design A5's acceptance requires.
  private backup(path: Array<[MatchNode, number]>, leaf: MatchNode, evaluation: WdlEval): void {
    void evaluation; // UnitWeight.weight ignores the evaluation entirely.
    const weight = 1.0;
    let value = Float64Array.from(leaf.wdlNet);
    let below: MatchNode = leaf;
    for (let i = path.length - 1; i >= 0; i--) {
      const [parent, index] = path[i]!;
      if (below.toPlay !== parent.toPlay) value = swap(value);
      this.updateBias(below, parent);
      parent.visits[index]! += 1;
      parent.weight[index]! += weight;
      parent.wdlSum[index * WDL + 0]! += weight * value[0]!;
      parent.wdlSum[index * WDL + 1]! += weight * value[1]!;
      parent.wdlSum[index * WDL + 2]! += weight * value[2]!;
      const outer = packOuter(value);
      const momentBase = index * outer.length;
      for (let j = 0; j < outer.length; j++) parent.moment[momentBase + j]! += weight * outer[j]!;
      below = parent;
    }
  }

  // _update_bias (tree.py:592-653): replace `node`'s contribution to its
  // key, skipped while W_tot(node) < 1, stored in `parent`'s frame.
  private updateBias(node: MatchNode, parent: MatchNode): void {
    const total = node.totalWeight();
    if (total < 1.0) return;
    const subtreeSum = node.subtreeWdlSum();
    let delta = new Float64Array(WDL);
    for (let i = 0; i < WDL; i++) delta[i] = subtreeSum[i]! / total - node.wdlNet[i]!;
    if (node.toPlay !== parent.toPlay) delta = swap(delta);
    const result = this._bias.update({
      key: node.biasKey,
      contribution: node.biasContribution,
      weight: node.biasWeight,
      gen: node.biasGen,
      delta,
      newWeight: Math.pow(total, this._config.bias_weight_exponent),
    });
    node.biasContribution = result.contribution;
    node.biasWeight = result.weight;
    node.biasGen = result.gen;
  }
}
