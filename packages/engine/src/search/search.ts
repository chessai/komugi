// Ports src/komugi/search/match/search.py: the playout loop, the KOM-233
// root-win / refutation-survey filter and its three-way rule
// (docs/design/clientside-engine-port.md section 2.6, section 6 U7b2).
//
// Deliberately NOT a MatchSearch *class*: Python's class exists to hold a
// retained tree across moves for tree_reuse, and reuse is off by default and
// unused by this worker (tree.ts's own note on descend/checkConsistency).
// This module is the one call tree_reuse=False always makes: a fresh search
// over a tree the caller owns. There is also no TacticalProvider seam here
// -- the Python protocol exists so the match package need not import the
// engine (design §M2); this port has exactly one engine, so "who won this
// already-terminal position" is answered directly from MatchState's own
// terminalWdl()/toPlay(), the mover-frame one-hot search.py already expects
// (the GameMarshalCaptureProvider's terminal_winner reads the same absolute
// outcome by a different route, in the Python tactical-eval helper).

import type { Game } from '../game.ts';
import type { MatchNode } from './node.ts';
import { lcbScores, nodeUtility, selectAction, type MatchConfig as SelectConfig } from './select.ts';
import type { MatchState, WdlEval } from './state.ts';
import {
  runBatchGenAsync,
  runBatchGenSync,
  runGenAsync,
  runGenSync,
  type BatchWdlEvaluator,
  type MatchConfig as TreeConfig,
  type MatchTree,
  type MaybeAsyncWdlEvaluator,
} from './tree.ts';
import type { Overlay } from './virtual.ts';

const WDL = 3;

// Reply-survey transitions one move may spend before the scan stops
// (search.py's MatchSearch.DEFAULT_TRANSITION_CAP).
export const DEFAULT_TRANSITION_CAP = 20000;

// MatchTactical.skipped_reason's only value (search.py's _SKIPPED_DRAFT).
const SKIPPED_DRAFT = 'draft';

// GameMarshalCaptureProvider.provider_id (tactical.py:65): recorded on every
// tactical record this module produces, matching what the server's
// discovery provider identifies itself as. Not a comparison-rule field
// (design 4.3 lists witness/refuted/overridden/calls/transitions/
// skipped_reason/truncated, not provider_id), kept only for diagnostic
// parity with the shape MatchTactical has.
const PROVIDER_ID = 'game-marshal-capture-v1';

export interface SearchConfig extends TreeConfig {
  playouts: number;
}

export interface MatchTactical {
  providerId: string;
  rootWitness: number | null;
  refuted: number[];
  overridden: number | null;
  providerCalls: number;
  candidateTransitions: number;
  skippedReason: string | null;
  truncated: boolean;
}

export interface MatchResult {
  action: number;
  evaluationsUsed: number;
  playoutsUsed: number;
  nodesAtEntry: number;
  rootWdl: Float64Array;
  rootUtility: number;
  actions: Int32Array;
  visits: Int32Array;
  weights: Float64Array;
  lcbScores: Float64Array;
  tactical: MatchTactical | null;
  // KOM-1081: per-phase wall time, present only when SearchOptions.clock was
  // given (null otherwise, so the default search reads no clock at all).
  phases: SearchPhases | null;
  // Fast-mode U3 (design 1.7 C3): true only when SearchOptions.shouldStop
  // fired before config.playouts was reached, in which case playoutsUsed is
  // the real (partial) count. Always false through search()/searchAsync(),
  // which never see a deadline. Deliberately not named `truncated` --
  // MatchTactical.truncated already means "the survey hit its transition
  // cap", an unrelated condition.
  deadlineHit: boolean;
}

// KOM-1081: where one search's wall time went, in the clock's units
// (milliseconds for performance.now()). setup + rootEval + rootWin +
// playouts + select + survey + result == searchMs; evaluateMs is the
// evaluator's share of rootEval + playouts (a subset, not an extra phase).
export interface SearchPhases {
  setupMs: number; // tree.clear, consistency check, legal/terminal guards
  rootEvalMs: number; // ensureRoot: the root's one evaluation and expansion
  rootWinMs: number; // root-win scan (0 when skipped)
  playoutsMs: number; // every playout, evaluator time included
  selectMs: number; // the unfiltered selectAction
  surveyMs: number; // refutation survey + the filtered selectAction (0 when skipped)
  resultMs: number; // buildResult (LCB scores, root WDL)
  searchMs: number; // entry to return
  evaluateMs: number; // inside the evaluator, summed over every call
  evaluatorCalls: number;
}

export interface SearchOptions {
  // KOM-233's root decisive/anti-decisive filter (discovery=false constructs
  // no survey at all, es.EngineServer's own "no provider" reading).
  discovery: boolean;
  tacticalTransitionCap?: number;
  ply?: number;
  // KOM-1081: a monotonic clock (e.g. () => performance.now()). Given, the
  // result carries `phases`; absent, nothing is timed. Instrumentation only:
  // it never changes what is computed or in which order.
  clock?: () => number;
  // Fast-mode U3 (design 1.7 C3, 4.3): checked once per completed batch
  // (once per completed descent when the deadline forces batch=1) -- never
  // mid-batch, since an in-flight evaluation cannot be cancelled (design
  // 5.3). Present => searchFast/searchFastAsync route through
  // searchBatchedGen regardless of the requested batch (the deadline has
  // exactly one implementation). Unread by search()/searchAsync().
  shouldStop?: () => boolean;
  // KOM-1195 U3 (design section 4): called at every chunk boundary of the
  // chunked loop, the last included, right after playoutsUsed is updated and
  // before the yield and the shouldStop poll. A pure counter callback --
  // this module has no SAN encoder and no knowledge of protocol.ts. Unread
  // by the continuous branch (no opts.shouldStop) and by search()/searchAsync().
  onProgress?: (playoutsUsed: number) => void;
  // KOM-1195 U3 (design section 3): yields one macrotask per chunk boundary
  // where playouts remain, before that boundary's shouldStop poll, so a
  // `stop` message dispatched during a wasm/microtask-only evaluation is
  // seen by the next poll. Only the async chunked loop sets this; the
  // generator ignores the resumption value.
  yieldToHost?: () => Promise<WdlEval[]>;
}

function isPromiseLike<T>(value: T | Promise<T>): value is Promise<T> {
  return typeof value === 'object' && value !== null && typeof (value as Promise<T>).then === 'function';
}

type LapPhase = 'setupMs' | 'rootEvalMs' | 'rootWinMs' | 'playoutsMs' | 'selectMs' | 'surveyMs' | 'resultMs';

// KOM-1081: accumulates one search's phase times. `lap(name)` charges the
// time since the previous lap to `name`.
class PhaseTimer {
  private readonly clock: () => number;
  private readonly start: number;
  private last: number;
  private readonly phases: SearchPhases = {
    setupMs: 0,
    rootEvalMs: 0,
    rootWinMs: 0,
    playoutsMs: 0,
    selectMs: 0,
    surveyMs: 0,
    resultMs: 0,
    searchMs: 0,
    evaluateMs: 0,
    evaluatorCalls: 0,
  };

  constructor(clock: () => number) {
    this.clock = clock;
    this.start = clock();
    this.last = this.start;
  }

  lap(name: LapPhase): void {
    const now = this.clock();
    this.phases[name] += now - this.last;
    this.last = now;
  }

  finish(): SearchPhases {
    this.lap('resultMs');
    this.phases.searchMs = this.last - this.start;
    return { ...this.phases };
  }

  // The same evaluator, timed: a sync answer is charged on return, an async
  // one when its promise settles. The value passes through untouched.
  wrap(inner: MaybeAsyncWdlEvaluator): MaybeAsyncWdlEvaluator {
    return (state: MatchState) => {
      const t0 = this.clock();
      this.phases.evaluatorCalls += 1;
      const out = inner(state);
      if (isPromiseLike(out)) {
        return out.then((value) => {
          this.phases.evaluateMs += this.clock() - t0;
          return value;
        });
      }
      this.phases.evaluateMs += this.clock() - t0;
      return out;
    };
  }

  // Fast-mode U3 (design 6.3 N-4): the batch seam, timed the same way but
  // charging one `evaluatorCalls` per BATCH (not per row) -- a batch of `k`
  // rows is one evaluator call, whatever `k` is. Callers that want the root
  // call counted too must also `wrap` `tree.evaluator` with the same timer
  // (startBatchTimer does both), and must give the batched seam its OWN
  // evaluator that does not itself call through the wrapped single-leaf
  // seam, or the root's one call would be charged twice.
  wrapBatch(inner: BatchWdlEvaluator): BatchWdlEvaluator {
    return (states: MatchState[]) => {
      const t0 = this.clock();
      this.phases.evaluatorCalls += 1;
      const out = inner(states);
      if (isPromiseLike(out)) {
        return out.then((value) => {
          this.phases.evaluateMs += this.clock() - t0;
          return value;
        });
      }
      this.phases.evaluateMs += this.clock() - t0;
      return out;
    };
  }
}

// Ascending action ids where `mask` is truthy (np.flatnonzero's own order).
function actionsFromMask(mask: Uint8Array): Int32Array {
  let count = 0;
  for (let i = 0; i < mask.length; i++) if (mask[i]) count++;
  const out = new Int32Array(count);
  let k = 0;
  for (let i = 0; i < mask.length; i++) if (mask[i]) out[k++] = i;
  return out;
}

// _legal_actions (search.py:461-469): `state`'s legal action ids, ascending
// (np.flatnonzero's own order).
function legalActionsOf(state: MatchState): Int32Array {
  return actionsFromMask(state.legalMask());
}

// KOM-1219 design section 2.1: the sole legal action, or null when there is
// more than one -- a plain `legalMask()` read, with no drafting branch
// (unlike `draftIsOpen` below). Exported so `engine.ts`'s `tryForced` can
// certify a forced move off the legal mask alone, before any evaluator call.
export function soleLegalAction(state: MatchState): number | null {
  const legal = legalActionsOf(state);
  return legal.length === 1 ? legal[0]! : null;
}

// KOM-1193a: the survey's probes are never the root view (every probe is at
// least one ply below it), so MatchGameState.legalMask()'s isRoot/threshold
// branch never fires for them -- `game.validMoves()` (its non-root "else")
// is the exact same mask a clone-and-play of the probe would have produced.
// Playing straight on `state.game` and reading it this way is what lets the
// survey below skip a MatchGameState allocation per probe.
function gameLegalActions(game: Game): Int32Array {
  return actionsFromMask(game.validMoves());
}

// KOM-1193a: terminal_wdl/winnerOf folded onto a bare Game (no MatchState
// wrapper to allocate), for the same reason as gameLegalActions above --
// winState() and player are absolute/mover-frame facts that never depend on
// MatchGameState's depth/isRoot bookkeeping.
function gameIsTerminal(game: Game): boolean {
  const w = game.winState();
  return w[0] !== 0 || w[1] !== 0 || w[2] !== 0;
}

function gameWinner(game: Game): number | null {
  const outcome = game.winState();
  const mover = game.player;
  if (outcome[mover] === 1) return mover;
  if (outcome[1 - mover] === 1) return 1 - mover;
  return null;
}

// _draft_is_open (search.py:430-458): whether EITHER side still holds
// drafting rights. Python duck-types across three state shapes because the
// match package may not import the engine; this port has exactly one state
// implementation (state.ts's MatchGameState), so the equivalent read is
// straight to its public `game.board` fields rather than a getattr chain --
// still "False -- scan, do not skip" for anything that does not look like
// that shape, the same quiet default Python's fallback ends on.
function draftIsOpen(state: MatchState): boolean {
  const board = (state as unknown as { game?: { board?: { draftW?: number; draftB?: number } } }).game
    ?.board;
  if (board === undefined) return false;
  return Boolean(board.draftW) || Boolean(board.draftB);
}

// _winner, folded into "ask who won an already-terminal state" (search.py's
// _TacticalSurvey._winner, tactical.py's terminal_winner): terminalWdl() is
// already the mover-frame one-hot a terminal MatchState carries, so the
// absolute winner is read off it and `toPlay()` with no provider call at
// all. Only ever called on a state that `isTerminal()` has already said yes
// to, exactly like the Python call sites.
function winnerOf(state: MatchState): number | null {
  const wdl = state.terminalWdl();
  if (wdl[0] === 1.0) return state.toPlay();
  if (wdl[2] === 1.0) return 1 - state.toPlay();
  return null;
}

function allTrue(mask: Uint8Array): boolean {
  for (let i = 0; i < mask.length; i++) if (!mask[i]) return false;
  return true;
}

function anyTrue(mask: Uint8Array): boolean {
  for (let i = 0; i < mask.length; i++) if (mask[i]) return true;
  return false;
}

// _TacticalSurvey (search.py:260-423): one move's worth of rules-layer
// questions and what they cost. Created per `search` call, discarded with
// it -- nothing here survives a move and the tree never sees it.
// Exported for survey_undo.test.ts (KOM-1193a), which drives it directly
// against a clone-based reference; `search`/`searchBatchedGen` below are its
// only other callers.
export class TacticalSurvey {
  private readonly cap: number;
  private calls = 0;
  private transitions = 0;
  private refuted: number[] = [];
  private truncated = false;

  constructor(transitionCap: number) {
    this.cap = transitionCap;
  }

  private winner(state: MatchState): number | null {
    this.calls += 1;
    return winnerOf(state);
  }

  // root_win (search.py:306-333): every legal move, ascending, played on a
  // clone; the first the rules layer adjudicates a win for the mover.
  // KOM-1193a: `state.game` is make/undo'd in place instead -- one probe,
  // same order, same outcome, no MatchGameState allocation. A state with no
  // `game` (defensive only: the interface requires one) falls back to the
  // original clone-per-probe path.
  rootWin(state: MatchState): number | null {
    const mover = state.toPlay();
    const legal = legalActionsOf(state);
    const game = state.game;
    if (game === undefined) {
      for (let i = 0; i < legal.length; i++) {
        const action = legal[i]!;
        const probe = state.clone();
        probe.play(action);
        this.transitions += 1;
        if (!probe.isTerminal()) continue;
        if (this.winner(probe) === mover) return action;
      }
      return null;
    }
    for (let i = 0; i < legal.length; i++) {
      const action = legal[i]!;
      game.play(action);
      this.transitions += 1;
      const terminal = gameIsTerminal(game);
      let winner: number | null = null;
      if (terminal) {
        this.calls += 1;
        winner = gameWinner(game);
      }
      game.undo();
      if (!terminal) continue;
      if (winner === mover) return action;
    }
    return null;
  }

  // refutation_mask (search.py:335-394): `(L,)` over `root`'s edges, false
  // where a reply ends the game in the opponent's favour. Root edge order
  // (root.actions, ascending); each reply ascending; stops at the first
  // winning reply. The 20,000-transition cap is checked before every
  // transition and counts this pass only.
  // KOM-1193a: same make/undo swap as rootWin -- two nested plays on the one
  // shared `game` (the reply undone before the answer, the answer undone
  // before the next root edge), same probe order and cap accounting as the
  // clone-per-probe version it replaces.
  refutationMask(state: MatchState, root: MatchNode): Uint8Array {
    const allowed = new Uint8Array(root.width).fill(1);
    const refuted: number[] = [];
    let spent = 0;
    const opponent = 1 - state.toPlay();
    const game = state.game;
    if (game === undefined) {
      for (let index = 0; index < root.width; index++) {
        if (spent >= this.cap) {
          this.truncated = true;
          break;
        }
        const action = root.actions[index]!;
        const answered = state.clone();
        answered.play(action);
        this.transitions += 1;
        spent += 1;
        if (answered.isTerminal()) continue;
        const replies = legalActionsOf(answered);
        for (let r = 0; r < replies.length; r++) {
          if (spent >= this.cap) {
            this.truncated = true;
            break;
          }
          const reply = replies[r]!;
          const refutation = answered.clone();
          refutation.play(reply);
          this.transitions += 1;
          spent += 1;
          if (refutation.isTerminal() && this.winner(refutation) === opponent) {
            allowed[index] = 0;
            refuted.push(action);
            break;
          }
        }
      }
      this.refuted = refuted;
      return allowed;
    }
    for (let index = 0; index < root.width; index++) {
      if (spent >= this.cap) {
        this.truncated = true;
        break;
      }
      const action = root.actions[index]!;
      game.play(action);
      this.transitions += 1;
      spent += 1;
      if (!gameIsTerminal(game)) {
        const replies = gameLegalActions(game);
        for (let r = 0; r < replies.length; r++) {
          if (spent >= this.cap) {
            this.truncated = true;
            break;
          }
          const reply = replies[r]!;
          game.play(reply);
          this.transitions += 1;
          spent += 1;
          const refutationTerminal = gameIsTerminal(game);
          let isRefutation = false;
          if (refutationTerminal) {
            this.calls += 1;
            isRefutation = gameWinner(game) === opponent;
          }
          game.undo();
          if (isRefutation) {
            allowed[index] = 0;
            refuted.push(action);
            break;
          }
        }
      }
      game.undo();
    }
    this.refuted = refuted;
    return allowed;
  }

  record(opts: {
    rootWitness?: number | null;
    overridden?: number | null;
    skippedReason?: string | null;
  }): MatchTactical {
    return {
      providerId: PROVIDER_ID,
      rootWitness: opts.rootWitness ?? null,
      refuted: this.refuted.slice(),
      overridden: opts.overridden ?? null,
      providerCalls: this.calls,
      candidateTransitions: this.transitions,
      skippedReason: opts.skippedReason ?? null,
      truncated: this.truncated,
    };
  }
}

function buildResult(
  tree: MatchTree,
  config: SelectConfig,
  root: MatchNode,
  args: {
    action: number;
    playoutsUsed: number;
    nodesAtEntry: number;
    evaluationsBefore: number;
    tactical: MatchTactical | null;
    deadlineHit: boolean;
  },
  timer: PhaseTimer | null,
): MatchResult {
  const total = root.totalWeight();
  let rootWdl: Float64Array;
  if (total > 0.0) {
    const subtreeSum = root.subtreeWdlSum();
    rootWdl = new Float64Array(WDL);
    for (let i = 0; i < WDL; i++) rootWdl[i] = subtreeSum[i]! / total;
  } else {
    rootWdl = Float64Array.from(root.wdlNet);
  }
  const result: MatchResult = {
    action: args.action,
    evaluationsUsed: tree.evaluations - args.evaluationsBefore,
    playoutsUsed: args.playoutsUsed,
    nodesAtEntry: args.nodesAtEntry,
    rootWdl,
    rootUtility: nodeUtility(root, config),
    actions: Int32Array.from(root.actions),
    visits: Int32Array.from(root.visits),
    weights: Float64Array.from(root.weight),
    lcbScores: lcbScores(root, config, tree.bias),
    tactical: args.tactical,
    phases: null,
    deadlineHit: args.deadlineHit,
  };
  if (timer !== null) result.phases = timer.finish();
  return result;
}

// MatchSearch.search (search.py:552-681): run one move's search from `state`
// over `tree` and return what it decided. `state` is never mutated -- every
// playout and every survey transition walks a clone. `tree` is always
// cleared first: this port never models tree_reuse=True (tree.ts's own
// note), so "nodes_at_entry == 0" is this function's constant answer rather
// than a regime it has to prove.
//
// KOM-1078: written once as a generator so both the sync `search()` (fake
// evaluator, tests, the frozen o12 torch-fed control) and the async
// `searchAsync()` (a real per-leaf-async evaluator, e.g. onnxruntime-web)
// run the exact same statements in the exact same order -- the two differ
// only in how `tree.ensureRootGen`/`tree.playoutGen`'s suspension points (if
// any) are driven, never in what gets computed.
function* searchGen(
  tree: MatchTree,
  state: MatchState,
  config: SearchConfig,
  opts: SearchOptions,
  timer: PhaseTimer | null,
): Generator<Promise<WdlEval>, MatchResult, WdlEval> {
  tree.clear();
  const nodesAtEntry = tree.nodeCount;
  const evaluationsBefore = tree.evaluations;

  tree.checkConsistency(state, opts.ply ?? -1);

  const legal = legalActionsOf(state);
  if (legal.length === 0) {
    throw new Error('match search on a state with no legal actions: there is no move to return');
  }
  if (state.isTerminal()) {
    throw new Error(
      `match search on a terminal state that reports ${legal.length} legal actions`,
    );
  }

  timer?.lap('setupMs');
  yield* tree.ensureRootGen(state);
  timer?.lap('rootEvalMs');
  const root = tree.root;
  if (root === null) {
    throw new Error('the root vanished between expansion and search');
  }

  const survey = opts.discovery ? new TacticalSurvey(opts.tacticalTransitionCap ?? DEFAULT_TRANSITION_CAP) : null;
  // Both passes are skipped wholesale inside the draft: the engine suppresses
  // every terminal while either side still has pieces to place (game.ts's
  // winState), so both scans are guaranteed to find nothing and the scan is
  // the most expensive thing the move would do.
  const drafting = survey !== null && draftIsOpen(state);
  if (survey !== null && !drafting) {
    const witness = survey.rootWin(state);
    timer?.lap('rootWinMs');
    if (witness !== null) {
      // A proven win, played without spending a playout on deciding whether
      // to. The record is the root's statistics as they stand -- one
      // evaluation, no descent.
      return buildResult(tree, config, root, {
        action: witness,
        playoutsUsed: 0,
        nodesAtEntry,
        evaluationsBefore,
        tactical: survey.record({ rootWitness: witness }),
        deadlineHit: false,
      }, timer);
    }
  }

  let playoutsUsed = 0;
  if (root.width > 1) {
    for (let i = 0; i < config.playouts; i++) {
      yield* tree.playoutGen(state);
      playoutsUsed += 1;
    }
  }
  timer?.lap('playoutsMs');

  // The move choice reads the same design §A2-corrected child utilities the
  // descent does.
  let index = selectAction(root, config, { bias: tree.bias });
  timer?.lap('selectMs');
  let tactical: MatchTactical | null = null;
  if (survey !== null && drafting) {
    tactical = survey.record({ skippedReason: SKIPPED_DRAFT });
  } else if (survey !== null) {
    const allowed = survey.refutationMask(state, root);
    // Three cases, and only the middle one is the filter doing anything.
    // Nothing refuted: the legacy choice verbatim. Everything refuted: the
    // legacy choice again -- a filter with nothing to offer must not invent
    // a preference.
    let filtered: number;
    if (allTrue(allowed)) {
      filtered = index;
    } else if (anyTrue(allowed)) {
      filtered = selectAction(root, config, { bias: tree.bias, allowed });
    } else {
      filtered = index;
    }
    tactical = survey.record({
      overridden: filtered !== index ? root.actions[index]! : null,
    });
    index = filtered;
    timer?.lap('surveyMs');
  }

  return buildResult(tree, config, root, {
    action: root.actions[index]!,
    playoutsUsed,
    nodesAtEntry,
    evaluationsBefore,
    tactical,
    deadlineHit: false,
  }, timer);
}

// KOM-1103: `ensureRootGen` is always single-leaf (the root is one
// evaluation, never a batch -- tree.ts has no batched root-ensure), so its
// generator protocol is `Generator<Promise<WdlEval>, void, WdlEval>`, one
// WdlEval at a time. `searchBatchedGen` below is `Generator<Promise<WdlEval[]>,
// ..., WdlEval[]>` throughout (its other suspension point, `playoutBatchGen`,
// is genuinely batched), so `yield* tree.ensureRootGen(state)` cannot be
// delegated directly -- the two generators disagree on what a yield/resume
// value even is. This relays the exact same promise/value through a
// length-1 array, never touching the WdlEval itself: for a synchronous
// evaluator `ensureRootGen` never actually suspends (isPromiseLike is
// false), so the loop body below never runs at all, byte-for-byte the same
// as before; for an async evaluator it adds one `.then` on the root's own
// evaluation only, never on a playout.
function* ensureRootForBatch(tree: MatchTree, state: MatchState): Generator<Promise<WdlEval[]>, void, WdlEval[]> {
  const gen = tree.ensureRootGen(state);
  let step = gen.next();
  while (!step.done) {
    const row = yield step.value.then((wdl) => [wdl]);
    step = gen.next(row[0]!);
  }
}

// Fast-mode U3 (design 1.3-1.8): searchGen's statement sequence around a
// batched playout loop. The root-win scan, refutation survey, three-way rule
// and DRAFTMIN are untouched -- only the `root.width > 1` playout loop is
// replaced, by `tree.playoutBatchGen` (U2).
//
// Deadline support (design 1.7 C3, 4.3, KOM-1220): with no `opts.shouldStop`,
// the whole budget is handed to ONE `playoutBatchGen` call (U2's own tested
// continuous semantics: a batch closes only at `batch` held leaves or the
// first collision, never on an artificial boundary). With `opts.shouldStop`
// set, the budget is walked in the SAME `batch`-sized chunks via repeated
// `playoutBatchGen(state, n, batch, ...)` calls (n = min(batch, playouts
// remaining), so the final chunk never overruns `config.playouts`).
// `shouldStop` is polled between chunks, never mid-evaluation (an in-flight
// evaluation cannot be cancelled, design 5.3), and only when playouts still
// remain -- so a clock that trips exactly as the budget is exhausted never
// sets `deadlineHit`.
// Exported (re-review F7): pondering (design 5.2-5.3, U7) drives this
// generator directly over its own scratch tree/bias copy, needing its own
// abortable driver rather than runBatchGenSync/Async.
export function* searchBatchedGen(
  tree: MatchTree,
  state: MatchState,
  config: SearchConfig,
  opts: SearchOptions,
  timer: PhaseTimer | null,
  batch: number,
): Generator<Promise<WdlEval[]>, MatchResult, WdlEval[]> {
  tree.clear();
  const nodesAtEntry = tree.nodeCount;
  const evaluationsBefore = tree.evaluations;

  tree.checkConsistency(state, opts.ply ?? -1);

  const legal = legalActionsOf(state);
  if (legal.length === 0) {
    throw new Error('match search on a state with no legal actions: there is no move to return');
  }
  if (state.isTerminal()) {
    throw new Error(
      `match search on a terminal state that reports ${legal.length} legal actions`,
    );
  }

  timer?.lap('setupMs');
  yield* ensureRootForBatch(tree, state);
  timer?.lap('rootEvalMs');
  const root = tree.root;
  if (root === null) {
    throw new Error('the root vanished between expansion and search');
  }

  const survey = opts.discovery ? new TacticalSurvey(opts.tacticalTransitionCap ?? DEFAULT_TRANSITION_CAP) : null;
  const drafting = survey !== null && draftIsOpen(state);
  if (survey !== null && !drafting) {
    const witness = survey.rootWin(state);
    timer?.lap('rootWinMs');
    if (witness !== null) {
      return buildResult(tree, config, root, {
        action: witness,
        playoutsUsed: 0,
        nodesAtEntry,
        evaluationsBefore,
        tactical: survey.record({ rootWitness: witness }),
        deadlineHit: false,
      }, timer);
    }
  }

  let playoutsUsed = 0;
  let deadlineHit = false;
  if (root.width > 1) {
    const overlay: Overlay = new Map();
    if (opts.shouldStop === undefined) {
      const stats = yield* tree.playoutBatchGen(state, config.playouts, batch, { overlay });
      playoutsUsed = stats.playouts;
    } else {
      while (playoutsUsed < config.playouts) {
        const n = Math.min(batch, config.playouts - playoutsUsed);
        const stats = yield* tree.playoutBatchGen(state, n, batch, { overlay });
        playoutsUsed += stats.playouts;
        opts.onProgress?.(playoutsUsed);
        if (playoutsUsed < config.playouts) {
          if (opts.yieldToHost) yield opts.yieldToHost();
          if (opts.shouldStop()) {
            deadlineHit = true;
            break;
          }
        }
      }
    }
  }
  timer?.lap('playoutsMs');

  let index = selectAction(root, config, { bias: tree.bias });
  timer?.lap('selectMs');
  let tactical: MatchTactical | null = null;
  if (survey !== null && drafting) {
    tactical = survey.record({ skippedReason: SKIPPED_DRAFT });
  } else if (survey !== null) {
    const allowed = survey.refutationMask(state, root);
    let filtered: number;
    if (allTrue(allowed)) {
      filtered = index;
    } else if (anyTrue(allowed)) {
      filtered = selectAction(root, config, { bias: tree.bias, allowed });
    } else {
      filtered = index;
    }
    tactical = survey.record({
      overridden: filtered !== index ? root.actions[index]! : null,
    });
    index = filtered;
    timer?.lap('surveyMs');
  }

  return buildResult(tree, config, root, {
    action: root.actions[index]!,
    playoutsUsed,
    nodesAtEntry,
    evaluationsBefore,
    tactical,
    deadlineHit,
  }, timer);
}

// KOM-1081: with a clock, the tree's evaluator is swapped for a timed
// wrapper for the duration of the call and always restored, error or not.
function startTimer(tree: MatchTree, opts: SearchOptions): { timer: PhaseTimer; inner: MaybeAsyncWdlEvaluator } | null {
  if (opts.clock === undefined) return null;
  const timer = new PhaseTimer(opts.clock);
  const inner = tree.evaluator;
  tree.evaluator = timer.wrap(inner);
  return { timer, inner };
}

// Fast-mode U3 (design 6.3 N-4): like startTimer, but also times the batch
// seam searchBatchedGen's playouts use. Each row of a DEFAULT batchEvaluator
// (tree.ts's `mapEvaluatorOverRows`, built from `evaluator` at construction
// time) reads `tree.evaluator` live at call time, so once this wraps BOTH
// seams a default batchEvaluator's rows would also re-enter the wrapped
// single-leaf seam and be charged twice. Callers that want a clock on the
// batched path must give the tree its own standalone `batchEvaluator` (as
// U6/U7 do, design 6.3 S-1) rather than relying on the default.
function startBatchTimer(
  tree: MatchTree,
  opts: SearchOptions,
): { timer: PhaseTimer; innerEvaluator: MaybeAsyncWdlEvaluator; innerBatchEvaluator: BatchWdlEvaluator } | null {
  if (opts.clock === undefined) return null;
  const timer = new PhaseTimer(opts.clock);
  const innerEvaluator = tree.evaluator;
  const innerBatchEvaluator = tree.batchEvaluator;
  tree.evaluator = timer.wrap(innerEvaluator);
  tree.batchEvaluator = timer.wrapBatch(innerBatchEvaluator);
  return { timer, innerEvaluator, innerBatchEvaluator };
}

export function search(tree: MatchTree, state: MatchState, config: SearchConfig, opts: SearchOptions): MatchResult {
  const timed = startTimer(tree, opts);
  try {
    return runGenSync(searchGen(tree, state, config, opts, timed === null ? null : timed.timer));
  } finally {
    if (timed !== null) tree.evaluator = timed.inner;
  }
}

// KOM-1078: the resumable entry point. Identical to `search()` apart from
// awaiting each evaluation as it is produced (runGenAsync), so a real
// per-leaf-async evaluator (onnxruntime-web) can be driven leaf by leaf
// without blocking and without redoing any earlier work.
export async function searchAsync(
  tree: MatchTree,
  state: MatchState,
  config: SearchConfig,
  opts: SearchOptions,
): Promise<MatchResult> {
  const timed = startTimer(tree, opts);
  try {
    return await runGenAsync(searchGen(tree, state, config, opts, timed === null ? null : timed.timer));
  } finally {
    if (timed !== null) tree.evaluator = timed.inner;
  }
}

// Fast-mode U3 (design 1.7): the dispatch. `batch === 1` with no deadline
// routes to the existing `search()` untouched; every other case (batch > 1,
// or any batch with a deadline, design 1.7 C3) runs `searchBatchedGen`.
export function searchFast(
  tree: MatchTree,
  state: MatchState,
  config: SearchConfig,
  opts: SearchOptions,
  batch: number,
): MatchResult {
  if (batch === 1 && opts.shouldStop === undefined) {
    return search(tree, state, config, opts);
  }
  const timed = startBatchTimer(tree, opts);
  try {
    return runBatchGenSync(searchBatchedGen(tree, state, config, opts, timed === null ? null : timed.timer, batch));
  } finally {
    if (timed !== null) {
      tree.evaluator = timed.innerEvaluator;
      tree.batchEvaluator = timed.innerBatchEvaluator;
    }
  }
}

// KOM-1078-style async counterpart of searchFast: identical dispatch, awaits
// each yielded batch instead of requiring a synchronous one.
export async function searchFastAsync(
  tree: MatchTree,
  state: MatchState,
  config: SearchConfig,
  opts: SearchOptions,
  batch: number,
): Promise<MatchResult> {
  if (batch === 1 && opts.shouldStop === undefined) {
    return searchAsync(tree, state, config, opts);
  }
  const timed = startBatchTimer(tree, opts);
  try {
    return await runBatchGenAsync(
      searchBatchedGen(tree, state, config, opts, timed === null ? null : timed.timer, batch),
    );
  } finally {
    if (timed !== null) {
      tree.evaluator = timed.innerEvaluator;
      tree.batchEvaluator = timed.innerBatchEvaluator;
    }
  }
}
