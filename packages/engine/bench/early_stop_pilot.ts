// KOM-1219 unit 2a (docs/design/kom1219-adaptive-time.md section 6.1): an
// OFFLINE measurement of section 3's early-stop rule. No `src` change -- this
// file carries its own reference implementation of `isMoveDecided` (3.1/3.3)
// and drives `searchFastAsync` directly with a `shouldStop` that always
// returns false, so the full search always runs to completion; at every
// chunk boundary (the same ones section 5 would hook) it checks whether the
// rule would have fired and records the first such checkpoint. Soundness is
// then: did the full search's final action equal the action certified at
// that first-fire checkpoint.

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

import { fakeLogits, fakeWdl, fnv1a32, makeFakeMlhEvaluator } from '../src/fake_evaluator.ts';
import { boardFromFen, boardToFen } from '../src/fen.ts';
import { ADVANCED_POSITION } from '../src/constants.ts';
import { Game } from '../src/game.ts';
import { generateRecords } from '../src/movegen.ts';
import { canonicalMove, RepetitionRule } from '../src/repetition.ts';
import { LEVELS, LEVEL_NAMES } from '../src/levels.ts';
import { MlhContext, wrapMlhEvaluator } from '../src/search/mlh.ts';
import type { MlhConfig } from '../src/search/mlh.ts';
import { quadraticForm } from '../src/search/node.ts';
import type { MatchNode } from '../src/search/node.ts';
import { childRawUtilities, lcbEligible, selectAction } from '../src/search/select.ts';
import type { BiasReader, MatchConfig as SelectConfig } from '../src/search/select.ts';
import { DEFAULT_TRANSITION_CAP, TacticalSurvey, searchFastAsync } from '../src/search/search.ts';
import type { SearchConfig, SearchOptions } from '../src/search/search.ts';
import { MatchGameState } from '../src/search/state.ts';
import type { MatchState, WdlEval } from '../src/search/state.ts';
import { MatchTree } from '../src/search/tree.ts';
import { loadSearchFixture } from '../test/fixture.ts';

// ---- the reference rule (design section 3) ---------------------------------

// isMoveDecided (3.1 Case A + the eligibility-stability check (i)): pure --
// reads only root/children/bias, never mutates, never calls bias.generation().
// Returns the certified action and its root-edge index, or null.
type Certified = { action: number; index: number };

function certifiedAction(root: MatchNode, config: SelectConfig, bias: BiasReader, R: number): Certified | null {
  const width = root.width;
  const ellIndex = selectAction(root, config, { bias });
  const eligible = lcbEligible(root, config);
  if (!eligible[ellIndex]) return null;

  const weight = root.weight;
  let bestNow = 0;
  for (let i = 0; i < width; i++) if (weight[i]! > bestNow) bestNow = weight[i]!;
  const p = config.min_visit_prop_for_lcb;
  const threshold = Math.max(2, p * bestNow);
  const wEll = weight[ellIndex]!;

  // Review F5: a challenger that cannot become eligible whatever happens is
  // skipped from (ii) below.
  const skipped: boolean[] = new Array(width).fill(false);
  for (let j = 0; j < width; j++) {
    if (j === ellIndex) continue;
    if (weight[j]! + R < threshold) skipped[j] = true;
  }

  // Eligibility stability (i): ell must stay eligible for EVERY split,
  // against every j != ell, including the skipped ones.
  for (let j = 0; j < width; j++) {
    if (j === ellIndex) continue;
    if (wEll < p * (weight[j]! + R)) return null;
  }

  const c = config.contempt;
  const umax = Math.max(1, c);
  const umin = Math.min(-1, c);
  const D = umax - umin;
  const m = Math.max(1, Math.abs(c));
  const lambda = config.subtree_bias_factor;
  const varFloor2 = config.var_floor * config.var_floor;
  const plugin = config.variance_mode === 'plugin';
  const k = config.k_lcb;

  const rawUtil = childRawUtilities(root, config);
  const qf = quadraticForm(root.moment, c);

  // Bias-term bound (review F3): 0 on a terminal child, lambda*D on an
  // unvisited edge (no child, or weight 0 -- the two coincide, 3's own note),
  // else lambda*D*(w-T)/w.
  const biasBoundFor = (i: number): number => {
    const child = root.children.get(i);
    if (child === undefined) return lambda * D;
    if (child.isTerminal) return 0;
    const wi = weight[i]!;
    const deficit = wi - child.totalWeight();
    return (lambda * D * deficit) / wi;
  };

  const Sell = rawUtil[ellIndex]! * wEll;
  const Qell = qf[ellIndex]!;
  const Bell = biasBoundFor(ellIndex);

  // F_ell = min over integer r in [0, R] of F_ell(r) -- O(R) per checkpoint.
  let Fell = Infinity;
  for (let r = 0; r <= R; r++) {
    const denom = wEll + r;
    const mean = (Sell + r * umin) / denom;
    const variance = plugin ? Math.max(varFloor2, m * m) : Math.max(varFloor2, (Qell + r * m * m) / denom);
    const f = mean - Bell - k * Math.sqrt(variance / denom);
    if (f < Fell) Fell = f;
  }

  const EPS = 1e-9;
  for (let j = 0; j < width; j++) {
    if (j === ellIndex || skipped[j]) continue;
    const Sj = rawUtil[j]! * weight[j]!;
    const Bj = biasBoundFor(j);
    const denomJ = weight[j]! + R;
    const Lceil = (Sj + R * umax) / denomJ + Bj - k * (config.var_floor / Math.sqrt(denomJ));
    if (!(Fell > Lceil + EPS)) return null;
  }

  return { action: root.actions[ellIndex]!, index: ellIndex };
}

// draftIsOpen (search.ts's own private helper, re-derived here since unit 2a
// may not touch src): whether either side still holds drafting rights.
function draftIsOpenRef(state: MatchState): boolean {
  const board = state.game.board;
  return Boolean(board.draftW) || Boolean(board.draftB);
}

function allTrueMask(mask: Uint8Array): boolean {
  for (let i = 0; i < mask.length; i++) if (!mask[i]) return false;
  return true;
}

function anyTrueMask(mask: Uint8Array): boolean {
  for (let i = 0; i < mask.length; i++) if (mask[i]) return true;
  return false;
}

function legalActionsOfMask(mask: Uint8Array): Int32Array {
  let count = 0;
  for (let i = 0; i < mask.length; i++) if (mask[i]) count++;
  const out = new Int32Array(count);
  let k = 0;
  for (let i = 0; i < mask.length; i++) if (mask[i]) out[k++] = i;
  return out;
}

// ---- a MatchState that remembers which root edge its own descent took -----
//
// The decisive evaluator (below) needs to know, for a leaf anywhere in the
// tree, which ROOT action its descent started with -- information no
// MatchState exposes. Rather than touch src, this bench file supplies its
// own MatchState implementation that tags itself with the root action the
// first play() (at depth 0) took, and carries that tag through clone() --
// exactly the information a pending leaf's `simulation` snapshot needs.
class TrackedState implements MatchState {
  private readonly inner: MatchGameState;
  rootAction: number | null;

  constructor(inner: MatchGameState, rootAction: number | null = null) {
    this.inner = inner;
    this.rootAction = rootAction;
  }

  get game(): Game {
    return this.inner.game;
  }

  legalMask(): Uint8Array {
    return this.inner.legalMask();
  }

  play(action: number): void {
    const wasRoot = this.inner.isRoot;
    this.inner.play(action);
    if (wasRoot && this.rootAction === null) this.rootAction = action;
  }

  undo(): void {
    this.inner.undo();
    if (this.inner.isRoot) this.rootAction = null;
  }

  clone(): TrackedState {
    return new TrackedState(this.inner.clone(), this.rootAction);
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

// ---- the two built-in evaluators -------------------------------------------

// Standard: the committed fake MLH evaluator (fake_evaluator.ts/mlh.ts), same
// combinator every parity test already builds.
function buildStandardEvaluator(numActions: number, mlhConfig: MlhConfig): (state: MatchState) => WdlEval {
  const ctx = new MlhContext();
  const leaf = makeFakeMlhEvaluator(numActions);
  return wrapMlhEvaluator(leaf, mlhConfig, ctx);
}

// Decisive: deterministic, favours one root action by position hash (design
// 6.1). On the first call (always the root itself -- tree.ts evaluates the
// root before any descent), picks favoredAction = legal[hash % legal.length]
// and remembers the root's own mover. On every later call (a leaf, always a
// TrackedState here), returns a near-extreme WDL in the LEAF's own mover
// frame: a win for whichever side the favoured line actually helps.
class DecisiveContext {
  favoredAction: number | null = null;
  rootToPlay = 0;
}

function buildDecisiveEvaluator(numActions: number): (state: MatchState) => WdlEval {
  const ctx = new DecisiveContext();
  return (state: MatchState): WdlEval => {
    const h = fnv1a32(state.placementKeyBytes());
    if (ctx.favoredAction === null) {
      const legal = legalActionsOfMask(state.legalMask());
      if (legal.length === 0) throw new Error('decisive evaluator: root reported no legal action');
      ctx.favoredAction = legal[h % legal.length]!;
      ctx.rootToPlay = state.toPlay();
      return { policyLogits: fakeLogits(h, numActions), wdl: fakeWdl(h) };
    }
    const tracked = state as TrackedState;
    const favored = tracked.rootAction === ctx.favoredAction;
    const sameMover = state.toPlay() === ctx.rootToPlay;
    const winForMover = favored === sameMover;
    const wdl = winForMover
      ? Float64Array.from([0.95, 0.03, 0.02])
      : Float64Array.from([0.02, 0.03, 0.95]);
    return { policyLogits: fakeLogits(h, numActions), wdl };
  };
}

// ---- the 16-position corpus (section 6) ------------------------------------

type SearchRecordConfig = SearchConfig & {
  num_actions: number;
  discovery: boolean;
  draftmin: number;
  mlh: MlhConfig;
};

type SearchRecord = { config: SearchRecordConfig; position: { fen: string; stamp: string } };
type SearchFixture = { searches: SearchRecord[] };

type Position = { id: string; build: () => TrackedState };

function rootStateForFixture(rec: SearchRecord): MatchGameState {
  const board = boardFromFen(rec.position.fen);
  const rule = rec.position.stamp === 'present' ? RepetitionRule.PLACEMENT : RepetitionRule.FULL;
  const game = new Game(board, rule);
  return new MatchGameState(game, { minPlacements: 0, isRoot: true });
}

// The first 12 KOM_SEARCH positions in fixture order, via loadSearchFixture()
// (never a raw gzip read).
function loadFixturePositions(): { positions: Position[]; baseConfig: SearchRecordConfig } {
  const fixture = loadSearchFixture<SearchFixture>();
  const records = fixture.searches.slice(0, 12);
  if (records.length < 12) {
    throw new Error(`expected >= 12 KOM_SEARCH positions, got ${records.length}`);
  }
  const baseConfig = records[0]!.config;
  const positions = records.map((rec, i) => ({
    id: `kom_search#${i}`,
    build: () => new TrackedState(rootStateForFixture(rec)),
  }));
  return { positions, baseConfig };
}

type SeededGame = { plies: string[]; ligungiTerminal: { winner: string | null } };
type SeededFixture = { schema: string; method: string; games: Record<string, SeededGame> };

function loadSeededFixture(): SeededFixture {
  const path = fileURLToPath(new URL('../test/fixtures/ligungi_seeded_games.json', import.meta.url));
  return JSON.parse(readFileSync(path, 'utf8')) as SeededFixture;
}

// Replays `plies` (the recorded search fixture's canonicalMove strings, the
// same vocabulary repetition.ts's canonicalMove produces) from the ADVANCED
// start under the 'ligungi' profile, collecting the FEN after every ply once
// both drafts have finished and before any terminal ply.
function postDraftFensFor(plies: readonly string[]): string[] {
  const game = new Game(boardFromFen(ADVANCED_POSITION), RepetitionRule.FULL, 1, 'ligungi');
  const fens: string[] = [];
  for (const ply of plies) {
    const records = generateRecords(game.board, 'ligungi');
    const found = records.find((r) => canonicalMove(r.struct) === ply);
    if (found === undefined) throw new Error(`no legal move canonicalises to ${JSON.stringify(ply)}`);
    game.play(found.id);
    const [w, b, d] = game.winState();
    if (w || b || d) break;
    if (!game.board.draftW && !game.board.draftB) fens.push(boardToFen(game.board));
  }
  return fens;
}

function pickEvenly<T>(items: readonly T[], count: number): T[] {
  if (items.length <= count) return items.slice();
  const out: T[] = [];
  for (let i = 0; i < count; i++) {
    const idx = Math.min(items.length - 1, Math.floor(((i + 1) * items.length) / (count + 1)));
    out.push(items[idx]!);
  }
  return out;
}

// 4 post-draft plies of ligungi_seeded_games.json: 2 evenly-spaced post-draft
// positions from each of its 2 games.
function loadSeededPositions(): Position[] {
  const fixture = loadSeededFixture();
  const positions: Position[] = [];
  let gi = 0;
  for (const g of Object.values(fixture.games)) {
    const fens = pickEvenly(postDraftFensFor(g.plies), 2);
    fens.forEach((fen, i) => {
      positions.push({
        id: `ligungi_seeded#${gi}.${i}`,
        build: () =>
          new TrackedState(
            new MatchGameState(new Game(boardFromFen(fen), RepetitionRule.FULL, 1, 'ligungi'), {
              minPlacements: 0,
              isRoot: true,
            }),
          ),
      });
    });
    gi++;
  }
  return positions;
}

// ---- the 11 (playouts, batch) pairs in LEVELS ------------------------------

type Pair = { level: string; playouts: number; batch: number };

function buildPairs(): Pair[] {
  const pairs: Pair[] = [];
  for (const name of LEVEL_NAMES) {
    const def = LEVELS[name];
    const seen = new Set<number>();
    for (const batch of [def.batch.webgpu, def.batch.wasmMt, def.batch.wasm1t]) {
      if (seen.has(batch)) continue;
      seen.add(batch);
      pairs.push({ level: name, playouts: def.playouts, batch });
    }
  }
  return pairs;
}

function withPlayouts(base: SearchRecordConfig, playouts: number): SearchRecordConfig {
  const next = { ...base };
  next.playouts = playouts;
  return next;
}

// ---- one offline run: full search, checkpoint-checking the rule -----------

type RunStats = {
  fired: boolean;
  savedFraction: number;
  soundnessViolation: boolean;
  playoutsUsed: number;
  ms: number;
};

type EvaluatorKind = 'standard' | 'decisive';

async function runOne(
  position: Position,
  pair: Pair,
  baseConfig: SearchRecordConfig,
  mlhConfig: MlhConfig,
  evaluatorKind: EvaluatorKind,
): Promise<RunStats> {
  const config = withPlayouts(baseConfig, pair.playouts);
  const evaluator =
    evaluatorKind === 'standard'
      ? buildStandardEvaluator(config.num_actions, mlhConfig)
      : buildDecisiveEvaluator(config.num_actions);
  const tree = new MatchTree(config, evaluator);
  const state = position.build();
  // The pilot's OWN survey instance (design 6.1): the real search's internal
  // survey, if any, is never touched by this.
  const survey = new TacticalSurvey(DEFAULT_TRANSITION_CAP);
  let allowedCache: Uint8Array | null = null;
  let fired = false;
  let firedAt = 0;
  let firedAction = -1;

  const opts: SearchOptions = {
    discovery: true,
    shouldStop: () => {
      if (fired) return false;
      const root = tree.root;
      if (root === null || root.width <= 1) return false;
      const used = root.totalWeight();
      const R = config.playouts - used;
      if (R < 1) return false;
      const certified = certifiedAction(root, config, tree.bias, R);
      if (certified === null) return false;
      if (draftIsOpenRef(state)) {
        fired = true;
        firedAt = used;
        firedAction = certified.action;
        return false;
      }
      if (allowedCache === null) allowedCache = survey.refutationMask(state, root);
      const allowed = allowedCache;
      if (allTrueMask(allowed) || !anyTrueMask(allowed) || allowed[certified.index]) {
        fired = true;
        firedAt = used;
        firedAction = certified.action;
      }
      return false;
    },
  };

  const t0 = performance.now();
  const result = await searchFastAsync(tree, state, config, opts, pair.batch);
  const ms = performance.now() - t0;

  const savedFraction = fired ? (config.playouts - firedAt) / config.playouts : 0;
  const soundnessViolation = fired && firedAction !== result.action;
  return { fired, savedFraction, soundnessViolation, playoutsUsed: result.playoutsUsed, ms };
}

// ---- orchestration + report -------------------------------------------------

type RunRecord = {
  positionId: string;
  level: string;
  batch: number;
  evaluator: EvaluatorKind;
  fired: boolean;
  savedFraction: number;
  soundnessViolation: boolean;
  playoutsUsed: number;
  ms: number;
};

async function main(): Promise<void> {
  const { positions: fixturePositions, baseConfig } = loadFixturePositions();
  const seededPositions = loadSeededPositions();
  const positions = [...fixturePositions, ...seededPositions];
  if (positions.length !== 16) {
    console.error(`WARNING: expected 16 corpus positions, got ${positions.length}`);
  }
  const pairs = buildPairs();
  const evaluators: EvaluatorKind[] = ['standard', 'decisive'];
  const mlhConfig: MlhConfig = baseConfig.mlh;

  const records: RunRecord[] = [];
  for (const evaluatorKind of evaluators) {
    for (const pair of pairs) {
      for (const position of positions) {
        const stats = await runOne(position, pair, baseConfig, mlhConfig, evaluatorKind);
        records.push({
          positionId: position.id,
          level: pair.level,
          batch: pair.batch,
          evaluator: evaluatorKind,
          fired: stats.fired,
          savedFraction: stats.savedFraction,
          soundnessViolation: stats.soundnessViolation,
          playoutsUsed: stats.playoutsUsed,
          ms: stats.ms,
        });
      }
    }
  }

  console.log(
    `corpus: ${positions.length} positions (${fixturePositions.length} KOM_SEARCH + ${seededPositions.length} ligungi seeded)`,
  );
  console.log('level      batch  evaluator  fireRate  meanSaving  perPlayoutMs  soundnessViolations');
  for (const pair of pairs) {
    for (const evaluatorKind of evaluators) {
      const list = records.filter(
        (r) => r.level === pair.level && r.batch === pair.batch && r.evaluator === evaluatorKind,
      );
      const fireRate = list.filter((r) => r.fired).length / list.length;
      const meanSaving = list.reduce((s, r) => s + r.savedFraction, 0) / list.length;
      const totalPlayouts = list.reduce((s, r) => s + r.playoutsUsed, 0);
      const totalMs = list.reduce((s, r) => s + r.ms, 0);
      const perPlayoutMs = totalPlayouts > 0 ? totalMs / totalPlayouts : 0;
      const violations = list.filter((r) => r.soundnessViolation).length;
      console.log(
        `${pair.level.padEnd(10)} ${String(pair.batch).padStart(5)}  ${evaluatorKind.padEnd(9)}  ` +
          `${fireRate.toFixed(3)}     ${meanSaving.toFixed(4)}      ${perPlayoutMs.toFixed(4)}        ${violations}`,
      );
    }
  }

  const soundness: Record<string, number> = {};
  for (const evaluatorKind of evaluators) {
    soundness[evaluatorKind] = records.filter((r) => r.evaluator === evaluatorKind && r.soundnessViolation).length;
  }

  const saving: Record<string, number> = {};
  for (const name of LEVEL_NAMES) {
    const list = records.filter((r) => r.level === name);
    saving[name] = list.length > 0 ? list.reduce((s, r) => s + r.savedFraction, 0) / list.length : 0;
  }

  const decisiveFireCounts: Record<string, number> = {};
  for (const pair of pairs) {
    const key = `${pair.level}@${pair.batch}`;
    const list = records.filter(
      (r) => r.evaluator === 'decisive' && r.level === pair.level && r.batch === pair.batch,
    );
    decisiveFireCounts[key] = list.filter((r) => r.fired).length;
  }

  const totalPlayouts = records.reduce((s, r) => s + r.playoutsUsed, 0);
  const totalMs = records.reduce((s, r) => s + r.ms, 0);
  const perPlayoutMs = totalPlayouts > 0 ? totalMs / totalPlayouts : 0;

  console.log(
    `per-playout wall cost: ${perPlayoutMs.toFixed(4)} ms (${totalPlayouts} playouts, ${totalMs.toFixed(1)} ms)`,
  );
  console.log(`decisive fire counts by (level,batch): ${JSON.stringify(decisiveFireCounts)}`);

  const summary = { soundness, saving, perPlayoutMs, decisiveFireCounts };
  console.log(`PILOT_SUMMARY ${JSON.stringify(summary)}`);
}

main().catch((err: unknown) => {
  console.error(err);
  process.exitCode = 1;
});
