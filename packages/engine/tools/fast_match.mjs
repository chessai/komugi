// KOM-1097 (design doc docs/design/clientside-fast-mode.md section 7 +
// section 8 "Unit U9"; re-review docs/reviews/2026-11-12-clientside-fast-mode-re-review.md
// F5, C1). A Node match runner comparing two WorkerEngine instances -- exact
// handle() vs handleFast(batch) -- over paired seeded-draft games with
// colours swapped, so draft luck cancels across a pair. Reports a pentanomial
// SPRT LLR (section 7's stopping rules) plus W/D/L, Elo with a 95% interval.
//
// REFUSES a real-model match (`--model` given) unless `--approved` is also
// passed, checked before anything else runs -- no model load, no game played.
// This worker only ever exercises the fake evaluator (gated by
// KOM-1097's hard rules); `--approved` real-model support exists for the
// user's own go-ahead runs (design section 7's "needs the user's go-ahead")
// and is otherwise dead code from this ticket's own tests' point of view.
//
// Usage:
//   node fast_match.mjs --out games.jsonl --pairs N --sims S [--batch B]
//     [--fast-sims S] [--exact-sims S] [--seed N] [--workers N]
//     [--elo0 E] [--elo1 E] [--alpha A] [--beta B] [--min-pairs N]
//     [--max-plies N] [--model path.onnx --approved] [--plane-map-version 3|4]
//   node fast_match.mjs --emit-llr '{"counts":[...],"elo0":-30,"elo1":0,"alpha":0.05,"beta":0.05}'
//   node fast_match.mjs --emit-equal-time-sims '{"table":{...},"backend":"webgpu","batch":8}'
//
// `--plane-map-version` (KOM-1168, default 3): threaded onto every request
// and every MatchGameState this harness builds itself; the real-model
// evaluators (makeEvaluatorsFromSession) encode each row at its own state's
// version (src/host_encode.ts), not an assumed v3.
//
// F5 fix: this tool never "skips" on a missing dependency -- every failure
// path is a non-zero exit with a stderr message naming this file, and the
// Python test asserts both `returncode == 0` and the absence of any
// `unittest`-style skip marker, never an exit-0 "nothing to check" path.

import { appendFileSync, writeFileSync } from 'node:fs';
import { cpus } from 'node:os';
import { pathToFileURL } from 'node:url';

import { ACTION_SIZE, ADVANCED_POSITION } from '../src/constants.ts';
import { decodeAction } from '../src/actions.ts';
import { runOrtBatch } from '../src/batch_evaluator.ts';
import { makeFakeMlhEvaluator } from '../src/fake_evaluator.ts';
import { loadOrtSession } from '../test/support/ort_node.ts';
import { boardFromFen, boardToFen } from '../src/fen.ts';
import { Game, MaxTurns } from '../src/game.ts';
import { encodeState, encodeStatesBatch } from '../src/host_encode.ts';
import { observationSize } from '../src/observation.ts';
import { RepetitionRule } from '../src/repetition.ts';
import { parsePlaneMapVersion } from '../src/rule_features.ts';
import { buildSan } from '../src/san.ts';
import { MatchGameState } from '../src/search/state.ts';
import { WorkerEngine } from '../src/engine.ts';

// KOM-1168: resolves `--plane-map-version` the same way e2e_driver.mjs's
// own CLI flags are resolved -- `undefined` defaults to 3, anything else
// throws parsePlaneMapVersion's exact message.
export function resolvePlaneMapVersion(args) {
  const raw = args['plane-map-version'];
  return parsePlaneMapVersion(raw === undefined ? undefined : Number(raw));
}

// ---------------------------------------------------------------------------
// Pentanomial SPRT (design section 7): a GSPRT over paired games. Each pair
// contributes one observation, the pair's average score per game (`i / 4`
// for a doubled pair-score `i` in 0..4, since each game scores 0/0.5/1 and a
// pair sums two games). `elo0`/`elo1` map to expected per-game scores via the
// standard logistic elo<->score relation; the LLR is the Gaussian
// log-likelihood-ratio of the sample mean against those two hypothesised
// means, Wald's SPRT bounds `log((1-beta)/alpha)` / `log(beta/(1-alpha))`
// (alpha=beta=0.05 gives +-2.944, matching section 7's "LLR bounds about
// +-2.94"). This is this ticket's own implementation (no existing pentanomial
// code in the repo to port); test_llr_matches_hand_computed pins its
// arithmetic against hand-computed values of this exact formula.
// ---------------------------------------------------------------------------

const ELO_DIVISOR = 400;

export function eloToScore(elo) {
  return 1 / (1 + Math.pow(10, -elo / ELO_DIVISOR));
}

export function scoreToElo(score) {
  const clipped = Math.min(Math.max(score, 1e-9), 1 - 1e-9);
  return -ELO_DIVISOR * Math.log10(1 / clipped - 1);
}

// `counts[i]` = number of pairs whose doubled fast-side score equals `i`
// (i in 0..4 <-> pair score 0, 0.5, 1, 1.5, 2).
export function pentanomialStats(counts) {
  const n = counts.reduce((a, b) => a + b, 0);
  if (n === 0) return { n: 0, mu: 0, variance: 0 };
  let mu = 0;
  for (let i = 0; i < 5; i++) mu += counts[i] * (i / 4);
  mu /= n;
  let variance = 0;
  for (let i = 0; i < 5; i++) variance += counts[i] * (i / 4 - mu) ** 2;
  variance /= n;
  return { n, mu, variance };
}

export function llrPentanomial(counts, elo0, elo1) {
  const { n, mu, variance } = pentanomialStats(counts);
  if (n === 0 || variance === 0) return 0;
  const s0 = eloToScore(elo0);
  const s1 = eloToScore(elo1);
  return (n / (2 * variance)) * ((mu - s0) ** 2 - (mu - s1) ** 2);
}

export function sprtBounds(alpha, beta) {
  return { upper: Math.log((1 - beta) / alpha), lower: Math.log(beta / (1 - alpha)) };
}

const Z_95 = 1.959963984540054; // standard normal 97.5th percentile

export function eloInterval(counts) {
  const { n, mu, variance } = pentanomialStats(counts);
  if (n === 0) return { elo: null, lo: null, hi: null };
  const se = Math.sqrt(variance / n);
  return { elo: scoreToElo(mu), lo: scoreToElo(mu - Z_95 * se), hi: scoreToElo(mu + Z_95 * se) };
}

// KOM-1081's measured CPU-per-sim range (section 1.9, 4.2): 0.25-1.0 ms,
// independent of backend.
const CPU_PER_SIM_MS = { min: 0.25, max: 1.0 };

// C1 fix (design section 7): r(backend) = exact-on-backend / fast-on-backend,
// both measured on the SAME backend's net ms-per-position table (the probe's
// own `msPerPosition` shape, design 3.2), offset by the same CPU-per-sim
// range at both ends -- never a cross-backend ratio. `table` is keyed by
// backend name, each entry a map from bucket size to net ms/position.
// Returns the achievable range of the speed ratio `r`; `r` is largest at the
// smallest CPU offset and smallest at the largest offset, since both
// batch-1 and batch-B costs move by the same additive `c` while batch-1's
// net cost is always larger (that is what makes batching worth it at all).
export function equalTimeSims(table, backend, batch) {
  const net = table[backend];
  if (!net || net[1] === undefined || net[batch] === undefined) {
    throw new Error(
      `equalTimeSims: table[${backend}] is missing msPerPosition for batch 1 or batch ${batch}`,
    );
  }
  const net1 = net[1];
  const netB = net[batch];
  const rMax = (net1 + CPU_PER_SIM_MS.min) / (netB + CPU_PER_SIM_MS.min);
  const rMin = (net1 + CPU_PER_SIM_MS.max) / (netB + CPU_PER_SIM_MS.max);
  return { rMin, rMax };
}

// ---------------------------------------------------------------------------
// Seeded draft and full-game play.
// ---------------------------------------------------------------------------

// mulberry32: a small, public-domain, deterministic PRNG -- all this tool
// needs (seeded per pair so a draft's randomness is reproducible and the
// paired game's colour swap sees the IDENTICAL opening).
function mulberry32(seed) {
  let a = seed >>> 0;
  return function rng() {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// A guard against an unbounded loop if the draft rules ever produced no
// legal "close the draft" move -- should never fire (each side's hand is
// finite), but a thrown error is strictly better than an infinite loop.
const DRAFT_GUARD_PLIES = 400;

// Plays a uniformly-random legal action each ply, seeded, from
// ADVANCED_POSITION (root.ts's own from-scratch draftable start -- the one
// base `gameFromHistory` replays history against), until neither side holds
// drafting rights. The random policy decides the whole opening, including
// when each side ends its own draft, so the engines never influence it:
// "draft luck cancels" across a colour-swapped pair sharing this same
// opening.
function buildSeededOpening(seed) {
  const rng = mulberry32(seed);
  const game = new Game(boardFromFen(ADVANCED_POSITION), RepetitionRule.PLACEMENT);
  const history = [];
  let guard = 0;
  while (game.board.draftW || game.board.draftB) {
    if (++guard > DRAFT_GUARD_PLIES) {
      throw new Error(`fast_match.mjs: seeded draft (seed=${seed}) did not close within ${DRAFT_GUARD_PLIES} plies`);
    }
    const mask = game.validMoves();
    const legal = [];
    for (let i = 0; i < mask.length; i++) if (mask[i]) legal.push(i);
    if (legal.length === 0) {
      throw new Error(`fast_match.mjs: no legal moves during seeded draft (seed=${seed})`);
    }
    const action = legal[Math.floor(rng() * legal.length)];
    const { struct } = decodeAction(game.board, action);
    const san = buildSan(game.board, struct);
    game.play(action);
    history.push(san);
  }
  return { game, history };
}

// Which role (`'fast'` or `'exact'`) controls colour 0 (white) and colour 1
// (black) in one game of a pair; game 0 and game 1 of a pair swap it, so
// draft/colour luck cancels over the pair (section 7).
function colorToRoleFor(gameIndexInPair) {
  return gameIndexInPair === 0 ? ['fast', 'exact'] : ['exact', 'fast'];
}

// Plays one full game from a shared seeded opening to a terminal result,
// alternating `engines.exact.handle()` and `engines.fast.handleFast()` by
// colour. Never throws on a normal game outcome; an engine `{err}` reply
// (a real divergence between this harness's own tracked Game and the
// engine's from-scratch history replay) is a hard failure -- there is no
// sensible score to assign a position the engine itself refused.
// `maxPlies` (undefined by default): a CI-speed cap on how many plies past
// the opening this one game plays before it is scored as a forced draw,
// independent of the engine's own 512-ply MaxTurns cap (game.ts). Weak
// sims-4 fake-evaluator play rarely reaches a natural terminal position
// quickly and otherwise runs close to MaxTurns every game (hundreds of
// search calls at ~0.3-0.5s each) -- section 7's real matches leave this
// unset and run to a real result or MaxTurns, as designed.
async function playGame(opening, colorToRole, engines, fastBatchEvaluator, sims, fastSims, exactSims, fastBatch, maxPlies, planeMapVersion) {
  const game = opening.game.clone();
  const state = new MatchGameState(game, { isRoot: false, planeMapVersion });
  const history = opening.history.slice();
  const startPly = history.length;
  let ply = startPly;
  let forcedStop = false;

  while (!state.isTerminal()) {
    if (maxPlies !== undefined && ply - startPly >= maxPlies) {
      forcedStop = true;
      break;
    }
    const mover = state.toPlay();
    const role = colorToRole[mover];
    const fen = boardToFen(game.board);
    const request = { id: ply, fen, history, sims: role === 'fast' ? fastSims : exactSims, planeMapVersion };
    // `handleAsync` (not `handle`): the exact engine's leaf evaluator is
    // synchronous for the fake path but a genuinely async ORT call for the
    // real-model path (worker.ts's own note: wrapMlhEvaluatorAsync accepts
    // either), so one call shape works for both without branching on which
    // evaluator is installed.
    const reply =
      role === 'fast'
        ? await engines.fast.handleFast({ ...request, batch: fastBatch }, fastBatchEvaluator)
        : await engines.exact.handleAsync(request);
    if ('err' in reply) {
      throw new Error(`fast_match.mjs: ${role} engine refused at ply ${ply}: ${reply.err}`);
    }
    state.play(reply.action);
    history.push(reply.san);
    ply += 1;
  }

  // A forced stop (the `maxPlies` CI cap, never section 7's real matches)
  // scores as a draw rather than calling `winState()` on a non-terminal
  // position, which is undefined on an unfinished game.
  const outcome = forcedStop ? [0, 0, 1] : game.winState(); // [white, black, draw], one-hot
  return { history, outcome, plies: game.turns, plyCapHit: forcedStop || game.turns >= MaxTurns };
}

// One game's outcome -> the fast engine's score (1/0.5/0) given which colour
// it played.
function fastScoreOf(outcome, fastColor) {
  return outcome[fastColor] + 0.5 * outcome[2];
}

// ---------------------------------------------------------------------------
// Real-model wiring (section 7's "on the real gen-11 ONNX model... through
// onnxruntime-web wasm"), reached only behind `--model` + `--approved`: this
// unit's own tests never exercise it (KOM-1097's hard rule forbids this
// worker from running one), but the design's harness needs both the fake and
// the real path. Mirrors `web/engine/tools/e2e_driver.mjs`'s own
// ORT-wasm wiring.
// ---------------------------------------------------------------------------

// KOM-1168: each row is encoded at ITS OWN state's planeMapVersion
// (src/host_encode.ts), not assumed v3. Split from `makeRealEvaluators` so
// a test can drive it with a fake session/ort, no real model.
export function makeEvaluatorsFromSession(session, ort, obsShape) {
  const rowSize = obsShape.reduce((a, b) => a * b, 1);
  const leaf = async (state) => (await runOrtBatch(session, ort, encodeState(state), 1, obsShape))[0];
  const batch = async (states) => runOrtBatch(session, ort, encodeStatesBatch(states, rowSize), states.length, obsShape);
  return { leaf, batch };
}

async function makeRealEvaluators(modelPath) {
  const { ort, session } = await loadOrtSession(modelPath, { threads: 1 });
  return makeEvaluatorsFromSession(session, ort, observationSize());
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

function fail(message) {
  process.stderr.write(`fast_match.mjs: ${message}\n`);
  process.exit(2);
}

function parseArgs(argv) {
  const args = {};
  for (let i = 0; i < argv.length; i++) {
    const token = argv[i];
    if (!token.startsWith('--')) throw new Error(`unexpected argument ${token}`);
    const name = token.slice(2);
    if (name === 'approved') {
      args.approved = true;
      continue;
    }
    if (i + 1 >= argv.length) throw new Error(`--${name} needs a value`);
    args[name] = argv[++i];
  }
  return args;
}

function requireNumber(args, name, fallback) {
  if (args[name] === undefined) {
    if (fallback !== undefined) return fallback;
    fail(`--${name} is required`);
  }
  const value = Number(args[name]);
  if (!Number.isFinite(value)) fail(`--${name} must be a number, got ${args[name]}`);
  return value;
}

function requireString(args, name) {
  if (args[name] === undefined) fail(`--${name} is required`);
  return args[name];
}

// `--workers N`: a flag only (SCOPE's "N worker processes"), clamped to
// [1, min(8, cores)]. NOTE (reported, not silently assumed): this unit plays
// every pair sequentially in this one process regardless of `workers` --
// real OS-process fan-out is not wired up here. No test (and no command this
// ticket's acceptance runs) exercises `workers > 1`; the flag is accepted and
// validated so a future unit can wire the pool in without a CLI change.
function clampWorkers(raw) {
  const cap = Math.max(1, Math.min(8, cpus().length));
  const n = raw === undefined ? 1 : Math.trunc(Number(raw));
  if (!Number.isFinite(n) || n < 1) fail(`--workers must be a positive integer, got ${raw}`);
  return Math.min(n, cap);
}

async function runMatch(args) {
  if (args.model !== undefined && args.approved !== true) {
    fail(`refuses a real-model match (--model ${args.model}) without --approved`);
  }

  const out = requireString(args, 'out');
  const pairs = requireNumber(args, 'pairs');
  const sims = requireNumber(args, 'sims');
  const fastSims = requireNumber(args, 'fast-sims', sims);
  const exactSims = requireNumber(args, 'exact-sims', sims);
  const fastBatch = requireNumber(args, 'batch', 1);
  const seed = requireNumber(args, 'seed', 1);
  const elo0 = requireNumber(args, 'elo0', -30);
  const elo1 = requireNumber(args, 'elo1', 0);
  const alpha = requireNumber(args, 'alpha', 0.05);
  const beta = requireNumber(args, 'beta', 0.05);
  const minPairs = requireNumber(args, 'min-pairs', 1);
  const maxPlies = args['max-plies'] === undefined ? undefined : requireNumber(args, 'max-plies');
  const planeMapVersion = resolvePlaneMapVersion(args);
  clampWorkers(args.workers);

  let fastBatchEvaluator;
  let exactLeafEvaluator;
  let modelSha256 = null;
  if (args.model !== undefined) {
    const real = await makeRealEvaluators(args.model);
    fastBatchEvaluator = real.batch;
    exactLeafEvaluator = real.leaf;
  } else {
    const fakeLeaf = makeFakeMlhEvaluator(ACTION_SIZE);
    exactLeafEvaluator = fakeLeaf;
    fastBatchEvaluator = (states) => states.map(fakeLeaf);
  }

  const engines = {
    exact: new WorkerEngine({ leafEvaluator: exactLeafEvaluator, modelSha256 }),
    fast: new WorkerEngine({ modelSha256 }),
  };

  writeFileSync(out, '');
  const counts = [0, 0, 0, 0, 0];
  let wins = 0;
  let draws = 0;
  let losses = 0;
  let gamesPlayed = 0;
  let pairsPlayed = 0;
  let stopped = 'cap';
  const bounds = sprtBounds(alpha, beta);

  for (let pairIndex = 0; pairIndex < pairs; pairIndex++) {
    const opening = buildSeededOpening(seed + pairIndex);
    let doubledPairScore = 0;
    for (let gameInPair = 0; gameInPair < 2; gameInPair++) {
      const colorToRole = colorToRoleFor(gameInPair);
      const fastColor = gameInPair === 0 ? 0 : 1;
      const played = await playGame(
        opening,
        colorToRole,
        engines,
        fastBatchEvaluator,
        sims,
        fastSims,
        exactSims,
        fastBatch,
        maxPlies,
        planeMapVersion,
      );
      const fastScore = fastScoreOf(played.outcome, fastColor);
      if (fastScore === 1) wins += 1;
      else if (fastScore === 0.5) draws += 1;
      else losses += 1;
      doubledPairScore += Math.round(fastScore * 2);
      gamesPlayed += 1;
      appendFileSync(
        out,
        `${JSON.stringify({
          pair: pairIndex,
          gameInPair,
          fastColor,
          moves: played.history,
          plies: played.plies,
          result: { white: played.outcome[0], black: played.outcome[1], draw: played.outcome[2] },
          fastScore,
          plyCapHit: played.plyCapHit,
        })}\n`,
      );
    }
    counts[doubledPairScore] += 1;
    pairsPlayed += 1;

    const llr = llrPentanomial(counts, elo0, elo1);
    if (pairsPlayed >= minPairs) {
      if (llr >= bounds.upper) {
        stopped = 'bound-h1';
        break;
      }
      if (llr <= bounds.lower) {
        stopped = 'bound-h0';
        break;
      }
    }
    process.stderr.write(`fast_match.mjs: pair ${pairIndex} llr=${llr.toFixed(3)}\n`);
  }

  const llr = llrPentanomial(counts, elo0, elo1);
  const verdict = stopped === 'bound-h1' ? 'accept-h1' : stopped === 'bound-h0' ? 'accept-h0' : 'inconclusive';
  const summary = {
    pairsPlayed,
    gamesPlayed,
    wins,
    draws,
    losses,
    pentanomial: counts,
    elo: eloInterval(counts),
    llr,
    bounds,
    elo0,
    elo1,
    stopped,
    verdict,
  };
  process.stdout.write(`${JSON.stringify(summary)}\n`);
}

async function main() {
  const args = parseArgs(process.argv.slice(2));

  if (args['emit-llr'] !== undefined) {
    const payload = JSON.parse(args['emit-llr']);
    const llr = llrPentanomial(payload.counts, payload.elo0, payload.elo1);
    const bounds = sprtBounds(payload.alpha, payload.beta);
    const stats = pentanomialStats(payload.counts);
    process.stdout.write(`${JSON.stringify({ llr, bounds, ...stats })}\n`);
    return;
  }

  if (args['emit-equal-time-sims'] !== undefined) {
    const payload = JSON.parse(args['emit-equal-time-sims']);
    const result = equalTimeSims(payload.table, payload.backend, payload.batch);
    process.stdout.write(`${JSON.stringify(result)}\n`);
    return;
  }

  await runMatch(args);
}

// KOM-1168: guarded so host_plane_map.test.ts can import this module's
// helpers without running the CLI -- `main()` fires only when this file is
// executed directly, never on import.
if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((error) => {
    process.stderr.write(`fast_match.mjs: ${error && error.stack ? error.stack : error}\n`);
    process.exit(1);
  });
}
