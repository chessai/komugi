// KOM-1074/KOM-1078 (design U9) + KOM-1093 (design U6c, fast mode's o12
// harness): drives worker.ts's WorkerEngine (exact mode) or, for fast mode,
// its own searchFast-based engine (worker.ts has no `handleFast` yet -- U6a
// is a separate unit) on onnxruntime-web for the end-to-end port-agreement harness.
//
// Usage: node e2e_driver.mjs --model m.onnx --requests in.json --sims 32 --out out.json
//          [--evals torch.json] [--mode exact|fast] [--batch N]
//          [--plane-map-version 3|4]
// `in.json` is a JSON array of { fen, history }; `out.json` receives one
// WorkerReply per request, in order. Config is left at the production
// identity (protocol.ts PRODUCTION_IDENTITY) in both modes.
//
// `--plane-map-version` (KOM-1168, default 3): threaded onto every request
// and every MatchGameState this driver builds itself (fast mode's root
// state); encoded with `src/host_encode.ts`'s `encodeState`/
// `encodeStatesBatch`, which read it off the state rather than assuming v3.
//
// `--mode exact` (default) is unchanged from KOM-1074: `engine.handleAsync()`
// straight through onnxruntime-web, no cache, no replay -- today's referee.
//
// `--mode fast --batch N` (KOM-1093, M-3's o12 re-run) drives `searchFastAsync`
// with the U4 batch evaluator: `--evals` gives a batch evaluator that maps
// each row through the torch cache (design 2.1's frozen batch seam, applied
// to a lookup rather than a real net call -- at `batch=1` this reproduces
// the exact-mode torch control bit for bit, since `searchFast(batch=1)` with
// no deadline dispatches straight to `search()`, design 1.7); its absence
// means a genuinely batched ORT call (`runOrtBatch`). Per design 6.3 S-1,
// the root's single-leaf call and the batch evaluator share one
// `MlhContext`, both wrapping the SAME batch leaf lookup -- never two
// independent wrappers.

import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';

import { decodeAction } from '../src/actions.ts';
import { wrapMlhBatch, runOrtBatch } from '../src/batch_evaluator.ts';
import { ACTION_SIZE } from '../src/constants.ts';
import { makeOrtMlhLeafEvaluator } from '../src/ort_core.ts';
import { loadOrtSession } from '../test/support/ort_node.ts';
import { encodeState, encodeStatesBatch } from '../src/host_encode.ts';
import { observationSize } from '../src/observation.ts';
import { PRODUCTION_IDENTITY } from '../src/protocol.ts';
import { rootGame } from '../src/root.ts';
import { parsePlaneMapVersion } from '../src/rule_features.ts';
import { buildSan } from '../src/san.ts';
import { MlhContext } from '../src/search/mlh.ts';
import { searchFastAsync } from '../src/search/search.ts';
import { MatchGameState } from '../src/search/state.ts';
import { MatchTree } from '../src/search/tree.ts';
import { WorkerEngine } from '../src/engine.ts';

// KOM-1168: resolves `--plane-map-version` the same way every CLI flag this
// driver validates eagerly is resolved -- `undefined` defaults to 3 (parse-
// PlaneMapVersion's own default), anything else throws its exact message.
export function resolvePlaneMapVersion(args) {
  const raw = args['plane-map-version'];
  return parsePlaneMapVersion(raw === undefined ? undefined : Number(raw));
}

export function parseArgs(argv) {
  const args = {};
  for (let i = 0; i < argv.length; i += 2) {
    if (!argv[i].startsWith('--') || i + 1 >= argv.length) throw new Error(`bad argument ${argv[i]}`);
    args[argv[i].slice(2)] = argv[i + 1];
  }
  for (const name of ['model', 'requests', 'sims', 'out']) {
    if (!args[name]) throw new Error(`--${name} is required`);
  }
  if (args.mode !== undefined && args.mode !== 'exact' && args.mode !== 'fast') {
    throw new Error(`--mode must be 'exact' or 'fast', got ${args.mode}`);
  }
  return args;
}

// worker.ts's own private match-config defaults (BASE_MATCH_CONFIG,
// worker.ts:41-63), duplicated here for the same reason every test file
// duplicates it (async.test.ts, search_batch.test.ts, ...): it is not
// exported, and worker.ts is outside this ticket's file allowlist.
const BASE_MATCH_CONFIG = {
  num_actions: ACTION_SIZE,
  contempt: 0.0,
  dynamic_cpuct: true,
  c_puct: 1.0,
  c_explore: 1.0,
  c_log: 0.45,
  c_base: 500.0,
  sigma_prior: 0.4,
  sigma_prior_weight: 2.0,
  sigma_scale: 0.85,
  sigma_floor: 0.02,
  fpu_reduction: 0.2,
  fpu_reduction_root: 0.1,
  subtree_bias_factor: 0.45,
  bias_weight_exponent: 0.85,
  bias_free_prop: 0.8,
  bias_weight_floor: 1e-6,
  k_lcb: 5.0,
  min_visit_prop_for_lcb: 0.15,
  var_floor: 0.05,
  variance_mode: 'moment',
};

// Fast mode's own tiny engine (KOM-1093): the one retained MatchTree per
// process, mirroring worker.ts's WorkerEngine closely enough for this
// harness -- production identity, one tree reused request to request so the
// bias table persists -- without needing `handleFast` (U6a, not built yet).
export class FastEngine {
  constructor(batchLeafEvaluator, modelSha256, backend) {
    this.batchLeafEvaluator = batchLeafEvaluator;
    this.modelSha256 = modelSha256;
    this.backend = backend;
    this.tree = null;
  }

  async handle(request, batch) {
    try {
      const { game, stamp } = rootGame(request.fen, request.history);
      const rootBoard = game.board;
      const state = new MatchGameState(game, {
        minPlacements: PRODUCTION_IDENTITY.draftmin,
        isRoot: true,
        planeMapVersion: request.planeMapVersion,
      });
      const treeConfig = { ...BASE_MATCH_CONFIG, playouts: request.sims };
      const ctx = new MlhContext();
      // S-1: the batch evaluator and the root's single-leaf evaluator wrap
      // the SAME batch leaf lookup with the SAME MlhContext (a one-row call
      // for the root), so `ctx.mRoot`/`ctx.gated` are set exactly once.
      const batchWdl = wrapMlhBatch(this.batchLeafEvaluator, PRODUCTION_IDENTITY.mlh, ctx);
      const singleWdl = async (s) => (await batchWdl([s]))[0];
      if (this.tree === null) {
        this.tree = new MatchTree(treeConfig, singleWdl, batchWdl);
      } else {
        this.tree.config = treeConfig;
        this.tree.evaluator = singleWdl;
        this.tree.batchEvaluator = batchWdl;
      }
      const searchOpts = { discovery: PRODUCTION_IDENTITY.discovery, ply: game.turns };
      const started = Date.now();
      const result = await searchFastAsync(
        this.tree, state, { ...treeConfig, playouts: request.sims }, searchOpts, batch,
      );
      const { struct } = decodeAction(rootBoard, result.action);
      const san = buildSan(rootBoard, struct);
      return {
        id: request.id,
        san,
        action: result.action,
        stamp,
        stats: {
          playouts: result.playoutsUsed,
          evaluations: result.evaluationsUsed,
          nodes: this.tree.nodeCount,
          rootWdl: Array.from(result.rootWdl),
          tactical: result.tactical,
          ms: Date.now() - started,
          model_sha256: this.modelSha256,
          sims: request.sims,
          batch,
          backend: this.backend,
          deadlineHit: result.deadlineHit,
        },
      };
    } catch (error) {
      const message = error && error.message ? error.message : String(error);
      return { id: request.id, err: message };
    }
  }
}

// Fast mode's torch-fed batch evaluator: each row looked up in the same
// recorded-answer cache the exact-mode control reads (keyed by obs hash,
// the Python end-to-end agreement check's `obs_key`), mapped synchronously
// over the batch -- there is no real net call to batch here, only a cache
// lookup.
export function makeTorchBatchLeafEvaluator(evalsPath) {
  const leaf = makeTorchLeafEvaluator(evalsPath);
  return async (states) => states.map((state) => leaf(state));
}

// Fast mode's real batch evaluator: one genuinely batched `runOrtBatch` call
// per collected batch (design 2.2), rows concatenated in collection order.
// KOM-1168: each row is encoded at ITS OWN state's planeMapVersion
// (encodeStatesBatch), not assumed v3.
export function makeOrtBatchLeafEvaluator(session, ort, obsShape) {
  const rowSize = obsShape.reduce((a, b) => a * b, 1);
  return async (states) => {
    const obsRows = encodeStatesBatch(states, rowSize);
    return runOrtBatch(session, ort, obsRows, states.length, obsShape);
  };
}

function keyOf(obs) {
  return createHash('sha256').update(new Uint8Array(obs.buffer, obs.byteOffset, obs.byteLength)).digest('hex');
}

// Torch mode (--evals): the Python server's own recorded net answers,
// keyed the same way the recording evaluator keys them (the Python
// end-to-end agreement check's obs_key). A miss means the TS tree asked for
// a leaf the Python tree never evaluated -- an `err`, not a silent guess.
// KOM-1168: keyed on the state's own planeMapVersion (encodeState), not an
// assumed v3.
export function makeTorchLeafEvaluator(evalsPath) {
  const record = JSON.parse(readFileSync(evalsPath, 'utf8'));
  const cache = new Map();
  for (const [key, entry] of Object.entries(record)) {
    const raw = Buffer.from(entry.logits, 'base64');
    const values = new Float32Array(raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength));
    const logits = new Float32Array(ACTION_SIZE).fill(Number.NaN);
    entry.ids.forEach((id, i) => { logits[id] = values[i]; });
    cache.set(key, { policyLogits: logits, wdl: Float64Array.from(entry.wdl), movesLeft: entry.moves_left });
  }
  return (state) => {
    const key = keyOf(encodeState(state));
    const hit = cache.get(key);
    if (hit === undefined) throw new Error(`e2e_driver: torch cache miss at ${key}`);
    return { policyLogits: hit.policyLogits, wdl: Float64Array.from(hit.wdl), movesLeft: hit.movesLeft };
  };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const sims = Number(args.sims);
  const mode = args.mode ?? 'exact';
  const batch = args.batch !== undefined ? Number(args.batch) : 1;
  const planeMapVersion = resolvePlaneMapVersion(args);
  const requests = JSON.parse(readFileSync(args.requests, 'utf8'));
  const modelSha256 = createHash('sha256').update(readFileSync(args.model)).digest('hex');

  if (mode === 'fast') {
    let batchLeafEvaluator;
    let backend;
    if (args.evals) {
      batchLeafEvaluator = makeTorchBatchLeafEvaluator(args.evals);
      backend = 'torch';
    } else {
      const { ort, session } = await loadOrtSession(args.model, { threads: 1 });
      batchLeafEvaluator = makeOrtBatchLeafEvaluator(session, ort, observationSize());
      backend = 'wasm';
    }
    const engine = new FastEngine(batchLeafEvaluator, modelSha256, backend);
    const replies = [];
    for (let index = 0; index < requests.length; index++) {
      const request = { id: index, fen: requests[index].fen, history: requests[index].history, sims, planeMapVersion };
      const reply = await engine.handle(request, batch);
      replies.push(reply);
      process.stderr.write(`e2e_driver: fast[${batch}] ${index} ${reply.san ?? reply.err}\n`);
    }
    writeFileSync(args.out, JSON.stringify(replies));
    process.stderr.write(`e2e_driver: fast mode, batch=${batch}, ${requests.length} requests\n`);
    return;
  }

  let leafEvaluator;
  let ortCalls = 0;
  if (args.evals) {
    leafEvaluator = makeTorchLeafEvaluator(args.evals);
  } else {
    const { ort, session } = await loadOrtSession(args.model, { threads: 1 });
    const ortEvaluator = makeOrtMlhLeafEvaluator(session, ort, observationSize());
    leafEvaluator = async (state) => {
      ortCalls++;
      return ortEvaluator(state);
    };
  }

  const engine = new WorkerEngine({ leafEvaluator, modelSha256 });

  const replies = [];
  for (let index = 0; index < requests.length; index++) {
    const request = { id: index, fen: requests[index].fen, history: requests[index].history, sims, planeMapVersion };
    const reply = await engine.handleAsync(request);
    replies.push(reply);
    process.stderr.write(`e2e_driver: ${index} ${reply.san ?? reply.err}\n`);
  }
  writeFileSync(args.out, JSON.stringify(replies));
  process.stderr.write(`e2e_driver: ${requests.length} requests, ${ortCalls} ORT evaluations\n`);
}

// KOM-1168: guarded so host_plane_map.test.ts can import this module's
// helpers without running the CLI -- `main()` fires only when this file is
// executed directly, never on import.
if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((error) => {
    process.stderr.write(`e2e_driver: ${error && error.stack ? error.stack : error}\n`);
    process.exit(1);
  });
}
