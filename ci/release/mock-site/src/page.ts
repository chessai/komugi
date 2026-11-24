// KOM-1221 unit 3 (design docs/design/kom1221-release-ci.md section 4,
// section 5): the mock site's page script. Bundled by mock-site.sh into
// page.bundle.js and loaded by index.html. Exposes `window.__komugi`'s
// five functions (section 4's fixed contract, binding for units 4/5) for a
// Python Playwright driver to call through `page.evaluate`.
//
// Imports the engine's built dist/ output by relative path, not the
// `@komugi-ai/engine` package specifier: the mock site is never
// `npm install`ed (D19/F2 -- nothing may be left under
// packages/engine/node_modules but the onnxruntime-web symlink
// ensure_ort_web_linked makes), so there is no node_modules entry for a
// bare specifier to resolve against. `engine.sh` builds dist/ first
// (mock-site.sh's own G5b note).

import {
  ADVANCED_POSITION,
  KomugiClient,
  boardToFen,
  buildSan,
  decodeAction,
  isCheckmate,
  isGameOver,
  isInsufficientMaterial,
  isRepetitionDraw,
  isStalemate,
  loadModel,
  resolveLevel,
  rootGame,
} from '../../../../packages/engine/dist/index.js';
import type {
  Game,
  KomugiFailure,
  KomugiModel,
  KomugiOrtAssets,
  LevelName,
  PlayResult,
  TerminalReason,
} from '../../../../packages/engine/dist/index.js';

// --- deterministic PRNG (same formulas as
// packages/engine/src/fake_evaluator.ts's fnv1a32 and
// test/ligungi_filter_fuzz.test.ts's mulberry32 rng(), inlined here since
// this module hashes a string key, not a Uint8Array/needs no test harness
// import) ---

const FNV32_OFFSET = 0x811c9dc5;
const FNV32_PRIME = 0x01000193;

function fnv1a32(text: string): number {
  let h = FNV32_OFFSET;
  const bytes = new TextEncoder().encode(text);
  for (let i = 0; i < bytes.length; i++) {
    h = (h ^ bytes[i]!) >>> 0;
    h = Math.imul(h, FNV32_PRIME) >>> 0;
  }
  return h;
}

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// --- site asset wiring (integration.md section 7, verbatim pattern; the
// model id comes from models.json's recommended.advanced -- the mechanical
// form of "latest ships", section 3.2 step 5) ---

interface SiteAssets {
  ortAssets: KomugiOrtAssets;
  cardUrl: string;
}

async function resolveSiteAssets(): Promise<SiteAssets> {
  const ortAssetsUrl = new URL('ort-assets.json', location.href).href;
  const ort = await (await fetch(ortAssetsUrl)).json();
  const abs = (p: string): string => new URL(p, ortAssetsUrl).href;
  const ortAssets: KomugiOrtAssets = {
    wasm: { mjs: abs(ort.wasm.mjs.path), wasm: abs(ort.wasm.wasm.path) },
    ...(ort.webgpu ? { webgpu: { mjs: abs(ort.webgpu.mjs.path), wasm: abs(ort.webgpu.wasm.path) } } : {}),
  };
  const models = await (await fetch(abs('models.json'))).json();
  const id = models?.recommended?.advanced;
  return { ortAssets, cardUrl: abs(`${id}.card.json`) };
}

type LoadedSite =
  | { status: 'ok'; ortAssets: KomugiOrtAssets; model: KomugiModel; rules: string }
  | { status: 'failed'; failure: KomugiFailure };

async function loadSiteModel(): Promise<LoadedSite> {
  const { ortAssets, cardUrl } = await resolveSiteAssets();
  const loaded = await loadModel({ manifestUrl: cardUrl });
  if (loaded.status === 'failed') return { status: 'failed', failure: loaded.failure };
  return { status: 'ok', ortAssets, model: loaded.model, rules: loaded.card.compatibilityKey.rulesProfile };
}

function makeWorker(scriptName: string): Worker {
  return new Worker(new URL(scriptName, location.href), { type: 'module' });
}

// --- 5.4's reason derivation, independent of Game.winState() (which
// computes the same `reason` internally but never returns it) -- same
// predicate order, so a position winState() would call terminal is
// classified identically here. ---

function engineTerminalReason(game: Game): TerminalReason | null {
  const config = game.config;
  if (config.plyCap !== null && game.turns >= config.plyCap) return 'ply-cap';
  if (game.board.draftW || game.board.draftB) return null;
  const count = game.repetitionCount();
  if (!isGameOver(game.board, count, config)) return null;
  if (isCheckmate(game.board, config)) return 'checkmate';
  if (isStalemate(game.board, config)) return 'stalemate';
  if (isRepetitionDraw(count)) return 'repetition';
  if (isInsufficientMaterial(game.board)) return 'insufficient';
  return 'marshal-captured';
}

// --- 5.1: the driving loop ---

interface PlyEntry {
  side: 'komugi' | 'prng';
  preFen: string;
  action: number;
  san: string;
}

interface RunGameResult {
  plies: PlyEntry[];
  log: Record<string, unknown>[];
  status: 'ok' | 'failed';
  reason: string;
}

// `game` in every log/ply-group line is `seed`: runGame() takes no separate
// gameIndex (section 4's fixed 3-argument signature), so the caller
// (e2e.sh's driver, unit 4) is expected to vary `seed` per (level,
// gameIndex) pair -- this is what both the log's `game` field and the
// PRNG's own derivation below key off.
async function runGame(level: LevelName, seed: number, plyCap: number): Promise<RunGameResult> {
  const log: Record<string, unknown>[] = [];
  const plies: PlyEntry[] = [];
  let reason = 'ply_cap';
  const t0 = performance.now();
  let client: KomugiClient | undefined;

  try {
    const site = await loadSiteModel();
    if (site.status === 'failed') {
      log.push({ event: 'end', level, game: seed, reason: site.failure.code, plies: 0 });
      return { plies, log, status: 'failed', reason: site.failure.code };
    }

    client = new KomugiClient({
      createWorker: () => makeWorker('worker.js'),
      model: site.model,
      ortAssets: site.ortAssets,
    });

    const init = await client.init();
    log.push({ event: 'init', elapsedMs: performance.now() - t0 });
    if (init.status === 'failed') {
      log.push({ event: 'end', level, game: seed, reason: init.failure.code, plies: 0 });
      return { plies, log, status: 'failed', reason: init.failure.code };
    }

    const rules = site.rules;
    // Komugi's colour alternates with the caller-supplied seed's parity --
    // the only per-call signal available under the fixed 3-arg signature.
    const komugiColor = Math.abs(Math.trunc(seed)) % 2;
    const rng = mulberry32(fnv1a32(`${seed}:${level}`));
    const game = rootGame(ADVANCED_POSITION, undefined, undefined, rules).game;

    let ply = 0;
    for (; ply < plyCap; ply++) {
      const terminal = engineTerminalReason(game);
      if (terminal !== null) {
        reason = terminal;
        break;
      }
      const preFen = boardToFen(game.board);

      if (game.board.turn === komugiColor) {
        const resolved = resolveLevel(level, init.info);
        const started = performance.now();
        const playResult: PlayResult = await client.play({
          fen: preFen,
          sims: resolved.sims,
          maxMs: resolved.maxMs,
          batch: resolved.batch,
          rules,
        });
        const elapsedMs = performance.now() - started;

        if (playResult.status === 'move') {
          const stats = playResult.reply.stats;
          const line: Record<string, unknown> = {
            level,
            game: seed,
            ply,
            preFen,
            action: playResult.reply.action,
            san: playResult.reply.san,
            rules,
            status: 'move',
            failure: null,
            elapsedMs,
            statsMs: stats.ms,
            backend: stats.backend,
            fallback: stats.fallback ?? null,
            deadlineHit: stats.deadlineHit,
          };
          log.push(line);
          plies.push({ side: 'komugi', preFen, action: playResult.reply.action, san: playResult.reply.san });
          game.play(playResult.reply.action);
        } else {
          const line: Record<string, unknown> = {
            level,
            game: seed,
            ply,
            preFen,
            rules,
            status: playResult.status,
            elapsedMs,
            fallback: null,
          };
          if (playResult.status === 'failed') {
            line.failure = playResult.failure;
            reason = playResult.failure.code;
          } else {
            line.failure = null;
            reason = 'stopped';
          }
          log.push(line);
          break;
        }
      } else {
        const mask = game.validMoves();
        const candidates: number[] = [];
        for (let i = 0; i < mask.length; i++) if (mask[i]) candidates.push(i);
        const action = candidates[Math.floor(rng() * candidates.length)]!;
        const { struct } = decodeAction(game.board, action);
        const san = buildSan(game.board, struct);
        plies.push({ side: 'prng', preFen, action, san });
        game.play(action);
      }
    }

    log.push({ event: 'end', level, game: seed, reason, plies: plies.length });
    return { plies, log, status: 'ok', reason };
  } catch (error) {
    log.push({ event: 'end', level, game: seed, reason: 'error', plies: plies.length });
    return { plies, log, status: 'failed', reason: `error: ${String(error)}` };
  } finally {
    client?.dispose();
  }
}

// --- 5.2: MARGIN_MS derivation input. n deadline-bound `focused` searches
// (maxMs == capMs), each with a playout count no hardware can finish
// inside capMs (so deadlineHit always fires and the search is always
// stopped by the deadline, never by finishing early. ---

// Comfortably beyond anything a `focused` cap (8 000 ms, levels.ts) could
// finish on any hardware this project targets (benchmark-typical-hardware
// memory: middling GPU/8-16 cores) -- the point is only that the deadline,
// not the playout count, ends every one of these searches.
const OVERRUN_SIMS = 10_000_000;

async function measureOverrun(n: number): Promise<{ overrunsMs: number[] }> {
  const overrunsMs: number[] = [];
  let client: KomugiClient | undefined;
  try {
    const site = await loadSiteModel();
    if (site.status === 'failed') return { overrunsMs };
    client = new KomugiClient({ createWorker: () => makeWorker('worker.js'), model: site.model, ortAssets: site.ortAssets });
    const init = await client.init();
    if (init.status === 'failed') return { overrunsMs };

    const resolved = resolveLevel('focused', init.info);
    for (let i = 0; i < n; i++) {
      const started = performance.now();
      await client.play({ fen: ADVANCED_POSITION, sims: OVERRUN_SIMS, maxMs: resolved.maxMs, batch: resolved.batch, rules: site.rules });
      overrunsMs.push(performance.now() - started - resolved.maxMs);
    }
    return { overrunsMs };
  } catch {
    return { overrunsMs };
  } finally {
    client?.dispose();
  }
}

// --- 5.5: forced crash and "Try again" ---

interface CrashSequenceResult {
  latched: boolean;
  retried: boolean;
  status: 'ok' | 'failed';
}

async function crashSequence(): Promise<CrashSequenceResult> {
  let client: KomugiClient | undefined;
  let worker: Worker | undefined;
  try {
    const site = await loadSiteModel();
    if (site.status === 'failed') return { latched: false, retried: false, status: 'failed' };

    client = new KomugiClient({
      createWorker: () => {
        worker = makeWorker('crash_worker.js');
        return worker;
      },
      model: site.model,
      ortAssets: site.ortAssets,
      maxConsecutiveCrashes: 1,
    });

    const init = await client.init();
    if (init.status === 'failed' || worker === undefined) return { latched: false, retried: false, status: 'failed' };

    const resolved = resolveLevel('casual', init.info);
    const playOpts = { fen: ADVANCED_POSITION, sims: resolved.sims, maxMs: resolved.maxMs, batch: resolved.batch, rules: site.rules };

    // Post the crash mid-flight: no `await` between starting the search and
    // posting it (crash_worker_entry.ts's own listener fires on a
    // setTimeout(0), well before a real search on a fixture-sized model
    // completes).
    const firstPlay = client.play(playOpts);
    worker.postMessage({ type: '__crash' });
    const first = await firstPlay;
    const latchedNow =
      first.status === 'failed' && first.failure.code === 'worker-crash' && first.failure.stage === 'search' && client.state === 'failed';

    // A further play() on the same preFen returns the latched failure, no
    // new worker spawned -- `createWorker` must not have run again.
    const workerBeforeRetry = worker;
    const second = await client.play(playOpts);
    const stillLatched = second.status === 'failed' && worker === workerBeforeRetry;

    client.retry();
    const retriedResult = await client.play(playOpts);
    const retried = retriedResult.status === 'move';

    return { latched: latchedNow && stillLatched, retried, status: retried ? 'ok' : 'failed' };
  } catch {
    return { latched: false, retried: false, status: 'failed' };
  } finally {
    client?.dispose();
  }
}

// --- 5.6: negative cases, page paths ---

async function loadOnly(cardUrl: string): Promise<{ status: 'ok' | 'failed'; failure?: KomugiFailure }> {
  const result = await loadModel({ manifestUrl: cardUrl });
  if (result.status === 'failed') return { status: 'failed', failure: result.failure };
  result.model.release?.();
  return { status: 'ok' };
}

async function rawClientInit(onnxUrl: string, sha: string): Promise<{ status: 'ok' | 'failed'; failure?: KomugiFailure }> {
  let client: KomugiClient | undefined;
  try {
    const { ortAssets } = await resolveSiteAssets();
    const model: KomugiModel = { url: onnxUrl, sha256: sha };
    client = new KomugiClient({ createWorker: () => makeWorker('worker.js'), model, ortAssets });
    const result = await client.init();
    if (result.status === 'failed') return { status: 'failed', failure: result.failure };
    return { status: 'ok' };
  } finally {
    client?.dispose();
  }
}

declare global {
  interface Window {
    __komugi: {
      runGame: typeof runGame;
      measureOverrun: typeof measureOverrun;
      crashSequence: typeof crashSequence;
      loadOnly: typeof loadOnly;
      rawClientInit: typeof rawClientInit;
    };
  }
}

window.__komugi = { runGame, measureOverrun, crashSequence, loadOnly, rawClientInit };
