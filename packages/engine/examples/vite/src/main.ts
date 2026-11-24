// Loads the model the assets CLI copied into public/, then plays one move
// from the Advanced starting position.
import { ADVANCED_POSITION, DEFAULT_LEVEL, KomugiClient, LEVELS, loadModel } from '@komugi-ai/engine';
import type { KomugiOrtAssets, LevelName } from '@komugi-ai/engine';

interface OrtAssetFile {
  path: string;
}
interface OrtAssetsJson {
  wasm: { mjs: OrtAssetFile; wasm: OrtAssetFile };
  webgpu?: { mjs: OrtAssetFile; wasm: OrtAssetFile };
}
interface ModelsJson {
  recommended: Record<string, string>;
  models: Record<string, { card: string }>;
}

const logEl = document.getElementById('log');
function log(line: string): void {
  if (logEl) logEl.textContent = line;
}

async function run(): Promise<void> {
  // Every path in ort-assets.json, and the card URL, is resolved against the
  // absolute URL ort-assets.json was served from -- never passed through
  // relative (a relative path is a bare specifier inside the worker).
  const ortAssetsUrl = new URL('/ort-assets.json', location.href).href;
  const ort = (await (await fetch(ortAssetsUrl)).json()) as OrtAssetsJson;
  const abs = (p: string) => new URL(p, ortAssetsUrl).href;
  const ortAssets: KomugiOrtAssets = {
    wasm: { mjs: abs(ort.wasm.mjs.path), wasm: abs(ort.wasm.wasm.path) },
    ...(ort.webgpu ? { webgpu: { mjs: abs(ort.webgpu.mjs.path), wasm: abs(ort.webgpu.wasm.path) } } : {}),
  };

  // The CLI writes <id>.card.json beside models.json; take the id it lists.
  const index = (await (await fetch(abs('models.json'))).json()) as ModelsJson;
  const id = index.recommended.advanced ?? Object.keys(index.models)[0];
  if (id === undefined) throw new Error('models.json lists no model');
  const cardUrl = abs(`${id}.card.json`);

  const loaded = await loadModel(
    { manifestUrl: cardUrl },
    { onProgress: (done, total) => log(`Loading model: ${done} / ${total ?? '?'} bytes`) },
  );
  if (loaded.status === 'failed') {
    log(`Model load failed (${loaded.failure.code}): ${loaded.failure.message}`);
    return;
  }

  const client = new KomugiClient({
    createWorker: () => new Worker(new URL('./komugi.worker.ts', import.meta.url), { type: 'module' }),
    model: loaded.model,
    ortAssets,
  });
  const level: LevelName = DEFAULT_LEVEL;
  const { playouts, capMs } = LEVELS[level];
  const result = await client.play({ fen: ADVANCED_POSITION, sims: playouts, maxMs: capMs });
  if (result.status === 'move') {
    log(`Komugi (${level}) plays ${result.reply.san}`);
  } else if (result.status === 'failed') {
    // A typed failure: show it and offer Try again, never another engine's move.
    log(`Search failed (${result.failure.code}): ${result.failure.message}`);
  } else {
    log('Search stopped');
  }
  client.dispose();
}

void run().catch((error: unknown) => log(`Error: ${String(error)}`));
