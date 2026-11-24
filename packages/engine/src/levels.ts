import type { BackendRecord } from './backend.ts';

export const LEVEL_NAMES = ['casual', 'focused', 'sharp', 'awakened'] as const;
export type LevelName = (typeof LEVEL_NAMES)[number];

export const DEFAULT_LEVEL: LevelName = 'focused';

export interface LevelBatch {
  webgpu: number;
  wasmMt: number;
  wasm1t: number;
}

export interface LevelDef {
  playouts: number;
  capMs: number;
  batch: LevelBatch;
}

// Frozen at run time, not just readonly at the type level: a page importing
// LEVELS must not be able to mutate the shared table.
function level(playouts: number, capMs: number, batch: LevelBatch): Readonly<LevelDef> {
  return Object.freeze({ playouts, capMs, batch: Object.freeze({ ...batch }) });
}

export const LEVELS: Readonly<Record<LevelName, Readonly<LevelDef>>> = Object.freeze({
  casual: level(100, 2_000, { webgpu: 2, wasmMt: 2, wasm1t: 1 }),
  focused: level(400, 8_000, { webgpu: 8, wasmMt: 4, wasm1t: 1 }),
  sharp: level(800, 15_000, { webgpu: 8, wasmMt: 4, wasm1t: 1 }),
  awakened: level(1600, 30_000, { webgpu: 8, wasmMt: 4, wasm1t: 1 }),
});

// Runtime guard for untyped hosts (postMessage payloads): only an own entry
// of LEVEL_NAMES is a level; 'Focused', 'champion', 7, '__proto__' and
// 'constructor' are all rejected, never looked up in LEVELS.
export function parseLevel(value: unknown): LevelName {
  if (typeof value === 'string' && (LEVEL_NAMES as readonly string[]).includes(value)) {
    return value as LevelName;
  }
  throw new Error(`unknown level ${describeLevelValue(value)}; expected one of ${LEVEL_NAMES.join(', ')}`);
}

// An untyped host can hand this a BigInt, which JSON.stringify throws on
// (TypeError) -- that must not turn a validation error into a crash with a
// worse message.
function describeLevelValue(value: unknown): string {
  try {
    return JSON.stringify(value) ?? String(value);
  } catch {
    return String(value);
  }
}

export type BackendCase = 'webgpu' | 'wasm-mt' | 'wasm-1t';

export function backendCaseOf(backend: BackendRecord['backend'], threads: number): BackendCase {
  if (backend === 'webgpu') return 'webgpu';
  return threads > 1 ? 'wasm-mt' : 'wasm-1t';
}

export interface ResolvedLevel {
  sims: number;
  batch: number;
  maxMs: number;
}

// `record` only needs backend/threads; a full BackendRecord satisfies this
// structurally, so callers never have to destructure first.
export function resolveLevel(
  name: LevelName,
  record: Pick<BackendRecord, 'backend' | 'threads'>,
): ResolvedLevel {
  const def = LEVELS[name];
  const kase = backendCaseOf(record.backend, record.threads);
  const batch = kase === 'webgpu' ? def.batch.webgpu : kase === 'wasm-mt' ? def.batch.wasmMt : def.batch.wasm1t;
  return { sims: def.playouts, batch, maxMs: def.capMs };
}
