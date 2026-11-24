// KOM-1188 unit 2 (design docs/design/kom1188-model-releases.md section 5):
// resolve a released model card (and, by ID, a models.json index) into a
// `KomugiModel` for `KomugiClient`. Pure and synchronous: no fetch, no
// hashing -- the worker host already fetches and sha256-verifies the bytes
// at init. The card's `planeMapVersion` is stamped onto the resolved model
// (`KomugiModel.compatibility`) so every search the client posts is bound to
// it; `rulesProfile` and `architectureTag` are recorded only (2.2.3/2.2.4).
//
// Imports: rule_features.ts/rules_config.ts types, constants.ts, and
// type-only client.ts/client_failures.ts -- no search, engine or ORT code.

import type { PlaneMapVersion } from './rule_features.ts';
import type { RulesProfile } from './rules_config.ts';
import { ACTION_ENCODING_VERSION } from './constants.ts';
import type { KomugiModel } from './client.ts';
import type { KomugiFailure } from './client_failures.ts';

export interface ModelCompatibilityKey {
  planeMapVersion: PlaneMapVersion;
  actionEncodingVersion: number;
  rulesProfile: RulesProfile;
  architectureTag: string; // e.g. "b10c128" -- informational, 2.2.4
}
export interface ModelAsset {
  path: string;
  sha256: string;
  bytes?: number;
}
export interface ModelCard {
  id: string;
  campaign: string;
  iteration: number;
  step: number;
  compatibilityKey: ModelCompatibilityKey;
  modesTrained: string[];
  assets: { pt: ModelAsset; onnx: ModelAsset };
  sourceCheckpoint: { name: string; sha256: string };
  exportOpset: number;
  exportParity: { n: number; maxAbsDelta: number; atol: number; rtol: number };
  evalPointers: string[];
  license: string;
  createdAt: string;
}
export interface ModelsIndex {
  schemaVersion: 1;
  recommended: Partial<Record<'intro' | 'beginner' | 'intermediate' | 'advanced', string>>;
  models: Record<string, { card: string; onnxSha256: string }>;
}

// What THIS engine build expects. Only `actionEncodingVersion` is an exact
// match (section 5.1); `planeMapVersion` is a membership check in {3, 4}
// and its value is carried onto the resolved model instead.
export type EngineCompatibilityKey = Pick<ModelCompatibilityKey, 'actionEncodingVersion'>;
export const ENGINE_COMPATIBILITY_KEY: EngineCompatibilityKey = {
  actionEncodingVersion: ACTION_ENCODING_VERSION,
};

// Re-spelled from the card schema (schemas/model-card.schema.json) so this
// module stays a leaf.
const ID_PATTERN = /^komugi-[a-z0-9]+-s[0-9]+-[0-9a-f]{8}$/;
const NAME_PATTERN = /^[A-Za-z0-9._-]+$/;
const SHA256_PATTERN = /^[0-9a-f]{64}$/;
const ARCH_PATTERN = /^b[0-9]+c[0-9]+$/;
const DATE_TIME_PATTERN = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})$/;
const MODES = ['intro', 'beginner', 'intermediate', 'advanced'];
const PROFILES: readonly string[] = ['komugi', 'gungijs', 'ligungi'] satisfies readonly RulesProfile[];
const LICENSE = 'MIT, Copyright (c) 2026 chessai';

type Rec = Record<string, unknown>;

function isRec(value: unknown): value is Rec {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function has(obj: object, key: string): boolean {
  return Object.prototype.hasOwnProperty.call(obj, key);
}

function isInt(value: unknown, min: number): boolean {
  return typeof value === 'number' && Number.isInteger(value) && value >= min;
}

function isNum(value: unknown, min: number): boolean {
  return typeof value === 'number' && Number.isFinite(value) && value >= min;
}

function matches(value: unknown, pattern: RegExp): boolean {
  return typeof value === 'string' && pattern.test(value);
}

function isAsset(value: unknown): boolean {
  if (!isRec(value) || !matches(value.path, NAME_PATTERN) || !matches(value.sha256, SHA256_PATTERN)) return false;
  return !has(value, 'bytes') || isInt(value.bytes, 1);
}

function isDateTime(value: unknown): boolean {
  return matches(value, DATE_TIME_PATTERN) && !Number.isNaN(Date.parse(value as string));
}

// The first field `isModelCard` rejects, or null for a well-formed card.
function badCardField(value: unknown): string | null {
  if (!isRec(value)) return '<root>';
  if (!matches(value.id, ID_PATTERN)) return 'id';
  if (typeof value.campaign !== 'string' || value.campaign.length === 0) return 'campaign';
  if (!isInt(value.iteration, 0)) return 'iteration';
  if (!isInt(value.step, 0)) return 'step';
  const key = value.compatibilityKey;
  if (!isRec(key)) return 'compatibilityKey';
  if (key.planeMapVersion !== 3 && key.planeMapVersion !== 4) return 'compatibilityKey.planeMapVersion';
  if (!isInt(key.actionEncodingVersion, 1)) return 'compatibilityKey.actionEncodingVersion';
  if (typeof key.rulesProfile !== 'string' || !PROFILES.includes(key.rulesProfile)) return 'compatibilityKey.rulesProfile';
  if (!matches(key.architectureTag, ARCH_PATTERN)) return 'compatibilityKey.architectureTag';
  const modes = value.modesTrained;
  if (!Array.isArray(modes) || modes.length === 0 || !modes.every((m) => typeof m === 'string' && MODES.includes(m))) return 'modesTrained';
  const assets = value.assets;
  if (!isRec(assets)) return 'assets';
  if (!isAsset(assets.pt)) return 'assets.pt';
  if (!isAsset(assets.onnx)) return 'assets.onnx';
  const source = value.sourceCheckpoint;
  if (!isRec(source) || !matches(source.name, NAME_PATTERN) || !matches(source.sha256, SHA256_PATTERN)) return 'sourceCheckpoint';
  if (!isInt(value.exportOpset, 1)) return 'exportOpset';
  const parity = value.exportParity;
  if (!isRec(parity) || !isInt(parity.n, 1) || !isNum(parity.maxAbsDelta, 0) || !isNum(parity.atol, 0) || !isNum(parity.rtol, 0)) {
    return 'exportParity';
  }
  const pointers = value.evalPointers;
  if (!Array.isArray(pointers) || !pointers.every((p) => typeof p === 'string')) return 'evalPointers';
  if (value.license !== LICENSE) return 'license';
  if (!isDateTime(value.createdAt)) return 'createdAt';
  return null;
}

export function isModelCard(value: unknown): value is ModelCard {
  return badCardField(value) === null;
}

function isModelsIndex(value: unknown): value is ModelsIndex {
  if (!isRec(value) || value.schemaVersion !== 1) return false;
  const { recommended, models } = value;
  if (!isRec(recommended) || !isRec(models)) return false;
  for (const mode of Object.keys(recommended)) {
    if (!MODES.includes(mode) || typeof recommended[mode] !== 'string') return false;
  }
  for (const entry of Object.values(models)) {
    if (!isRec(entry) || !matches(entry.card, NAME_PATTERN) || !matches(entry.onnxSha256, SHA256_PATTERN)) return false;
  }
  return true;
}

function incompatible(message: string): KomugiFailure {
  return { code: 'incompatible-model', stage: 'init', message };
}

export function checkCompatibility(card: ModelCard, expected: EngineCompatibilityKey = ENGINE_COMPATIBILITY_KEY): KomugiFailure | null {
  const key = card.compatibilityKey;
  // Schema-guaranteed for a guarded card; re-checked for a cast one.
  if (key.planeMapVersion !== 3 && key.planeMapVersion !== 4) {
    return incompatible(`planeMapVersion: card says ${JSON.stringify(key.planeMapVersion)}, engine expects 3 or 4`);
  }
  if (key.actionEncodingVersion !== expected.actionEncodingVersion) {
    return incompatible(`actionEncodingVersion: card says ${JSON.stringify(key.actionEncodingVersion)}, engine expects ${expected.actionEncodingVersion}`);
  }
  return null;
}

// No fetch assumed -- the host supplies a path -> URL resolver.
export type ResolveAssetUrl = (asset: ModelAsset) => string;
export type ModelResolution = { status: 'ok'; model: KomugiModel; card: ModelCard } | { status: 'failed'; failure: KomugiFailure };

function failed(failure: KomugiFailure): ModelResolution {
  return { status: 'failed', failure };
}

function malformedCard(field: string): ModelResolution {
  return failed(incompatible(`card: ${field} missing or malformed`));
}

function resolveChecked(card: ModelCard, resolveUrl: ResolveAssetUrl, expected: EngineCompatibilityKey | undefined): ModelResolution {
  const failure = checkCompatibility(card, expected);
  if (failure !== null) return failed(failure);
  const model: KomugiModel = {
    url: resolveUrl(card.assets.onnx),
    sha256: card.assets.onnx.sha256,
    compatibility: { planeMapVersion: card.compatibilityKey.planeMapVersion },
  };
  return { status: 'ok', model, card };
}

export function resolveModelCard(card: unknown, resolveUrl: ResolveAssetUrl, expected?: EngineCompatibilityKey): ModelResolution {
  const bad = badCardField(card);
  if (bad !== null) return malformedCard(bad);
  return resolveChecked(card as ModelCard, resolveUrl, expected);
}

// `index` and `card` are the caller's own already-fetched/parsed JSON. The
// card is guarded first, then `id`, `card` and `index` are bound together
// before the compatibility check (section 5.2, review F6).
export function resolveModelById(
  id: string,
  index: unknown,
  card: unknown,
  resolveUrl: ResolveAssetUrl,
  expected?: EngineCompatibilityKey,
): ModelResolution {
  const bad = badCardField(card);
  if (bad !== null) return malformedCard(bad);
  const typed = card as ModelCard;
  if (!isModelsIndex(index)) return failed({ code: 'model-not-found', stage: 'init', message: 'models index is malformed' });
  const entry = has(index.models, id) ? index.models[id] : undefined;
  if (entry === undefined) return failed({ code: 'model-not-found', stage: 'init', message: `model ${JSON.stringify(id)} is not in the models index` });
  if (typed.id !== id) return failed(incompatible(`id: card says ${typed.id}, requested ${id}`));
  const onnxSha = typed.assets.onnx.sha256;
  if (entry.onnxSha256 !== onnxSha) {
    return failed(incompatible(`onnxSha256: index says ${entry.onnxSha256}, card says ${onnxSha}`));
  }
  const sha8 = id.slice(id.lastIndexOf('-') + 1);
  if (!onnxSha.startsWith(sha8)) return failed(incompatible(`id: sha8 ${sha8} is not a prefix of assets.onnx.sha256 ${onnxSha}`));
  return resolveChecked(typed, resolveUrl, expected);
}
