// Host-supplied RulesConfig (docs/design/kom1203-rules-config.md section 3):
// widens the engine's closed RulesProfile string enum to also accept a
// typed, host-constructible object built from the same knobs the three
// public profiles already differ on (section 2's table). This module is a
// **leaf** (F6): no value import of repetition.ts or anything reaching
// movegen -- repetition.ts value-imports movegen.ts, and if this module
// value-imported repetition.ts instead, a test importing rules_config.ts
// first could hit a TDZ ReferenceError evaluating a movegen module-level
// holder before RULES_PRESETS exists. It owns RulesProfile/RULES_PROFILES/
// DEFAULT_RULES_PROFILE (moved out of repetition.ts, which re-exports them).
//
// Section 4's engine-scoped scoreTerminal hook types also live here (not
// worker_host.ts): game.ts must not import worker_host.ts, even `import
// type` (the import-graph walker counts type-only imports too), so the
// hook's types are defined where game.ts already lives in the closure, and
// worker_host.ts/engine.ts just reuse them. `import type { Board }` is a
// type-only import of a module (board.ts) already in the rules closure, so
// this stays a leaf: no new module enters the closure.

import type { Board } from './board.ts';

export type RulesProfile = 'komugi' | 'gungijs' | 'ligungi';
export const RULES_PROFILES = ['komugi', 'gungijs', 'ligungi'] as const satisfies readonly RulesProfile[];
export const DEFAULT_RULES_PROFILE: RulesProfile = 'komugi';

export type TacticianBetrayalMode = 'any-subset' | 'all-or-nothing';
export type StalemateResult = 'draw' | 'loss-for-mover';
export type RepetitionKeyMode = 'placement' | 'full';
// F7: the only non-null value any shipped profile uses is 512 -- not an open
// numeric range (section 2's "boolean or closed string union" claim must
// hold for every row, including this one).
export type PlyCap = 512 | null;

export interface RulesConfig {
  readonly selfExposureFilter: boolean;
  readonly archerElbowBlocking: boolean;
  readonly tacticianBetrayalMode: TacticianBetrayalMode;
  readonly stalemateResult: StalemateResult;
  readonly marshalCaptureIsEscape: boolean;
  readonly plyCap: PlyCap;
  readonly repetitionKey: RepetitionKeyMode;
}
// Section 4's host hook is deliberately NOT a field here: it cannot cross
// postMessage (functions throw DataCloneError under structured clone), so
// it is never part of the wire RulesConfig -- see WorkerHostOptions/
// WorkerEngine's own `scoreTerminal` constructor option instead (unit 6).

// N3: the reason winState() decided a position is terminal -- the ply-cap
// return now carries 'ply-cap' into the hook instead of returning early with
// no hook call.
export type TerminalReason =
  | 'checkmate'
  | 'stalemate'
  | 'repetition'
  | 'insufficient'
  | 'marshal-captured'
  | 'ply-cap';

// N4: named alias, always absolute (white-perspective), never mover-relative
// -- terminalWdl() (search/state.ts) is mover-relative W/D/L; a hook must
// not reuse that frame.
export type AbsoluteWdl = readonly [white: number, black: number, draw: number];

export interface TerminalScoreContext {
  readonly board: Board; // a read-only copy (Board.copy(), N7), never the live search board
  readonly turns: number;
  readonly reason: TerminalReason;
}

// N3: the hook's type, defined once here; WorkerHostOptions (worker_host.ts)
// reuses this type rather than declaring its own.
export type ScoreTerminalHook = (
  ctx: TerminalScoreContext,
  defaultResult: AbsoluteWdl,
) => AbsoluteWdl | null | undefined;

const RULES_CONFIG_FIELDS = [
  'selfExposureFilter', 'archerElbowBlocking', 'tacticianBetrayalMode',
  'stalemateResult', 'marshalCaptureIsEscape', 'plyCap', 'repetitionKey',
] as const;

export const RULES_PRESETS: Readonly<Record<RulesProfile, RulesConfig>> = {
  komugi:  Object.freeze({ selfExposureFilter: false, archerElbowBlocking: false, tacticianBetrayalMode: 'any-subset',     stalemateResult: 'draw',           marshalCaptureIsEscape: true,  plyCap: 512,  repetitionKey: 'placement' }),
  gungijs: Object.freeze({ selfExposureFilter: false, archerElbowBlocking: false, tacticianBetrayalMode: 'any-subset',     stalemateResult: 'draw',           marshalCaptureIsEscape: false, plyCap: 512,  repetitionKey: 'placement' }),
  ligungi: Object.freeze({ selfExposureFilter: true,  archerElbowBlocking: true,  tacticianBetrayalMode: 'all-or-nothing', stalemateResult: 'loss-for-mover', marshalCaptureIsEscape: true,  plyCap: null, repetitionKey: 'full' }),
};

// N2: membership, not Object.isFrozen, is the toRulesConfig fast-path test --
// a host can Object.freeze() an arbitrary unvalidated object, so frozen-ness
// alone cannot stand in for "this module already validated it". Populated
// below with the three presets; validateRulesConfig adds every object it
// returns.
const KNOWN_RULES_CONFIGS = new WeakSet<RulesConfig>();
for (const preset of Object.values(RULES_PRESETS)) KNOWN_RULES_CONFIGS.add(preset);

// F3: selfExposureFilter without marshalCaptureIsEscape is incoherent --
// ligungi_filter.ts:494 always treats a marshal capture as legal, which is
// sound only when marshalCaptureIsEscape is true; a config asking for the
// filter without the escape would get divergent movegen (section 5's F3 fix
// closes the one branch that skips the filter, but this rejects the
// combination outright rather than relying on that alone).
// F8: unknown keys are a REJECT, not a silently-ignored extra field -- a
// newer host sending a knob this build doesn't know must fail loud, not play
// silently under a wrong rule.
function validateRulesConfig(rules: Readonly<Record<string, unknown>>): RulesConfig {
  for (const key of Object.keys(rules)) {
    if (!(RULES_CONFIG_FIELDS as readonly string[]).includes(key)) {
      throw new Error(`RulesConfig: unknown key ${JSON.stringify(key)}`);
    }
  }
  if (typeof rules.selfExposureFilter !== 'boolean') {
    throw new Error(`RulesConfig: selfExposureFilter must be a boolean, got ${JSON.stringify(rules.selfExposureFilter)}`);
  }
  if (typeof rules.archerElbowBlocking !== 'boolean') {
    throw new Error(`RulesConfig: archerElbowBlocking must be a boolean, got ${JSON.stringify(rules.archerElbowBlocking)}`);
  }
  if (rules.tacticianBetrayalMode !== 'any-subset' && rules.tacticianBetrayalMode !== 'all-or-nothing') {
    throw new Error(`RulesConfig: tacticianBetrayalMode must be 'any-subset'|'all-or-nothing', got ${JSON.stringify(rules.tacticianBetrayalMode)}`);
  }
  if (rules.stalemateResult !== 'draw' && rules.stalemateResult !== 'loss-for-mover') {
    throw new Error(`RulesConfig: stalemateResult must be 'draw'|'loss-for-mover', got ${JSON.stringify(rules.stalemateResult)}`);
  }
  if (typeof rules.marshalCaptureIsEscape !== 'boolean') {
    throw new Error(`RulesConfig: marshalCaptureIsEscape must be a boolean, got ${JSON.stringify(rules.marshalCaptureIsEscape)}`);
  }
  if (rules.plyCap !== 512 && rules.plyCap !== null) {
    throw new Error(`RulesConfig: plyCap must be 512|null, got ${JSON.stringify(rules.plyCap)}`);
  }
  if (rules.repetitionKey !== 'placement' && rules.repetitionKey !== 'full') {
    throw new Error(`RulesConfig: repetitionKey must be 'placement'|'full', got ${JSON.stringify(rules.repetitionKey)}`);
  }
  const candidate = {
    selfExposureFilter: rules.selfExposureFilter,
    archerElbowBlocking: rules.archerElbowBlocking,
    tacticianBetrayalMode: rules.tacticianBetrayalMode,
    stalemateResult: rules.stalemateResult,
    marshalCaptureIsEscape: rules.marshalCaptureIsEscape,
    plyCap: rules.plyCap,
    repetitionKey: rules.repetitionKey,
  } as RulesConfig;
  if (candidate.selfExposureFilter && !candidate.marshalCaptureIsEscape) {
    throw new Error('RulesConfig: selfExposureFilter requires marshalCaptureIsEscape');
  }
  // F8: a config equal in value to a preset resolves to that preset's
  // singleton object, so the bias-reset/ponder-key fast paths (section 5)
  // that compare by identity still fire for a host that happened to rebuild
  // komugi's own values from scratch.
  for (const preset of Object.values(RULES_PRESETS)) {
    if (rulesConfigKey(candidate) === rulesConfigKey(preset)) return preset;
  }
  const resolved = Object.freeze({ ...candidate }); // fixed field order (declaration order above), frozen
  KNOWN_RULES_CONFIGS.add(resolved); // N2: marks this object as validated for toRulesConfig's fast path
  return resolved;
}

// Public entry points call this ONCE, at the boundary (F1); every internal
// helper below this point takes RulesConfig only, never RulesProfile.
// A string resolves to the RULES_PRESETS singleton (object identity, section
// 5's bias-reset fast path). An object is validated, rejected on any unknown
// key or incoherent combination, and canonicalised onto the matching preset
// singleton when its knobs are equal to one (F8).
export function toRulesConfig(rules: RulesProfile | RulesConfig | undefined): RulesConfig {
  if (rules === undefined) return RULES_PRESETS[DEFAULT_RULES_PROFILE];
  if (typeof rules === 'string') {
    if (!(RULES_PROFILES as readonly string[]).includes(rules)) {
      throw new Error(`rules must be one of 'komugi'|'gungijs'|'ligungi' or a RulesConfig object, got ${JSON.stringify(rules)}`);
    }
    return RULES_PRESETS[rules];
  }
  // N2: fast path is WeakSet membership (this module already validated/built
  // this exact object), never Object.isFrozen -- a host-frozen object that
  // never went through validateRulesConfig is not in the set and still gets
  // fully re-validated below.
  if (KNOWN_RULES_CONFIGS.has(rules as RulesConfig)) return rules as RulesConfig;
  return validateRulesConfig(rules as unknown as Record<string, unknown>);
}

// Structural equality key, fixed field order (F8), used wherever section 5
// needs a stable, order-independent comparison (bias-reset, ponder key) --
// never JSON.stringify on an arbitrary-order object (F2(c)).
export function rulesConfigKey(rules: RulesConfig): string {
  return [
    rules.selfExposureFilter, rules.archerElbowBlocking, rules.tacticianBetrayalMode,
    rules.stalemateResult, rules.marshalCaptureIsEscape, rules.plyCap, rules.repetitionKey,
  ].join('|');
}

// Validates an optional request `rules` field: absent defaults to
// DEFAULT_RULES_PROFILE, any other value throws (worker.ts validates this
// before both the ponder cache lookup and buildRoot). Kept here (not
// repetition.ts) now that it is typed against this module's RulesProfile;
// repetition.ts re-exports it unchanged.
export function parseRulesProfile(rules: unknown): RulesProfile {
  if (rules === undefined) return DEFAULT_RULES_PROFILE;
  if ((RULES_PROFILES as readonly unknown[]).includes(rules)) return rules as RulesProfile;
  throw new Error(`rules must be one of 'komugi'|'gungijs'|'ligungi', got ${JSON.stringify(rules)}`);
}
