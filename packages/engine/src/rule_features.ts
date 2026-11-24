// KOM-1162: the v4 rule-feature planes (docs/design/multi-mode-training.md
// rev 3, sections 1.2 and 5.1). The table is TRANSCRIBED BY HAND from the
// design's 1.2 table -- never imported or generated from the Python side
// (review G1: no Python crosses into the TS engine).
//
//   plane 121  PLANE_RULE_TIER_CAP_2        the game's max tier is 2
//   plane 122  PLANE_RULE_MARSHAL_NO_STACK  the marshal may not stack
//   plane 123  PLANE_RULE_FIXED_SETUP       the game started from a fixed setup
//   plane 124  PLANE_RULE_LEGACY_PROFILE    reserved, never filled in v4
//   125-127    spare, always 0
//
// Every feature is a binary deviation from Advanced, so a v4 Advanced
// observation is byte-equal to its v3 observation. Constant planes, like
// 112-115.

import { MODE_ADVANCED, MODE_BEGINNER, MODE_INTERMEDIATE, MODE_INTRO, NUM_SQUARES } from './constants.ts';

export const PLANE_RULE_TIER_CAP_2 = 121;
export const PLANE_RULE_MARSHAL_NO_STACK = 122;
export const PLANE_RULE_FIXED_SETUP = 123;
export const PLANE_RULE_LEGACY_PROFILE = 124;

// The observation's plane-map version: 3 leaves 121-127 at 0 (today's
// encoding, the default); 4 fills 121-123 from RULE_FEATURES. Chosen by
// the model's sidecar, never inferred.
export type PlaneMapVersion = 3 | 4;
export const DEFAULT_PLANE_MAP_VERSION: PlaneMapVersion = 3;

// mode -> (tier_cap_2, marshal_no_stack, fixed_setup), planes 121/122/123.
export const RULE_FEATURES: ReadonlyMap<number, readonly [number, number, number]> = new Map([
  [MODE_INTRO, [1, 1, 1] as const],
  [MODE_BEGINNER, [1, 1, 1] as const],
  [MODE_INTERMEDIATE, [1, 0, 0] as const],
  [MODE_ADVANCED, [0, 0, 0] as const],
]);

// Validates an optional `planeMapVersion` (request field or encoder
// option): absent defaults to DEFAULT_PLANE_MAP_VERSION, any value other
// than 3 or 4 throws -- the same contract as repetition.ts's
// parseRulesProfile.
export function parsePlaneMapVersion(version: unknown): PlaneMapVersion {
  if (version === undefined) return DEFAULT_PLANE_MAP_VERSION;
  if (version === 3 || version === 4) return version;
  throw new Error(`planeMapVersion must be 3 or 4, got ${JSON.stringify(version)}`);
}

// Writes planes 121-123 for `mode` under `planeMapVersion`. v3 writes
// nothing (the caller's cleared buffer already holds 0 there); v4 fills the
// three constant planes from RULE_FEATURES. 124-127 are never written.
export function fillRulePlanes(out: Float32Array, mode: number, planeMapVersion: PlaneMapVersion): void {
  const version = parsePlaneMapVersion(planeMapVersion);
  if (version === 3) return;
  const features = RULE_FEATURES.get(mode);
  if (features === undefined) throw new Error(`no rule features for mode ${mode}`);
  for (let k = 0; k < 3; k++) {
    const base = (PLANE_RULE_TIER_CAP_2 + k) * NUM_SQUARES;
    for (let i = 0; i < NUM_SQUARES; i++) out[base + i] = features[k]!;
  }
}
