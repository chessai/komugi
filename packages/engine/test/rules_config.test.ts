// KOM-1203 unit 1 (docs/design/kom1203-rules-config.md section 3): the
// RulesConfig core -- RULES_PRESETS, toRulesConfig, rulesConfigKey and
// validateRulesConfig's coherence/F7/F8/N2 rules -- in isolation, with no
// WorkerEngine involved (that lives in unit 5's rules_config_engine.test.ts,
// N5).

import assert from 'node:assert/strict';
import { test } from 'node:test';

import {
  DEFAULT_RULES_PROFILE,
  RULES_PRESETS,
  RULES_PROFILES,
  type RulesConfig,
  rulesConfigKey,
  toRulesConfig,
} from '../src/rules_config.ts';

test('the three presets have pairwise-distinct rulesConfigKeys', () => {
  const keys = RULES_PROFILES.map((p) => rulesConfigKey(RULES_PRESETS[p]));
  assert.equal(new Set(keys).size, RULES_PROFILES.length);
});

test('toRulesConfig resolves each profile string to its preset singleton', () => {
  for (const p of RULES_PROFILES) {
    assert.equal(toRulesConfig(p), RULES_PRESETS[p]);
  }
});

test('toRulesConfig(undefined) returns the default preset singleton', () => {
  assert.equal(toRulesConfig(undefined), RULES_PRESETS[DEFAULT_RULES_PROFILE]);
});

test('toRulesConfig rejects an unknown profile string', () => {
  assert.throws(() => toRulesConfig('not-a-profile' as never), /rules must be one of/);
});

test('toRulesConfig rejects plyCap: 300 (F7)', () => {
  assert.throws(
    () => toRulesConfig({ ...RULES_PRESETS.ligungi, plyCap: 300 } as unknown as RulesConfig),
    /plyCap/,
  );
});

test('toRulesConfig rejects an unknown key (F8)', () => {
  assert.throws(
    () => toRulesConfig({ ...RULES_PRESETS.komugi, extra: true } as unknown as RulesConfig),
    /unknown key "extra"/,
  );
});

test('toRulesConfig rejects an empty object -- every knob undefined, not a silent hybrid (N2)', () => {
  assert.throws(() => toRulesConfig(Object.freeze({}) as unknown as RulesConfig));
});

test('toRulesConfig rejects a host-frozen incoherent pair even though it is frozen (N2)', () => {
  assert.throws(
    () =>
      toRulesConfig(
        Object.freeze({ ...RULES_PRESETS.ligungi, marshalCaptureIsEscape: false }) as unknown as RulesConfig,
      ),
    /selfExposureFilter requires marshalCaptureIsEscape/,
  );
});

test('toRulesConfig rejects the incoherent pair from an unfrozen literal too (F3)', () => {
  assert.throws(
    () => toRulesConfig({ ...RULES_PRESETS.ligungi, marshalCaptureIsEscape: false }),
    /selfExposureFilter requires marshalCaptureIsEscape/,
  );
});

test('F7: only 512 and null are accepted plyCap values', () => {
  for (const bad of [300, 0, -1, 1.5]) {
    assert.throws(
      () => toRulesConfig({ ...RULES_PRESETS.komugi, plyCap: bad } as unknown as RulesConfig),
      /plyCap/,
    );
  }
  assert.doesNotThrow(() => toRulesConfig({ ...RULES_PRESETS.komugi, plyCap: 512 }));
  assert.doesNotThrow(() => toRulesConfig({ ...RULES_PRESETS.ligungi, plyCap: null }));
});

test('F8: a freshly-built literal equal to a preset resolves to that preset singleton, not a copy', () => {
  for (const p of RULES_PROFILES) {
    const literal: RulesConfig = { ...RULES_PRESETS[p] };
    assert.equal(toRulesConfig(literal), RULES_PRESETS[p]);
    assert.notEqual(literal, RULES_PRESETS[p]); // the input itself was a distinct object
  }
});

test('a genuinely custom config resolves to a new frozen object, not any preset', () => {
  const custom: RulesConfig = {
    selfExposureFilter: false,
    archerElbowBlocking: true, // komugi + archer-elbow-blocking only: not a preset's exact combination
    tacticianBetrayalMode: 'any-subset',
    stalemateResult: 'draw',
    marshalCaptureIsEscape: true,
    plyCap: 512,
    repetitionKey: 'placement',
  };
  const resolved = toRulesConfig(custom);
  assert.deepEqual(resolved, custom);
  assert.ok(Object.isFrozen(resolved));
  for (const p of RULES_PROFILES) assert.notEqual(resolved, RULES_PRESETS[p]);
});

test('toRulesConfig is idempotent on an already-resolved RulesConfig (fast path, N2)', () => {
  const custom: RulesConfig = {
    selfExposureFilter: false,
    archerElbowBlocking: true,
    tacticianBetrayalMode: 'any-subset',
    stalemateResult: 'draw',
    marshalCaptureIsEscape: true,
    plyCap: 512,
    repetitionKey: 'placement',
  };
  const resolved = toRulesConfig(custom);
  assert.equal(toRulesConfig(resolved), resolved);
});

test('each field is type-checked individually', () => {
  const base = RULES_PRESETS.komugi;
  assert.throws(() => toRulesConfig({ ...base, selfExposureFilter: 'yes' } as unknown as RulesConfig), /selfExposureFilter/);
  assert.throws(() => toRulesConfig({ ...base, archerElbowBlocking: 1 } as unknown as RulesConfig), /archerElbowBlocking/);
  assert.throws(() => toRulesConfig({ ...base, tacticianBetrayalMode: 'bogus' } as unknown as RulesConfig), /tacticianBetrayalMode/);
  assert.throws(() => toRulesConfig({ ...base, stalemateResult: 'bogus' } as unknown as RulesConfig), /stalemateResult/);
  assert.throws(() => toRulesConfig({ ...base, marshalCaptureIsEscape: 0 } as unknown as RulesConfig), /marshalCaptureIsEscape/);
  assert.throws(() => toRulesConfig({ ...base, repetitionKey: 'bogus' } as unknown as RulesConfig), /repetitionKey/);
});

test('rulesConfigKey is order-independent field-wise (same fields, same key)', () => {
  const a = RULES_PRESETS.ligungi;
  const b: RulesConfig = {
    repetitionKey: a.repetitionKey,
    plyCap: a.plyCap,
    marshalCaptureIsEscape: a.marshalCaptureIsEscape,
    stalemateResult: a.stalemateResult,
    tacticianBetrayalMode: a.tacticianBetrayalMode,
    archerElbowBlocking: a.archerElbowBlocking,
    selfExposureFilter: a.selfExposureFilter,
  };
  assert.equal(rulesConfigKey(a), rulesConfigKey(b));
});
