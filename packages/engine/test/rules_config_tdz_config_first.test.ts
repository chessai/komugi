// KOM-1203 F6: rules_config.ts is a leaf (no value import of repetition.ts
// or anything reaching movegen) precisely so that importing it first, before
// any movegen module has had a chance to populate its own module-level
// RulesConfig holders, cannot hit a TDZ ReferenceError. This file pins that
// import order: rules_config.ts first, movegen second.

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { RULES_PRESETS } from '../src/rules_config.ts';
import { Board } from '../src/board.ts';
import { generateRecords } from '../src/movegen.ts';

test('TDZ: importing rules_config.ts before movegen.ts, then calling generateRecords, does not throw', () => {
  const board = new Board();
  assert.doesNotThrow(() => generateRecords(board, RULES_PRESETS.komugi));
});
