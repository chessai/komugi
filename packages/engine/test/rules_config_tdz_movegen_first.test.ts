// KOM-1203 F6: the reverse import order from
// rules_config_tdz_config_first.test.ts -- movegen_arata.ts first, then
// rules_config.ts. Since rules_config.ts is a leaf, neither order can hit a
// TDZ ReferenceError on the other module's holders.

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { forEachArataMove } from '../src/movegen_arata.ts';
import { RULES_PRESETS } from '../src/rules_config.ts';
import { Board } from '../src/board.ts';

test('TDZ: importing movegen_arata.ts before rules_config.ts, then reading RULES_PRESETS, does not throw', () => {
  assert.doesNotThrow(() => {
    const board = new Board();
    forEachArataMove(board, () => {}, RULES_PRESETS.komugi);
  });
});
