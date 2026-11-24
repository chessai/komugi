// KOM-1195 unit U2 (docs/design/kom1195-analysis-api.md section 5/9):
// rankRootMoves is a pure sort over the same primitives selectAction already
// uses (lcbEligible/lcbScores/weight), so this checks it directly against a
// real root built by a 96-playout search over every committed non-terminal
// parity position -- no second search, no fixture env var required (the
// committed parity.jsonl.gz sample via test/fixture.ts's loadFixture).

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { WorkerEngine } from '../src/engine.ts';
import type { WorkerRequest } from '../src/protocol.ts';
import { rankRootMoves, selectAction } from '../src/search/select.ts';
import { loadFixture } from './fixture.ts';

test('rankRootMoves: entry 0 equals selectAction (no allowed), every visited edge precedes every unvisited one', async () => {
  const positions = loadFixture().filter((rec) => !rec.terminal);
  assert.ok(positions.length > 0, 'no non-terminal parity positions');

  let checked = 0;
  for (const pos of positions) {
    const engine = new WorkerEngine({ clock: () => 0 });
    engine.handlePonderSettings({ id: 'kom1195-rank', type: 'ponderSettings', enabled: false });
    const evaluator = engine.defaultBatchEvaluator();
    const request: WorkerRequest = { id: `rank-${pos.id}`, fen: pos.fen, sims: 96, batch: 1 };
    const reply = await engine.handleFast(request, evaluator);
    assert.ok(!('err' in reply), `${pos.id}: handleFast failed: ${JSON.stringify(reply)}`);

    const tree = engine.tree;
    assert.ok(tree !== null, `${pos.id}: no tree after search`);
    const root = tree!.root;
    assert.ok(root !== null, `${pos.id}: no root after search`);
    const config = tree!.config;
    const bias = tree!.bias;

    const order = rankRootMoves(root!, config, bias);
    assert.equal(order.length, root!.width, `${pos.id}: rankRootMoves must return one index per edge`);
    assert.deepEqual(
      [...order].sort((a, b) => a - b),
      Array.from({ length: root!.width }, (_, i) => i),
      `${pos.id}: rankRootMoves must be a permutation of the edge indices`,
    );

    const want = selectAction(root!, config, { bias });
    assert.equal(order[0], want, `${pos.id}: entry 0 must equal selectAction's own choice`);

    let sawUnvisited = false;
    for (const idx of order) {
      const visited = root!.weight[idx]! > 0.0;
      if (!visited) {
        sawUnvisited = true;
      } else {
        assert.ok(!sawUnvisited, `${pos.id}: visited edge ${idx} sorted after an unvisited one`);
      }
    }
    checked++;
  }

  console.log(`kom1195_rank: checked=${checked}`);
  assert.ok(checked > 0, 'no positions checked');
});
