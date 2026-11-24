// U7b1: state/tree/fake evaluator against the KOM-1060 search fixture
// (docs/design/clientside-engine-port.md section 4.3, section 6 U7b1).
// KOM_SEARCH: the recorded search fixture's search.json (plain JSON, not
// JSONL -- read directly, as select.test.ts does).

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { fakeLogitAt, fakeMovesLeft, fakeWdl, makeFakeEvaluator } from '../src/fake_evaluator.ts';
import { C_START } from '../src/constants.ts';
import { boardFromFen } from '../src/fen.ts';
import { Game } from '../src/game.ts';
import { MatchNode, WDL, utility } from '../src/search/node.ts';
import { nodeUtility } from '../src/search/select.ts';
import { MatchGameState } from '../src/search/state.ts';
import { MatchTree, type MatchConfig } from '../src/search/tree.ts';
import { RepetitionRule } from '../src/repetition.ts';
import { loadSearchFixture } from './fixture.ts';

type EvaluatorVector = {
  h: number;
  sample_action_ids: number[];
  sample_logits: number[];
  wdl: number[];
  moves_left: number;
};

type SearchRecord = {
  config: MatchConfig & { discovery: boolean; draftmin: number; mlh: { lam: number; tau0: number; tau1: number; slope: number } };
  position: { fen: string; history: string[]; stamp: string; to_play: number; is_root: boolean; bias_key: number };
  result: { action: number; playouts_used: number; evaluations: number; nodes: number; mlh_gated: number };
  root: {
    actions: number[];
    prior: number[];
    logits: number[];
    visits: number[];
    weight: number[];
    wdl_sum: number[][];
    moment: number[][];
    wdl_net: number[];
  };
  intermediates: { node_utility: number };
};

type SearchFixture = { evaluator_vectors: EvaluatorVector[]; searches: SearchRecord[] };

function flatten2D(rows: number[][], cols: number): Float64Array {
  const out = new Float64Array(rows.length * cols);
  for (let i = 0; i < rows.length; i++) {
    for (let j = 0; j < cols; j++) out[i * cols + j] = rows[i][j];
  }
  return out;
}

function loadFixture(): SearchFixture {
  return loadSearchFixture<SearchFixture>();
}

test('fake evaluator matches the fixture evaluator_vectors exactly (4.3 S3)', () => {
  const fixture = loadFixture();
  assert.ok(fixture.evaluator_vectors.length > 0, 'no evaluator vectors');

  let logitsChecked = 0;
  for (const vec of fixture.evaluator_vectors) {
    for (let i = 0; i < vec.sample_action_ids.length; i++) {
      const got = fakeLogitAt(vec.h, vec.sample_action_ids[i]);
      assert.ok(
        Object.is(got, vec.sample_logits[i]),
        `h=${vec.h} action=${vec.sample_action_ids[i]}: got ${got}, want ${vec.sample_logits[i]}`,
      );
      logitsChecked++;
    }

    const wdl = fakeWdl(vec.h);
    for (let i = 0; i < WDL; i++) {
      assert.ok(Object.is(wdl[i], vec.wdl[i]), `h=${vec.h} wdl[${i}]: got ${wdl[i]}, want ${vec.wdl[i]}`);
    }

    const movesLeft = fakeMovesLeft(vec.h);
    assert.ok(
      Object.is(movesLeft, vec.moves_left),
      `h=${vec.h} moves_left: got ${movesLeft}, want ${vec.moves_left}`,
    );
  }
  assert.ok(logitsChecked > 0, 'no sample logits checked');
});

// Builds the root `MatchGameState` for a fixture record directly from its
// FEN (ignoring `history`): root reconstruction by SAN replay is root.ts's
// job (U8, which depends on U3 and U4b both), not this unit's -- see design
// section 2.5. The rule follows the record's own stamp: 'present' means the
// server replayed history under PLACEMENT, 'absent' means the history-less
// FULL fallback (design section 5 D1); either way the fresh board starts at
// one occurrence, which cannot affect these shallow (<=14 ply), low-sims
// positions.
function rootStateFor(rec: SearchRecord): MatchGameState {
  const board = boardFromFen(rec.position.fen);
  const rule = rec.position.stamp === 'present' ? RepetitionRule.PLACEMENT : RepetitionRule.FULL;
  const game = new Game(board, rule);
  return new MatchGameState(game, { minPlacements: rec.config.draftmin, isRoot: true });
}

test('MatchTree playouts reproduce exact root statistics on lam-0/discovery-off records (U7b1)', () => {
  const fixture = loadFixture();
  const records = fixture.searches.filter(
    (rec) => rec.config.discovery === false && rec.config.draftmin === 0 && rec.config.mlh.lam === 0,
  );
  assert.ok(records.length > 0, 'no lam-0/discovery-off records in the fixture');

  let checked = 0;
  for (const rec of records) {
    const state = rootStateFor(rec);
    const evaluator = makeFakeEvaluator(rec.config.num_actions);
    const tree = new MatchTree(rec.config, evaluator);
    tree.ensureRoot(state);
    for (let i = 0; i < rec.result.playouts_used; i++) tree.playout(state);

    const root = tree.root;
    assert.ok(root !== null, `${rec.position.fen}: tree has no root after ensureRoot`);
    const node = root as MatchNode;

    assert.deepEqual(
      Array.from(node.actions),
      rec.root.actions,
      `${rec.position.fen}: root.actions mismatch`,
    );
    for (let i = 0; i < node.width; i++) {
      assert.ok(
        Object.is(node.visits[i], rec.root.visits[i]),
        `${rec.position.fen}: visits[${i}] got ${node.visits[i]}, want ${rec.root.visits[i]}`,
      );
      assert.ok(
        Object.is(node.weight[i], rec.root.weight[i]),
        `${rec.position.fen}: weight[${i}] got ${node.weight[i]}, want ${rec.root.weight[i]}`,
      );
    }
    const wantWdlSum = flatten2D(rec.root.wdl_sum, WDL);
    for (let i = 0; i < wantWdlSum.length; i++) {
      assert.ok(
        Object.is(node.wdlSum[i], wantWdlSum[i]),
        `${rec.position.fen}: wdl_sum[${i}] got ${node.wdlSum[i]}, want ${wantWdlSum[i]}`,
      );
    }
    const wantMoment = flatten2D(rec.root.moment, 6);
    for (let i = 0; i < wantMoment.length; i++) {
      assert.ok(
        Object.is(node.moment[i], wantMoment[i]),
        `${rec.position.fen}: moment[${i}] got ${node.moment[i]}, want ${wantMoment[i]}`,
      );
    }

    const gotUtility = nodeUtility(node, rec.config);
    assert.ok(
      Object.is(gotUtility, rec.intermediates.node_utility),
      `${rec.position.fen}: node_utility got ${gotUtility}, want ${rec.intermediates.node_utility}`,
    );

    assert.equal(tree.evaluations, rec.result.evaluations, `${rec.position.fen}: evaluations`);
    assert.equal(tree.nodeCount, rec.result.nodes, `${rec.position.fen}: node count`);
    checked++;
  }

  console.log(`tree playouts: lam-0/discovery-off records checked=${checked}`);
  assert.ok(checked > 0, 'no records checked');
});

test('the DRAFTMIN root mask applies only at depth 0 (design section 2.6 S5)', () => {
  const fixture = loadFixture();
  const draftRoot = fixture.searches.find(
    (rec) => rec.config.draftmin > 0 && rec.position.history.length === 0,
  );
  assert.ok(draftRoot !== undefined, 'no DRAFTMIN draft-root record (history-less, draftmin > 0) in the fixture');
  const rec = draftRoot as SearchRecord;

  const board = boardFromFen(rec.position.fen);
  const game = new Game(board, RepetitionRule.PLACEMENT);
  const rootState = new MatchGameState(game, { minPlacements: rec.config.draftmin, isRoot: true });

  const rawMask = game.validMoves();
  let anyWithheld = false;
  for (let i = C_START; i < rawMask.length; i++) {
    if (rawMask[i]) {
      anyWithheld = true;
      break;
    }
  }
  assert.ok(anyWithheld, 'fixture draft root has no C-block action to withhold: test setup is wrong');

  assert.equal(rootState.depth, 0, 'a freshly constructed root state must be at depth 0');
  const maskedAtRoot = rootState.legalMask();
  for (let i = C_START; i < maskedAtRoot.length; i++) {
    assert.equal(maskedAtRoot[i], 0, `legalMask()[${i}]: DRAFTMIN must withhold every C-block action at the root`);
  }

  let firstLegal = -1;
  for (let i = 0; i < maskedAtRoot.length; i++) {
    if (maskedAtRoot[i]) {
      firstLegal = i;
      break;
    }
  }
  assert.ok(firstLegal >= 0, 'the masked root has no legal action at all');

  rootState.play(firstLegal);
  assert.equal(rootState.depth, 1, 'play() must move the state away from depth 0');
  assert.deepEqual(
    Array.from(rootState.legalMask()),
    Array.from(game.validMoves()),
    'off the root (depth !== 0), legalMask() must be the engine\'s raw mask with no DRAFTMIN withholding',
  );
});

// A mutant check (design 4.3 M3's "pins this"): subtree_wdl_sum is a
// SEQUENTIAL row loop per column (node.ts's sumRowsSequential), matching
// numpy's axis=0 reduction over a small 2D array -- not a pairwise/blocked
// reduction. This mutant recomputes the same rows with a recursive pairwise
// halving instead, and the exact (`Object.is`) node_utility comparison
// above is only meaningful if that wrong choice is actually detectable: at
// least one recorded root must disagree between the two reductions.
function mutantSubtreeWdlSum(wdlSum: Float64Array, width: number): Float64Array {
  function sumCol(col: number, lo: number, hi: number): number {
    const n = hi - lo;
    if (n === 1) return wdlSum[lo * WDL + col];
    const mid = lo + Math.floor(n / 2);
    return sumCol(col, lo, mid) + sumCol(col, mid, hi);
  }
  const out = new Float64Array(WDL);
  if (width > 0) for (let c = 0; c < WDL; c++) out[c] = sumCol(c, 0, width);
  return out;
}

function mutantNodeUtility(node: MatchNode, contempt: number): number {
  const total = node.totalWeight();
  if (total <= 0.0) return utility(node.wdlNet, contempt);
  return utility(mutantSubtreeWdlSum(node.wdlSum, node.width), contempt) / total;
}

test('a pairwise mutant of subtree_wdl_sum disagrees with the recorded node_utility somewhere (4.3 M3)', () => {
  const fixture = loadFixture();
  assert.ok(fixture.searches.length > 0, 'no search records');

  let mismatchFound = false;
  for (const rec of fixture.searches) {
    const node = new MatchNode({
      toPlay: rec.position.to_play,
      isTerminal: false,
      actions: Int32Array.from(rec.root.actions),
      logits: Float32Array.from(rec.root.logits),
      wdlNet: Float64Array.from(rec.root.wdl_net),
      biasKey: rec.position.bias_key,
      isRoot: rec.position.is_root,
    });
    node.weight = Float64Array.from(rec.root.weight);
    node.wdlSum = flatten2D(rec.root.wdl_sum, WDL);

    const got = nodeUtility(node, rec.config);
    assert.ok(
      Object.is(got, rec.intermediates.node_utility),
      `${rec.position.fen}: the real (sequential) node_utility must match exactly`,
    );

    const mutant = mutantNodeUtility(node, rec.config.contempt);
    if (!Object.is(mutant, rec.intermediates.node_utility)) mismatchFound = true;
  }

  assert.ok(
    mismatchFound,
    'the pairwise-reduction mutant agreed with every recorded node_utility: the exact comparison ' +
      'has no power to catch a wrong reduction order here',
  );
});
