// KOM-1193a: the refutation survey (search.ts's TacticalSurvey) now plays
// its probes on `state.game` with make/undo instead of cloning per probe.
// This checks the swap is behaviour-preserving: a clone-based reference
// (the algorithm TacticalSurvey used before KOM-1193a, kept here only) must
// agree with the real TacticalSurvey on root witness, allowed mask, refuted
// list and the call/transition/truncated bookkeeping, on several post-draft
// positions pulled from the KOM_SEARCH fixture -- and the root game must come
// out of both passes exactly as it went in.

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { boardFromFen } from '../src/fen.ts';
import { Game } from '../src/game.ts';
import { RepetitionRule } from '../src/repetition.ts';
import { MatchNode } from '../src/search/node.ts';
import { DEFAULT_TRANSITION_CAP, TacticalSurvey } from '../src/search/search.ts';
import { MatchGameState } from '../src/search/state.ts';
import { loadSearchFixture } from './fixture.ts';

type SearchRecord = {
  config: { draftmin: number };
  position: { fen: string; stamp: string };
  result: { tactical: { skipped_reason: string | null } | null };
};

type SearchFixture = { searches: SearchRecord[] };

function loadFixture(): SearchFixture {
  return loadSearchFixture<SearchFixture>();
}

// Exactly rootStateFor (search.test.ts): the root MatchGameState straight
// from the record's FEN.
function rootStateFor(rec: SearchRecord): MatchGameState {
  const board = boardFromFen(rec.position.fen);
  const rule = rec.position.stamp === 'present' ? RepetitionRule.PLACEMENT : RepetitionRule.FULL;
  const game = new Game(board, rule);
  return new MatchGameState(game, { minPlacements: rec.config.draftmin, isRoot: true });
}

function legalActionsOf(state: MatchGameState): Int32Array {
  const mask = state.legalMask();
  const out: number[] = [];
  for (let i = 0; i < mask.length; i++) if (mask[i]) out.push(i);
  return Int32Array.from(out);
}

function rootNodeFor(state: MatchGameState, legal: Int32Array): MatchNode {
  return new MatchNode({
    toPlay: state.toPlay(),
    isTerminal: false,
    actions: legal,
    logits: new Float32Array(legal.length),
    wdlNet: new Float64Array(3),
    biasKey: 0,
  });
}

function winnerOf(state: MatchGameState): number | null {
  const wdl = state.terminalWdl();
  if (wdl[0] === 1.0) return state.toPlay();
  if (wdl[2] === 1.0) return 1 - state.toPlay();
  return null;
}

type ReferenceResult = {
  rootWitness: number | null;
  allowed: Uint8Array;
  refuted: number[];
  calls: number;
  transitions: number;
  truncated: boolean;
};

// The pre-KOM-1193a algorithm: every probe on `state.clone()`, discarded
// after one play. Kept here, and only here, as the behavioural oracle the
// make/undo TacticalSurvey above must match.
function referenceSurvey(state: MatchGameState, root: MatchNode, cap: number): ReferenceResult {
  let calls = 0;
  let transitions = 0;
  const mover = state.toPlay();
  const legal = legalActionsOf(state);
  let rootWitness: number | null = null;
  for (let i = 0; i < legal.length; i++) {
    const action = legal[i];
    const probe = state.clone();
    probe.play(action);
    transitions += 1;
    if (!probe.isTerminal()) continue;
    calls += 1;
    if (winnerOf(probe) === mover) {
      rootWitness = action;
      break;
    }
  }

  const allowed = new Uint8Array(root.width).fill(1);
  const refuted: number[] = [];
  let spent = 0;
  let truncated = false;
  const opponent = 1 - mover;
  for (let index = 0; index < root.width; index++) {
    if (spent >= cap) {
      truncated = true;
      break;
    }
    const action = root.actions[index];
    const answered = state.clone();
    answered.play(action);
    transitions += 1;
    spent += 1;
    if (answered.isTerminal()) continue;
    const replies = legalActionsOf(answered);
    for (let r = 0; r < replies.length; r++) {
      if (spent >= cap) {
        truncated = true;
        break;
      }
      const reply = replies[r];
      const refutation = answered.clone();
      refutation.play(reply);
      transitions += 1;
      spent += 1;
      if (refutation.isTerminal()) {
        calls += 1;
        if (winnerOf(refutation) === opponent) {
          allowed[index] = 0;
          refuted.push(action);
          break;
        }
      }
    }
  }
  return { rootWitness, allowed, refuted, calls, transitions, truncated };
}

type GameSnapshot = {
  cells: Int8Array;
  hand: Uint8Array;
  turn: number;
  key: Uint8Array;
  repetitionCount: number;
  undoDepth: number;
};

function snapshotGame(game: Game): GameSnapshot {
  const internal = game as unknown as { key: Uint8Array; undoStack: unknown[] };
  return {
    cells: Int8Array.from(game.board.cells),
    hand: Uint8Array.from(game.board.hand),
    turn: game.board.turn,
    key: Uint8Array.from(internal.key),
    repetitionCount: game.repetitionCount(),
    undoDepth: internal.undoStack.length,
  };
}

// Post-draft (drafting rights closed on both sides) and the survey actually
// ran on it (search.ts's `skipped_reason` is only ever set for a drafting
// position -- draftIsOpen's own gate).
function postDraftRecords(fixture: SearchFixture, count: number): SearchRecord[] {
  const out: SearchRecord[] = [];
  for (const rec of fixture.searches) {
    if (rec.result.tactical !== null && rec.result.tactical.skipped_reason === null) out.push(rec);
    if (out.length >= count) break;
  }
  return out;
}

test('refutation survey: make/undo matches a clone-based reference on 3+ post-draft positions', () => {
  const fixture = loadFixture();
  const records = postDraftRecords(fixture, 3);
  assert.ok(records.length >= 3, `expected >= 3 post-draft survey positions, found ${records.length}`);

  for (const rec of records) {
    const stateFast = rootStateFor(rec);
    const legalFast = legalActionsOf(stateFast);
    const rootFast = rootNodeFor(stateFast, legalFast);
    const before = snapshotGame(stateFast.game);

    const fastSurvey = new TacticalSurvey(DEFAULT_TRANSITION_CAP);
    const fastWitness = fastSurvey.rootWin(stateFast);
    const fastAllowed = fastSurvey.refutationMask(stateFast, rootFast);
    const fastRecord = fastSurvey.record({ rootWitness: fastWitness });

    const after = snapshotGame(stateFast.game);
    assert.deepStrictEqual(after, before, `${rec.position.fen}: root game mutated by the survey`);

    const stateRef = rootStateFor(rec);
    const legalRef = legalActionsOf(stateRef);
    const rootRef = rootNodeFor(stateRef, legalRef);
    const ref = referenceSurvey(stateRef, rootRef, DEFAULT_TRANSITION_CAP);

    assert.equal(fastWitness, ref.rootWitness, `${rec.position.fen}: root witness mismatch`);
    assert.deepStrictEqual(fastAllowed, ref.allowed, `${rec.position.fen}: allowed mask mismatch`);
    assert.deepStrictEqual(fastRecord.refuted, ref.refuted, `${rec.position.fen}: refuted list mismatch`);
    assert.equal(fastRecord.providerCalls, ref.calls, `${rec.position.fen}: provider call count mismatch`);
    assert.equal(
      fastRecord.candidateTransitions,
      ref.transitions,
      `${rec.position.fen}: candidate transition count mismatch`,
    );
    assert.equal(fastRecord.truncated, ref.truncated, `${rec.position.fen}: truncated flag mismatch`);
  }
});
