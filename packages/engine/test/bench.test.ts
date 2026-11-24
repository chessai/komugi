// U6: undo gate and transition benches (docs/design/clientside-engine-port.md
// unit U6, M9). Gates: undo restores every byte and count after every ply of
// the fixture games; a mutant whose undo skips the key-count decrement is
// reported by the same checker; the replay rate is >= 20000 transitions/s.
// The survey micro-bench (5 play positions, cap 20,000) is printed, not gated;
// the 200k/s target is an advisor measurement (section 3, R3).

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { Game } from '../src/game.ts';
import { RepetitionCounts, RepetitionRule } from '../src/repetition.ts';
import { benchReplay, captureState, checkUndo, gamesFromRecords, newGame, splitReplay } from '../bench/replay.ts';
import type { BenchGame } from '../bench/replay.ts';
import { DEFAULT_TRANSITION_CAP, playPositions, survey } from '../bench/survey.ts';
import { loadFixture } from './fixture.ts';

const REPLAY_FLOOR = 20000;

// The M9 mutant: undo() restores everything except that the undone
// position's key count is never decremented. Capture the about-to-be-undone
// position's (hash, bytes, count) before calling the real undo(), then
// write that count straight back in after -- undoing undo()'s own
// decrement, independent of whatever `RepetitionCounts` does internally.
class SkipDecrementGame extends Game {
  undo(): void {
    const self = this as unknown as { key: Uint8Array; keyHi: number; keyLo: number; counts: RepetitionCounts };
    const key = self.key;
    const hi = self.keyHi;
    const lo = self.keyLo;
    const before = self.counts.count(hi, lo, key);
    super.undo();
    if (before > 0) self.counts.setCount(key, before);
  }
}

let cached: BenchGame[] | null = null;
function benchGames(): BenchGame[] {
  if (cached === null) cached = gamesFromRecords(loadFixture());
  return cached;
}

function totalPlies(games: BenchGame[]): number {
  return games.reduce((n, g) => n + g.actions.length, 0);
}

test('undo restores every byte and count after every ply, under both rules', () => {
  const games = benchGames();
  const results: string[] = [];
  let failures = 0;
  let first = '';
  for (const rule of [RepetitionRule.PLACEMENT, RepetitionRule.FULL]) {
    const r = checkUndo(() => newGame(rule), games);
    results.push(`${rule}: plies=${r.plies} mismatches=${r.mismatches} first=${r.first || '-'}`);
    failures += r.mismatches;
    if (!first && r.first) first = `${rule}#${r.first}`;
    assert.equal(r.plies, totalPlies(games));
  }
  console.log(`undo gate: games=${games.length} ${results.join(' | ')}`);
  assert.ok(games.length > 0 && totalPlies(games) > 0, 'no fixture plies');
  assert.equal(failures, 0, `first undo mismatch at ${first}`);
});

test('mutant: an undo that skips the key-count decrement is reported by the checker', () => {
  const games = benchGames();
  const mutant = checkUndo(() => new SkipDecrementGame(newGame(RepetitionRule.PLACEMENT).board, RepetitionRule.PLACEMENT), games);
  console.log(`undo mutant: plies=${mutant.plies} mismatches=${mutant.mismatches} first=${mutant.first || '-'}`);
  assert.ok(mutant.mismatches > 0, 'the undo checker did not detect the skip-decrement mutant');
  // The very first ply already exposes it: its undone key keeps a count of 1.
  assert.equal(mutant.first, `${games[0].name}#undo1`);
});

test(`fixture replay rate >= ${REPLAY_FLOOR} transitions/s`, () => {
  const games = benchGames();
  const r = benchReplay(games, RepetitionRule.PLACEMENT);
  const s = splitReplay(games, RepetitionRule.PLACEMENT);
  console.log(
    `replay split: play=${((s.playMs * 1000) / s.transitions).toFixed(1)}us/transition ` +
      `isTerminal=${((s.terminalMs * 1000) / s.transitions).toFixed(1)}us/transition`,
  );
  console.log(
    `replay bench (PLACEMENT): games=${games.length} plies/round=${totalPlies(games)} rounds=${r.rounds} ` +
      `transitions=${r.transitions} ms=${r.ms.toFixed(1)} rate=${Math.round(r.rate)}/s floor=${REPLAY_FLOOR}/s`,
  );
  assert.equal(r.transitions, r.rounds * totalPlies(games));
  assert.ok(r.rate >= REPLAY_FLOOR, `replay rate ${Math.round(r.rate)}/s is below the ${REPLAY_FLOOR}/s floor`);
});

test(`survey micro-bench: 5 play positions, cap ${DEFAULT_TRANSITION_CAP}`, () => {
  const positions = playPositions(benchGames(), 5);
  const lines: string[] = [];
  let totalTransitions = 0;
  let totalMs = 0;
  let restoreMismatches = 0;
  for (const { label, game } of positions) {
    const before = captureState(game);
    const r = survey(game);
    if (captureState(game) !== before) restoreMismatches++;
    assert.ok(r.surveyTransitions <= DEFAULT_TRANSITION_CAP, `${label}: survey exceeded its cap`);
    assert.ok(r.decisiveTransitions <= r.width, `${label}: decisive pass exceeded |legal|`);
    totalTransitions += r.decisiveTransitions + r.surveyTransitions;
    totalMs += r.ms;
    lines.push(
      `  ${label}: width=${r.width} rootWin=${r.rootWin ?? '-'} decisive=${r.decisiveTransitions} ` +
        `survey=${r.surveyTransitions} truncated=${r.truncated} refuted=${r.refuted} ` +
        `ms=${r.ms.toFixed(1)} rate=${Math.round(r.rate)}/s`,
    );
  }
  console.log(
    `survey bench: positions=${positions.length} restoreMismatches=${restoreMismatches} ` +
      `transitions=${totalTransitions} ms=${totalMs.toFixed(1)} ` +
      `rate=${Math.round((totalTransitions * 1000) / totalMs)}/s`,
  );
  console.log(lines.join('\n'));
  assert.equal(positions.length, 5, 'fewer than 5 play-phase positions in the fixture games');
  assert.equal(restoreMismatches, 0, 'survey did not leave its position exactly as it found it');
});
