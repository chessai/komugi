// U6: survey micro-bench (docs/design/clientside-engine-port.md section 3).
// The workload of the match search's tactical pass, on one mutable Game with
// make/unmake instead of Python's clone per transition:
//   - root_win (src/komugi/search/match/search.py:306-333): every legal move
//     in ascending id order, stopping at the first win for the mover, uncapped;
//   - refutation_mask (src/komugi/search/match/search.py:335-394): for each
//     root move, every reply, stopping at the first reply that wins for the
//     opponent; this pass alone is capped (DEFAULT_TRANSITION_CAP = 20000,
//     search.py:496) and checks the cap before each transition.

import type { Game } from '../src/game.ts';
import { RepetitionRule } from '../src/repetition.ts';
import { isTerminal, newGame } from './replay.ts';
import type { BenchGame } from './replay.ts';

export const DEFAULT_TRANSITION_CAP = 20000;

// Ascending legal ids (search.py:461-469).
export function legalActions(game: Game): number[] {
  const mask = game.validMoves();
  const out: number[] = [];
  for (let i = 0; i < mask.length; i++) if (mask[i]) out.push(i);
  return out;
}

// Called only on a terminal position: win_state [white, black, draw].
function winner(game: Game): number | null {
  const w = game.winState();
  if (w[0]) return 0;
  if (w[1]) return 1;
  return null;
}

export type SurveyResult = {
  width: number;
  rootWin: number | null;
  decisiveTransitions: number;
  surveyTransitions: number;
  truncated: boolean;
  refuted: number;
  ms: number;
  rate: number;
};

// Leaves `game` exactly as it found it (every play is undone).
export function survey(game: Game, cap = DEFAULT_TRANSITION_CAP): SurveyResult {
  const t0 = performance.now();
  const mover = game.board.turn;
  const legal = legalActions(game);

  let decisive = 0;
  let rootWin: number | null = null;
  for (const action of legal) {
    game.play(action);
    decisive++;
    const won = isTerminal(game) && winner(game) === mover;
    game.undo();
    if (won) {
      rootWin = action;
      break;
    }
  }

  let spent = 0;
  let truncated = false;
  let refuted = 0;
  const opponent = 1 - mover;
  for (const action of legal) {
    if (spent >= cap) {
      truncated = true;
      break;
    }
    game.play(action);
    spent++;
    if (!isTerminal(game)) {
      for (const reply of legalActions(game)) {
        if (spent >= cap) {
          truncated = true;
          break;
        }
        game.play(reply);
        spent++;
        const refutes = isTerminal(game) && winner(game) === opponent;
        game.undo();
        if (refutes) {
          refuted++;
          break;
        }
      }
    }
    game.undo();
  }

  const ms = performance.now() - t0;
  const total = decisive + spent;
  return {
    width: legal.length,
    rootWin,
    decisiveTransitions: decisive,
    surveyTransitions: spent,
    truncated,
    refuted,
    ms,
    rate: (total * 1000) / ms,
  };
}

export type PlayPosition = { label: string; game: Game };

// `count` play-phase (draft over), non-terminal positions spread evenly over
// every ply of `games`, each rebuilt as a fresh Game replayed to that ply.
export function playPositions(
  games: BenchGame[],
  count: number,
  rule: RepetitionRule = RepetitionRule.PLACEMENT,
): PlayPosition[] {
  const candidates: { g: BenchGame; ply: number }[] = [];
  for (const g of games) {
    const game = newGame(rule);
    for (let ply = 0; ply <= g.actions.length; ply++) {
      if (!game.board.draftW && !game.board.draftB && !isTerminal(game)) candidates.push({ g, ply });
      if (ply < g.actions.length) game.play(g.actions[ply]);
    }
  }
  const picks: PlayPosition[] = [];
  const n = Math.min(count, candidates.length);
  for (let k = 0; k < n; k++) {
    const { g, ply } = candidates[Math.floor(((k + 0.5) * candidates.length) / n)];
    const game = newGame(rule);
    for (let i = 0; i < ply; i++) game.play(g.actions[i]);
    picks.push({ label: `${g.name}p${ply}`, game });
  }
  return picks;
}
