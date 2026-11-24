// KOM-1106 (SPLIT 2026-11-14): ligungi-only post-filters for move
// generation, kept out of movegen_board.ts/terminal.ts's hot loops so the
// komugi/gungijs code paths stay untouched -- every call site below is
// reached only behind an `if (rules === 'ligungi')` branch at the call site.
// CLEAN ROOM: the rule text comes from the ticket's prose only, never from
// ligungi source.
//
// Rule (2), archer tier-2/3 blocking. The archer's forward diagonal jump
// (movegen_board.ts/terminal.ts's own ARCHER + |dy|===1 + |dx|===1 guard)
// lands on a staircase of squares as the archer's own tier grows: step 0 is
// (lateral 1, forward 2), step 1 is (lateral 2, forward 3), step 2 is
// (lateral 3, forward 4) -- one more step reachable per tier the archer
// itself occupies (length === originTier). Under ligungi, reaching step
// `step` (>= 1) also needs the "elbow" square at (lateral = step+1,
// forward = step+1) -- the true diagonal corner, off the staircase's own
// path -- no taller than the archer: the same blocking test the engine
// already applies to every other path square. The elbow at a smaller k was
// already enforced when step k-1 was processed (the forward-fill loop stops
// extending the moment a step is blocked), so only the newest elbow (k =
// step+1) needs checking at each step.
//
// Rule (3), all-or-nothing tactician betrayal. getBetrayalCombos/
// qualifyingTierMask (movegen_board.ts) enumerate every non-empty subset of
// the enemies a tactician's tsuke may convert; under ligungi a betrayal
// survives only if it converts EVERY enemy in the target tower, and only if
// EVERY enemy type in that tower qualifies under the existing hand test --
// fullBetrayalCombo/qualifyingTierMask's own longest-subset therefore has to
// be compared against the tower's total enemy count (not just the qualifying
// count), since betrayalOptions/qualifying only ever hold the qualifying
// subset to begin with. If any enemy type fails the hand test, no combo
// survives and only the plain stack (sel 0) is offered.

import type { Board } from './board.ts';

export function archerElbowBlocked(
  board: Board,
  py: number,
  px: number,
  dy: number,
  dx: number,
  step: number,
  originTier: number,
): boolean {
  const k = step + 1;
  const rank = py + k * dy;
  const file = px + k * dx;
  if (rank < 1 || rank > 9 || file < 1 || file > 9) return false;
  if (board.top(rank - 1, file - 1) === 0) return false;
  return board.height(rank - 1, file - 1) > originTier;
}

// Of getBetrayalCombos' subsets (all drawn from the qualifying enemies
// only), keeps the unique longest subset -- but only if it covers every
// enemy in the target tower (`enemyCount`, the tower's total enemy count,
// qualifying or not). If even one enemy type in the tower fails the hand
// test, betrayalOptions/qualifying can never reach enemyCount, so no combo
// survives: [] (the plain stack, sel 0, stays the only option).
export function fullBetrayalCombo<T>(combos: readonly T[][], enemyCount: number): T[][] {
  if (combos.length === 0) return [];
  let full = combos[0]!;
  for (const c of combos) {
    if (c.length > full.length) full = c;
  }
  return full.length === enemyCount ? [full] : [];
}
