// Move dispatcher: concatenates the board (U2a) and arata (U2b) emitters in
// generate_all_moves' / fill_legal_mask's order (src/komugi/engine/cymovegen.pyx:1080-1141).
// Both forEachBoardMove and fillBoardMask already no-op during the draft
// (movegen_board.ts), so the dispatcher does not need to branch on
// draftW/draftB itself: calling both emitters unconditionally reproduces the
// Cython branch (arata-only during the draft, board-then-arata otherwise).

import { Board } from './board.ts';
import { fillArataMask, forEachArataMove } from './movegen_arata.ts';
import { fillBoardMask, forEachBoardMove } from './movegen_board.ts';
import { toRulesConfig } from './rules_config.ts';
import type { RulesConfig, RulesProfile } from './rules_config.ts';

export function fillLegalMask(board: Board, mask: Uint8Array, rules: RulesProfile | RulesConfig = 'komugi'): void {
  const config = toRulesConfig(rules);
  fillBoardMask(board, mask, config);
  fillArataMask(board, mask, config);
}

export type MoveRecord = { id: number; struct: Int32Array };

// One object per move (root / SAN / decode use only, per section 3: the hot
// path stays on the reused-struct callbacks above).
export function generateRecords(board: Board, rules: RulesProfile | RulesConfig = 'komugi'): MoveRecord[] {
  const config = toRulesConfig(rules);
  const out: MoveRecord[] = [];
  forEachBoardMove(board, (id, struct) => out.push({ id, struct: struct.slice() }), config);
  forEachArataMove(board, (id, struct) => out.push({ id, struct: struct.slice() }), config);
  return out;
}
