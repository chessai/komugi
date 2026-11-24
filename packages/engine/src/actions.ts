// Action id -> move-record decode. Port of
// action_encoding.decode_cython_action (src/komugi/engine/action_encoding.py:397-469):
// scans generate_all_moves for the first record whose id matches.
//
// Hot-path note (section 3, "no per-move objects on the hot path"; U6
// throughput gate): board ids are always < B_START and arata ids always
// >= B_START (movegen_board.ts/movegen_arata.ts never emit outside their own
// block), so decode only ever needs to run the one relevant emitter, and can
// stop at the first id match instead of materializing every record via
// generateRecords (which the root/SAN/decode callers still use directly).
import { Board, cellIndex, typeIndex } from './board.ts';
import { B_START } from './constants.ts';
import type { MoveRecord } from './movegen.ts';
import { forEachArataMove } from './movegen_arata.ts';
import { forEachBoardMove } from './movegen_board.ts';

// Tier-mask -> (type_index, tier) list, reading the board at the destination
// square BEFORE the move is applied (the captured/converted pieces are still
// present). Shared by apply.ts and san.ts; mirrors the ascending-tier `t in
// range(target_h)` loops cymovegen uses to build a record's `captured` list
// (cymovegen.pyx:657-668 for capture; betrayal/arata conversions never see
// more than 2 candidate tiers, so that ascending order is also combo order,
// cymovegen.pyx:630-651, 1042-1051).
export function capturedFromMask(
  board: Board,
  r: number,
  f: number,
  mask: number,
): Array<[number, number]> {
  const out: Array<[number, number]> = [];
  for (let t = 0; t < 3; t++) {
    if (mask & (1 << t)) {
      out.push([typeIndex(board.cells[cellIndex(r, f, t)]!), t + 1]);
    }
  }
  return out;
}

const DECODE_STOP = Symbol('decode-stop');

export function decodeAction(board: Board, id: number): MoveRecord {
  let found: MoveRecord | undefined;
  const cb = (candidateId: number, struct: Int32Array): void => {
    if (candidateId === id) {
      found = { id: candidateId, struct: struct.slice() };
      throw DECODE_STOP;
    }
  };
  try {
    if (id < B_START) forEachBoardMove(board, cb);
    else forEachArataMove(board, cb);
  } catch (e) {
    if (e !== DECODE_STOP) throw e;
  }
  if (!found) throw new Error(`No legal move for action ${id}`);
  return found;
}
