// SAN build and resolve.
// Out: a port of cymovegen._build_san (src/komugi/engine/cymovegen.pyx:440-496),
// the string the Python SAN oracle (Gungi, via the match server's move_for_action)
// also produces for a legal move (F-B in the design doc: with one engine,
// this spelling must equal the fixture's `legal_san`).
// In: a port of the matching loop in Gungi.move / the match server's own
// game_from_history (src/komugi/engine/api.py:142-187): strip trailing
// `=`/`#`, take the first enumerated move whose SAN matches. Gungi's
// enumeration mirrors cymovegen's (board squares in SQUARES order, then the
// hand), so movegen.ts's generation order already matches the oracle's
// first-match order.

import { Board } from './board.ts';
import { capturedFromMask } from './actions.ts';
import { MARSHAL, MT_ARATA, MT_BETRAY, MT_CAPTURE, MT_TSUKE, PIECE_KANJI } from './constants.ts';
import { generateRecords } from './movegen.ts';

const ARATA_MARK = '新'; // 新
const CAPTURE_MARK = '取'; // 取
const TSUKE_MARK = '付'; // 付
const BETRAY_MARK = '返'; // 返
const DRAFT_DONE_MARK = '終'; // 終

// Port of cymovegen._build_san. `struct` is the Int32Array(16) move record
// (movegen_board.ts / movegen_arata.ts layout); `board` must still be in the
// pre-move state the record was generated from (captured pieces are read off
// the destination square).
export function buildSan(board: Board, struct: Int32Array): string {
  const moveType = struct[0];
  const pieceTi = struct[1]!;
  const fromR = struct[3]!;
  const fromF = struct[4]!;
  const fromTier = struct[5]!;
  const toR = struct[6]!;
  const toF = struct[7]!;
  const toTier = struct[8]!;
  const draftFinished = struct[9] !== 0;
  const captured = capturedFromMask(board, toR, toF, struct[10]!);

  const kanji = PIECE_KANJI[pieceTi]!;
  const fromStr = fromTier !== 0 ? `(${fromR + 1}-${fromF + 1}-${fromTier})` : '';

  const arataMark = moveType === MT_ARATA ? ARATA_MARK : '';
  const captureMark = moveType === MT_CAPTURE ? CAPTURE_MARK : '';
  const tsukeMark =
    moveType === MT_TSUKE || (toTier !== 1 && toTier - fromTier > 0) ? TSUKE_MARK : '';

  let betrayMark = '';
  if (moveType === MT_BETRAY || (moveType === MT_ARATA && captured.length > 0)) {
    betrayMark = BETRAY_MARK;
    for (const [ti] of captured) betrayMark += PIECE_KANJI[ti];
  }

  const draftDone = draftFinished ? DRAFT_DONE_MARK : '';
  const sanSuffix = betrayMark || tsukeMark;

  let san = `${arataMark}${kanji}${fromStr}${captureMark}(${toR + 1}-${toF + 1}-${toTier})${sanSuffix}${draftDone}`;
  if (captured.some(([ti]) => ti === MARSHAL)) san += '#';
  return san;
}

export type SanRecord = { id: number; san: string; struct: Int32Array };

// Every legal move at `board`, in generation (= Gungi enumeration) order,
// with its SAN. Used both to resolve an input SAN and, in tests, to check
// for duplicate SANs / ids at a position.
export function sanRecords(board: Board): SanRecord[] {
  return generateRecords(board).map(({ id, struct }) => ({ id, san: buildSan(board, struct), struct }));
}

// Port of the string-match branch of Gungi.move (api.py:150-156): strip
// trailing `=`/`#`, return the id of the first move whose SAN equals the
// stripped string or the original.
export function resolveSan(board: Board, san: string): number {
  const normalized = san.replace(/[=#]+$/, '');
  for (const record of sanRecords(board)) {
    if (record.san === normalized || record.san === san) return record.id;
  }
  throw new Error(`Illegal move: ${san}`);
}
