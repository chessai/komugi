// Structured move descriptors: host-neutral move I/O (KOM-1198 unit 1,
// docs/design/kom1198-move-io.md section 3). No SAN/kanji parsing required.
// Defined here, not protocol.ts (N1 in the design's section 9): root.ts needs
// this type and sits in the rules-only closure test/import_graph.test.ts
// pins; protocol.ts's own imports (backend.ts, rule_features.ts,
// search/search.ts) must never join that closure. protocol.ts re-exports
// MoveHistoryEntry from here instead of defining it.

import { Board, cellIndex, colorOf, typeIndex } from './board.ts';
import { decodeAction } from './actions.ts';
import { MT_ARATA, MT_BETRAY, MT_CAPTURE, MT_ROUTE, MT_TSUKE } from './constants.ts';
import { generateRecords } from './movegen.ts';

// Section 2: 1-based rank/file/tier, exactly SAN's own numerals.
export interface Square {
  rank: number;
  file: number;
  tier: number;
}

export interface CapturedPiece {
  piece: number; // constants.ts type index, 0-13
  color: number; // 0-1; always the opponent of the mover (movegen_board.ts:525,562)
  tier: number; // 1-based tier of the destination square this piece occupied before the move
}

export type MoveKind = 'move' | 'drop'; // drop = from hand (MT_ARATA); move = already on the board

// What happens at `to`. 'move' = MT_ROUTE onto an empty square; 'stack' =
// MT_TSUKE onto a tower; 'capture' = MT_CAPTURE (captured pieces are
// REMOVED, apply.ts:91-94, SAN 取); 'betray' = MT_BETRAY, or an MT_ARATA
// drop with a non-zero conversion mask (pieces are FLIPPED, apply.ts:95-106,
// SAN 返). A drop is 'betray' iff captures.length > 0, else 'move' on an
// empty square or 'stack' onto a tower.
export type MoveEffect = 'move' | 'stack' | 'capture' | 'betray';

export interface MoveDescriptor {
  kind: MoveKind;
  effect: MoveEffect;
  piece: number; // 0-13
  color: number; // 0-1, the moving side
  from?: Square; // present iff kind === 'move'; the key is OMITTED (never `from: undefined`) for 'drop'
  to: Square;
  captures: CapturedPiece[]; // removed (capture) or flipped (betray) pieces; [] otherwise
  draftDone: boolean; // true iff this move completes the mover's draft (a black draftDone also ends white's, apply.ts:107-113)
}

export type MoveHistoryEntry = string | MoveDescriptor;

// Sibling of actions.ts's capturedFromMask (section 3: that function's
// [typeIndex, tier] tuple shape is pinned by san.ts's destructuring and must
// not change), adding the captured piece's color.
function capturedPiecesFromMask(board: Board, r: number, f: number, mask: number): CapturedPiece[] {
  const out: CapturedPiece[] = [];
  for (let t = 0; t < 3; t++) {
    if (mask & (1 << t)) {
      const code = board.cells[cellIndex(r, f, t)]!;
      out.push({ piece: typeIndex(code), color: colorOf(code), tier: t + 1 });
    }
  }
  return out;
}

// The one pure struct-to-descriptor mapping (section 3): reads only `struct`
// and the pre-move `board` cells at the destination, never enumerates moves.
export function describeRecord(board: Board, struct: Int32Array): MoveDescriptor {
  const moveType = struct[0]!;
  const piece = struct[1]!;
  const color = struct[2]!;
  const toR = struct[6]!;
  const toF = struct[7]!;
  const toTier = struct[8]!;
  const draftDone = struct[9] !== 0;
  const captures = capturedPiecesFromMask(board, toR, toF, struct[10]!);
  const to: Square = { rank: toR + 1, file: toF + 1, tier: toTier };

  let effect: MoveEffect;
  if (moveType === MT_ROUTE) effect = 'move';
  else if (moveType === MT_TSUKE) effect = 'stack';
  else if (moveType === MT_CAPTURE) effect = 'capture';
  else if (moveType === MT_BETRAY) effect = 'betray';
  else effect = captures.length > 0 ? 'betray' : toTier === 1 ? 'move' : 'stack'; // MT_ARATA

  if (moveType === MT_ARATA) {
    return { kind: 'drop', effect, piece, color, to, captures, draftDone };
  }
  const from: Square = { rank: struct[3]! + 1, file: struct[4]! + 1, tier: struct[5]! };
  return { kind: 'move', effect, piece, color, from, to, captures, draftDone };
}

// Convenience for single moves (e.g. a reply descriptor); never called once
// per candidate (that would be O(n^2), decodeAction re-enumerates moves).
export function describeMove(board: Board, id: number): MoveDescriptor {
  return describeRecord(board, decodeAction(board, id).struct);
}

function isSquareShaped(x: unknown): x is Square {
  if (typeof x !== 'object' || x === null) return false;
  const s = x as Record<string, unknown>;
  return Number.isInteger(s.rank) && Number.isInteger(s.file) && Number.isInteger(s.tier);
}

function isCapturedPieceShaped(x: unknown): x is CapturedPiece {
  if (typeof x !== 'object' || x === null) return false;
  const c = x as Record<string, unknown>;
  return Number.isInteger(c.piece) && Number.isInteger(c.color) && Number.isInteger(c.tier);
}

// A bad shape is never a TypeError from a property read (section 3): every
// field is checked before resolveMoveDescriptor relies on it.
function isDescriptorShaped(x: unknown): x is MoveDescriptor {
  if (typeof x !== 'object' || x === null || Array.isArray(x)) return false;
  const d = x as Record<string, unknown>;
  if (d.kind !== 'move' && d.kind !== 'drop') return false;
  if (d.effect !== 'move' && d.effect !== 'stack' && d.effect !== 'capture' && d.effect !== 'betray') return false;
  if (!Number.isInteger(d.piece)) return false;
  if (!Number.isInteger(d.color)) return false;
  if (!isSquareShaped(d.to)) return false;
  if (!Array.isArray(d.captures) || !d.captures.every(isCapturedPieceShaped)) return false;
  if (typeof d.draftDone !== 'boolean') return false;
  if (d.kind === 'move' && !isSquareShaped(d.from)) return false;
  return true;
}

function squaresEqual(a: Square, b: Square): boolean {
  return a.rank === b.rank && a.file === b.file && a.tier === b.tier;
}

function capturesEqual(a: readonly CapturedPiece[], b: readonly CapturedPiece[]): boolean {
  if (a.length !== b.length) return false;
  const remaining = b.slice();
  for (const item of a) {
    const idx = remaining.findIndex((c) => c.piece === item.piece && c.color === item.color && c.tier === item.tier);
    if (idx === -1) return false;
    remaining.splice(idx, 1);
  }
  return true;
}

function descriptorsEqual(a: MoveDescriptor, b: MoveDescriptor): boolean {
  if (a.kind !== b.kind || a.effect !== b.effect || a.piece !== b.piece || a.color !== b.color) return false;
  if (a.draftDone !== b.draftDone) return false;
  if (!squaresEqual(a.to, b.to)) return false;
  if ((a.from === undefined) !== (b.from === undefined)) return false;
  if (a.from !== undefined && b.from !== undefined && !squaresEqual(a.from, b.from)) return false;
  return capturesEqual(a.captures, b.captures);
}

// Inverse of resolveSan (san.ts:74-80) for descriptors: one generateRecords
// pass at `board` (sanRecords' own order), describeRecord on each record,
// structural comparison, first match's id. O(n) per ply, like resolveSan.
export function resolveMoveDescriptor(board: Board, descriptor: MoveDescriptor): number {
  if (!isDescriptorShaped(descriptor)) throw new Error('Illegal move descriptor');
  for (const { id, struct } of generateRecords(board)) {
    if (descriptorsEqual(describeRecord(board, struct), descriptor)) return id;
  }
  throw new Error('Illegal move descriptor');
}
