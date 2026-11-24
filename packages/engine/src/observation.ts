// The 128-plane observation encoder. Faithful port of
// src/komugi/engine/observation.py's fast path, `observation_from_cyboard`
// (observation.py:257-355), fed by a Game's own history-derived counters
// exactly as `Game.observation` does (src/komugi/engine/game.py:710-717):
// repetition_count(), turns, last_move and progress_plies are read off the
// game, never re-derived from a FEN or a history scan (docs/design/
// clientside-engine-port.md section 2.4).
//
// Plane layout (observation.py:6-33):
//   0-83    occupancy: plane = type_idx*6 + (tier-1)*2 + color
//   84-111  hand: plane = 84 + type_idx*2 + color, constant = count/MAX_COUNT
//   112/113 draft_w / draft_b
//   114     side to move (black = 1)
//   115     turn clock: min(turns, MAX_TURNS) / MAX_TURNS
//   116/117 repetition >= 2 / >= 3
//   118/119 last move origin tier/3 (skipped for arata) / destination tier/3
//   120     progress clock: min(progress_plies, MAX_TURNS) / MAX_TURNS
//   121-127 reserved, always 0 at planeMapVersion 3 (the default); at 4,
//           121-123 are the rule-feature planes (rule_features.ts, KOM-1162)

import { Board, cellIndex, colorOf, typeIndex } from './board.ts';
import {
  ARCHER, FORTRESS, LANCER, MAJOR_GENERAL, NUM_PIECE_TYPES, RANKS, FILES, TIERS,
  NUM_SQUARES, RIDER, SOLDIER, SPY, WARRIOR,
} from './constants.ts';
import { Game, MaxTurns } from './game.ts';
import type { LastMove } from './game.ts';
import { fillRulePlanes, parsePlaneMapVersion } from './rule_features.ts';
import type { PlaneMapVersion } from './rule_features.ts';

export const OBS_PLANES = 128;

const _OCCUPANCY_PLANES = 84;
const _PLANE_LAST_MOVE_FROM = 118;
const _PLANE_LAST_MOVE_TO = 119;
const _PLANE_PROGRESS = 120;
const _MAX_TIER = TIERS; // 3

// MAX_COUNT indexed by type index (observation.py:98-113): the advanced-init
// hand count for each piece type. Everything not listed below is 1, matching
// the Python dict's entries for MARSHAL/GENERAL/LIEUTENANT_GENERAL/CANNON/
// MUSKETEER/TACTICIAN.
const MAX_COUNT_BY_IDX: number[] = new Array(NUM_PIECE_TYPES).fill(1);
MAX_COUNT_BY_IDX[MAJOR_GENERAL] = 2;
MAX_COUNT_BY_IDX[WARRIOR] = 2;
MAX_COUNT_BY_IDX[LANCER] = 3;
MAX_COUNT_BY_IDX[RIDER] = 2;
MAX_COUNT_BY_IDX[SPY] = 2;
MAX_COUNT_BY_IDX[FORTRESS] = 2;
MAX_COUNT_BY_IDX[SOLDIER] = 4;
MAX_COUNT_BY_IDX[ARCHER] = 2;

export function observationSize(): [number, number, number] {
  return [OBS_PLANES, RANKS, FILES];
}

function fillPlane(out: Float32Array, plane: number, value: number): void {
  const base = plane * NUM_SQUARES;
  for (let i = 0; i < NUM_SQUARES; i++) out[base + i] = value;
}

// Port of observation_from_cyboard (observation.py:257-355). `out` is the
// caller's reused `Float32Array(128*81)` (section 2.4); allocated here only
// when the caller does not supply one. Every history-dependent quantity is a
// parameter with the same "bare position" default the Python function uses.
export function observationFromBoard(
  board: Board,
  repetitionCount = 1,
  plies = 0,
  lastMove: LastMove = null,
  progressPlies = 0,
  out: Float32Array = new Float32Array(OBS_PLANES * NUM_SQUARES),
): Float32Array {
  out.fill(0);

  // --- Occupancy planes (0..83) -------------------------------------------
  for (let r = 0; r < RANKS; r++) {
    for (let f = 0; f < FILES; f++) {
      const j = FILES - 1 - f; // observation-space file index (board.ts:cellIndex orientation)
      for (let t = 0; t < TIERS; t++) {
        const code = board.cells[cellIndex(r, f, t)];
        if (code === 0) continue;
        const ti = typeIndex(code!);
        const ci = colorOf(code!);
        const tier = t + 1;
        const plane = ti * 6 + (tier - 1) * 2 + ci;
        out[plane * NUM_SQUARES + r * FILES + j] = 1.0;
      }
    }
  }

  // --- Hand planes (84..111) ----------------------------------------------
  for (let ti = 0; ti < NUM_PIECE_TYPES; ti++) {
    for (let ci = 0; ci < 2; ci++) {
      const cnt = board.hand[ti * 2 + ci]!;
      if (cnt <= 0) continue;
      const mc = MAX_COUNT_BY_IDX[ti]!;
      const val = cnt / mc;
      fillPlane(out, _OCCUPANCY_PLANES + ti * 2 + ci, val);
    }
  }

  // --- Drafting-rights planes (112, 113) ----------------------------------
  if (board.draftW) fillPlane(out, 112, 1.0);
  if (board.draftB) fillPlane(out, 113, 1.0);

  // --- Side-to-move plane (114) --------------------------------------------
  if (board.turn === 1) fillPlane(out, 114, 1.0);

  // --- Turn-clock plane (115) ----------------------------------------------
  fillPlane(out, 115, Math.min(plies, MaxTurns) / MaxTurns);

  // --- Repetition threshold planes (116, 117) ------------------------------
  fillPlane(out, 116, repetitionCount >= 2 ? 1.0 : 0.0);
  fillPlane(out, 117, repetitionCount >= 3 ? 1.0 : 0.0);

  // --- Last-move planes (118, 119) -----------------------------------------
  // `fromSq`/`toSq` already use the action encoding's `s = r*9 + (8-f)`
  // numbering, which is also the row-major flat index within a plane under
  // the `j = 8-f` convention above (observation.py:340-347), so no square
  // translation is needed here.
  if (lastMove !== null) {
    if (lastMove.fromSq >= 0) {
      out[_PLANE_LAST_MOVE_FROM * NUM_SQUARES + lastMove.fromSq] = lastMove.fromTier / _MAX_TIER;
    }
    out[_PLANE_LAST_MOVE_TO * NUM_SQUARES + lastMove.toSq] = lastMove.toTier / _MAX_TIER;
  }

  // --- Progress-clock plane (120) ------------------------------------------
  fillPlane(out, _PLANE_PROGRESS, Math.min(progressPlies, MaxTurns) / MaxTurns);

  // --- Reserved planes (121..127) -------------------------------------------
  // Left at their allocation value, 0.0.

  return out;
}

export interface ObservationOptions {
  out?: Float32Array;
  // KOM-1162: 3 (default) leaves 121-127 at 0; 4 fills the rule-feature
  // planes 121-123 from the board's mode. Any other value throws.
  planeMapVersion?: PlaneMapVersion | undefined;
}

// Port of Game.observation (game.py:710-717): reads the game's own counters
// -- repetitionCount(), turns, lastMove, progressPlies -- and never
// re-derives any of them.
export function observation(game: Game, opts: ObservationOptions = {}): Float32Array {
  const planeMapVersion = parsePlaneMapVersion(opts.planeMapVersion);
  const out = observationFromBoard(
    game.board,
    game.repetitionCount(),
    game.turns,
    game.lastMove,
    game.progressPlies,
    opts.out,
  );
  fillRulePlanes(out, game.board.mode, planeMapVersion);
  return out;
}
