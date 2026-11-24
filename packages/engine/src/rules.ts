// @komugi-ai/engine/rules -- rules-only entry (design doc section 3.3).
// Pure rules surface: FEN, move generation, apply, SAN, terminal state.
// No ORT, no search, no worker. AC7 (test/import_graph.test.ts) asserts the
// import graph reachable from this file is exactly the 19-module closure.

export { Board, colorOf, cellIndex, typeIndex, makeCode } from './board.ts';

export * from './constants.ts';

export {
  validateFen,
  parseFen,
  encodeFen,
  canonicalFen,
  boardFromFen,
  boardToFen,
} from './fen.ts';

export { fillLegalMask, generateRecords } from './movegen.ts';
export { forEachBoardMove, fillBoardMask } from './movegen_board.ts';
export { forEachArataMove, fillArataMask } from './movegen_arata.ts';

export { buildFilterContext, isMoveLegalUnderRule1 } from './ligungi_filter.ts';
export type { FilterContext } from './ligungi_filter.ts';

export { applyMove } from './apply.ts';
export { decodeAction, capturedFromMask } from './actions.ts';
export { buildSan, resolveSan } from './san.ts';
export { Game, MaxTurns } from './game.ts';

export {
  inCheck,
  isCheckmate,
  isStalemate,
  isInsufficientMaterial,
  isRepetitionDraw,
  isTerminalNonmate,
  isGameOver,
} from './terminal.ts';

export {
  RULES_PROFILES,
  DEFAULT_RULES_PROFILE,
  parseRulesProfile,
  RepetitionRule,
  canonicalMove,
} from './repetition.ts';
export type { RulesProfile } from './repetition.ts';

export { RULES_PRESETS, toRulesConfig, rulesConfigKey } from './rules_config.ts';
export type {
  RulesConfig,
  TacticianBetrayalMode,
  StalemateResult,
  RepetitionKeyMode,
  PlyCap,
} from './rules_config.ts';

export { archerElbowBlocked, fullBetrayalCombo } from './rules_ligungi.ts';

export { rootGame, HISTORY_PRESENT } from './root.ts';
export type { HistoryStamp } from './root.ts';

export {
  describeRecord,
  describeMove,
  resolveMoveDescriptor,
} from './move_io.ts';
export type {
  Square,
  CapturedPiece,
  MoveKind,
  MoveEffect,
  MoveDescriptor,
  MoveHistoryEntry,
} from './move_io.ts';

export { validatePosition, buildPosition, parsePosition } from './position_builder.ts';
export type { PlainPiece, PlainCell, PlainHandEntry, PlainPosition } from './position_builder.ts';
