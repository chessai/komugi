// Root reconstruction (design doc section 2.5 "Root reconstruction", section
// 6 U8): the one place a request's repetition history enters the search.
// Port of the Python match server's root-game reconstruction and history
// replay, reduced to ONE engine and therefore ONE FEN check (design section
// 2.5: "The TS port keeps one
// FEN check" -- Python's second check compares the reference replay against
// the Cython engine's own board; this port has no second engine to diverge
// from).

import { decodeAction } from './actions.ts';
import type { Board } from './board.ts';
import { ADVANCED_POSITION } from './constants.ts';
import { boardFromFen, boardToFen, canonicalFen } from './fen.ts';
import { Game } from './game.ts';
import { resolveMoveDescriptor, type MoveDescriptor, type MoveHistoryEntry } from './move_io.ts';
import { canonicalMove, DEFAULT_RULES_PROFILE, legalSet, RepetitionRule, type RulesProfile } from './repetition.ts';
import { toRulesConfig } from './rules_config.ts';
import type { RulesConfig } from './rules_config.ts';
import { buildSan, resolveSan } from './san.ts';

export type HistoryStamp = 'absent' | 'present' | 'mismatch';

export const HISTORY_ABSENT: HistoryStamp = 'absent';
export const HISTORY_PRESENT: HistoryStamp = 'present';
export const HISTORY_MISMATCH: HistoryStamp = 'mismatch';

// design doc section 5 "Entry discrimination": a `string` entry is SAN; a
// non-null, non-array plain object is a descriptor (resolveMoveDescriptor);
// anything else keeps today's String(e) coercion, so it resolves, or fails,
// as SAN exactly as before this design.
function resolveHistoryEntry(board: Board, entry: unknown): number {
  if (typeof entry === 'string') return resolveSan(board, entry);
  if (entry !== null && typeof entry === 'object' && !Array.isArray(entry)) {
    return resolveMoveDescriptor(board, entry as MoveDescriptor);
  }
  return resolveSan(board, String(entry));
}

// Shared replay core (design section 5): resolves each entry against the
// position reached so far -- a string via resolveSan (unchanged), a
// descriptor via resolveMoveDescriptor -- under PLACEMENT (design F-A,
// section 2.1's REPETITION_RULE) from `start` (default the advanced start --
// KOM-1104, a TS-only extension over the Python reference, see
// docs/design/clientside-engine-port.md section 2.5). `boardFromFen(start)`
// is the validation: an invalid `start` throws here, same as an illegal
// entry, so `rootGame`'s catch sees it as just another reason the history
// didn't replay.
// KOM-1203 knob 7 (repetitionKey): the repetition key is FULL (board +
// hands + side to move + draft bits) when the resolved config's
// repetitionKey is 'full' (ligungi rule (5)), PLACEMENT otherwise --
// komugi/gungijs are unchanged. Both resolvers resolve purely structurally
// (san.ts's sanRecords and move_io.ts's resolveMoveDescriptor both ignore
// `rules`), so a history move that is pseudo-legal under the default-profile
// movegen but outside the resolved config's own legal set (a
// self-exposure/archer-elbow/all-or-nothing-betrayal divergence, KOM-1203
// F2(a)) would otherwise silently play and still land on the right FEN; this
// is checked against legalSet(board, config) before playing whenever any of
// those three knobs is off its komugi/gungijs default, so such a move throws
// here (caught by `rootGame`, which falls back to the FEN path with stamp
// 'mismatch') instead of being accepted -- for both entry types.
function replayEntries(
  history: readonly unknown[],
  start: string = ADVANCED_POSITION,
  rules: RulesProfile | RulesConfig = DEFAULT_RULES_PROFILE,
): { game: Game; sanHistory: string[] } {
  const config = toRulesConfig(rules);
  const rule = config.repetitionKey === 'full' ? RepetitionRule.FULL : RepetitionRule.PLACEMENT;
  const game = new Game(boardFromFen(start), rule, 1, rules);
  const sanHistory: string[] = [];
  const checkLegalSet =
    config.selfExposureFilter || config.archerElbowBlocking || config.tacticianBetrayalMode === 'all-or-nothing';
  for (const entry of history) {
    const action = resolveHistoryEntry(game.board, entry);
    const { struct } = decodeAction(game.board, action);
    if (checkLegalSet && !legalSet(game.board, config).has(canonicalMove(struct))) {
      throw new Error(`history move outside ligungi legal set: ${String(entry)}`);
    }
    sanHistory.push(buildSan(game.board, struct));
    game.play(action);
  }
  return { game, sanHistory };
}

// replayHistory (design section 5, NEW): gameFromHistory = replayHistory(...)
// .game. Also turns each resolved action id into its canonical SAN via
// buildSan and collects it into sanHistory. Because SAN is not injective
// (section 3), sanHistory is a display/key form, never a replay source for a
// history that contained descriptors.
export function replayHistory(
  history: readonly MoveHistoryEntry[],
  start?: string,
  rules?: RulesProfile | RulesConfig,
): { game: Game; sanHistory: string[] } {
  return replayEntries(history, start, rules);
}

// The Python match server's history replay: unchanged return type (Game)
// -- only its input widens from string[] to MoveHistoryEntry[] (design
// section 5). KOM-1203 N1: exported (root.ts:93) and called with a bare
// RulesProfile string by ligungi_terminal.test.ts/rules_profile.test.ts, so
// `rules` widens here too, never narrows to "takes the resolved RulesConfig".
export function gameFromHistory(
  history: readonly MoveHistoryEntry[],
  start?: string,
  rules?: RulesProfile | RulesConfig,
): Game {
  return replayHistory(history, start, rules).game;
}

// historyKey (design section 5, NEW): the ponder-continuity key engine.ts
// wires in. An all-string history (every existing caller) returns
// history.map(String), exactly today's key, with no canonicalization --
// resolveSan normalizes some spellings, and canonicalizing would change
// today's keys. A history with any non-string entry is replayed via
// replayHistory to its canonical sanHistory inside a try/catch; a replay
// failure (an unresolvable entry, including a malformed descriptor) returns
// null, never throws.
export function historyKey(
  history: readonly unknown[],
  start?: string,
  rules?: RulesProfile | RulesConfig,
): string[] | null {
  if (history.every((entry) => typeof entry === 'string')) {
    return history.map((entry) => String(entry));
  }
  try {
    return replayEntries(history, start, rules).sanHistory;
  } catch {
    return null;
  }
}

// The Python match server's root-game reconstruction: the game to search at `fen`, and
// the stamp for it. Three outcomes:
//   - no history at all -- the FEN path under FULL (design D1), turns = 0,
//     stamp absent;
//   - a history that replays and lands on `fen` (canonical-FEN compared,
//     section 2.1's canonicalFen) -- the replayed game under its own
//     repetition rule (gameFromHistory: PLACEMENT for komugi/gungijs, FULL
//     for ligungi -- KOM-1107 rule (5)), keys and all, stamp present;
//   - anything else (not an array, an illegal/unresolvable SAN, an invalid
//     `start`, a replay landing elsewhere, an unparseable FEN) -- the FEN
//     path again, stamp mismatch. A wrong history costs the search its
//     repetition awareness; trusting one would search the wrong position,
//     which is worse.
// `start` (KOM-1104) is the FEN the history replays from; absent keeps every
// existing caller byte-identical (gameFromHistory's own ADVANCED_POSITION
// default). The mode digit and hands the replay ends up with come from
// `start`, not from `fen` -- a Beginner `start` plus a Beginner-legal
// history replays under Beginner's own tower cap and piece set.
export function rootGame(
  fen: string,
  history: unknown,
  start?: string,
  rules?: RulesProfile | RulesConfig,
): { game: Game; stamp: HistoryStamp } {
  const effectiveRules = rules ?? DEFAULT_RULES_PROFILE;
  if (history === undefined || history === null) {
    return { game: new Game(boardFromFen(fen), RepetitionRule.FULL, 1, effectiveRules), stamp: HISTORY_ABSENT };
  }
  try {
    if (!Array.isArray(history)) {
      throw new Error(`history must be a list of SANs, got ${typeof history}`);
    }
    const replayed = gameFromHistory(history as readonly MoveHistoryEntry[], start, effectiveRules);
    const replayedFen = boardToFen(replayed.board);
    if (canonicalFen(replayedFen) !== canonicalFen(fen)) {
      throw new Error(`the history replays to ${replayedFen}, not to ${fen}`);
    }
    return { game: replayed, stamp: HISTORY_PRESENT };
  } catch {
    return { game: new Game(boardFromFen(fen), RepetitionRule.FULL, 1, effectiveRules), stamp: HISTORY_MISMATCH };
  }
}
