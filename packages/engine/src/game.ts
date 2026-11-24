// Game: terminal state, valid_moves, win_state, repetition, play/undo and the
// draft-curriculum root mask. Faithful port of src/komugi/engine/game.py's
// Game (391-932) and draft_placements/draft_root_mask (170-249).

import { Board, colorOf } from './board.ts';
import { decodeAction } from './actions.ts';
import { applyMove } from './apply.ts';
import { ACTION_SIZE, C_START, MT_ARATA, MT_BETRAY, MT_CAPTURE } from './constants.ts';
import { fillLegalMask } from './movegen.ts';
import {
  isCheckmate,
  isGameOver,
  isInsufficientMaterial,
  isRepetitionDraw,
  isStalemate,
  isTerminalNonmate,
  inCheck,
} from './terminal.ts';
import {
  DEFAULT_RULES_PROFILE,
  keyFor,
  RepetitionCounts,
  RepetitionRule,
  type RulesProfile,
} from './repetition.ts';
import { toRulesConfig } from './rules_config.ts';
import type { AbsoluteWdl, RulesConfig, ScoreTerminalHook, TerminalReason } from './rules_config.ts';

// game.py:501-506 / observation.py:70 -- the engine's own ply cap, not part
// of the reference rulebook.
export const MaxTurns = 512;

export type LastMove = { fromSq: number; fromTier: number; toSq: number; toTier: number } | null;

type UndoSnapshot = {
  cells: Int8Array;
  hand: Uint8Array;
  turn: number;
  mode: number;
  draftW: number;
  draftB: number;
  moveNumber: number;
  turns: number;
  progressPlies: number;
  lastMove: LastMove;
  key: Uint8Array;
  keyHi: number;
  keyLo: number;
  player: number;
};

const PROGRESS_MOVE_TYPES = new Set([MT_CAPTURE, MT_BETRAY, MT_ARATA]);

// KOM-1203 unit 6 (section 4): scoreTerminal's return must be one of the
// three one-hot absolute vectors -- never [0,0,0] (would silently turn a
// terminal position non-terminal downstream) and never fractional.
function isOneHotWdl(vector: AbsoluteWdl): boolean {
  if (vector.length !== 3) return false;
  let ones = 0;
  for (const v of vector) {
    if (v === 1) ones += 1;
    else if (v !== 0) return false;
  }
  return ones === 1;
}

function squareIndex(r: number, f: number): number {
  return r * 9 + (8 - f);
}

// Port of Game._decode_last_move (game.py:562-589). MUST be called with
// `board` still in the pre-move state the record `struct` was generated
// from: `struct[5]` (from_tier) is the source tower's height before apply,
// exactly as generate_all_moves' own records carry it.
function decodeLastMove(struct: Int32Array): LastMove {
  const moveType = struct[0];
  const toSq = squareIndex(struct[6]!, struct[7]!);
  const toTier = struct[8]!;
  if (moveType === MT_ARATA) return { fromSq: -1, fromTier: 0, toSq, toTier };
  return { fromSq: squareIndex(struct[3]!, struct[4]!), fromTier: struct[5]!, toSq, toTier };
}

// Port of draft_placements (game.py:170-204): every piece of `color` across
// all three tiers of the board. Only valid while `color` is still drafting.
export function draftPlacements(board: Board, color: number): number {
  const drafting = color === 0 ? board.draftW : board.draftB;
  if (!drafting) {
    throw new Error(
      `draftPlacements(${color}) is only defined while that color's draft bit is set`,
    );
  }
  let count = 0;
  for (let i = 0; i < board.cells.length; i++) {
    const code = board.cells[i]!;
    if (code !== 0 && colorOf(code) === color) count++;
  }
  return count;
}

// Port of draft_root_mask (game.py:207-249): withholds the C-block (終) at
// the root until `minPlacements` draft placements, with both fallbacks
// (no 終 legal at all; 終 the only legal move) returning the input mask
// unchanged rather than ever producing a zero-legal root.
export function draftRootMask(
  board: Board,
  validMask: Uint8Array,
  minPlacements: number,
): Uint8Array {
  if (minPlacements <= 0) return validMask;

  const side = board.turn;
  const drafting = side === 0 ? board.draftW : board.draftB;
  if (!drafting) return validMask;

  if (draftPlacements(board, side) >= minPlacements) return validMask;

  let anyC = false;
  for (let i = C_START; i < ACTION_SIZE; i++) {
    if (validMask[i]) {
      anyC = true;
      break;
    }
  }
  if (!anyC) return validMask; // nothing to withhold

  let anyBefore = false;
  for (let i = 0; i < C_START; i++) {
    if (validMask[i]) {
      anyBefore = true;
      break;
    }
  }
  if (!anyBefore) return validMask; // 終 is the only legal move -- never zero it out

  const masked = validMask.slice();
  masked.fill(0, C_START, ACTION_SIZE);
  return masked;
}

export class Game {
  board: Board;
  rule: RepetitionRule;
  // KOM-1105 (plumbing only, no rule change): exactly what the caller passed
  // (a RulesProfile string or a host RulesConfig object), immutable after
  // construction and carried by clone() (KOM-1203 F1 -- never re-resolved).
  rules: RulesProfile | RulesConfig;
  // KOM-1203: the resolved form of `rules`, read by every internal method
  // below instead of comparing `rules` against a profile string.
  config: RulesConfig;
  // KOM-1203 unit 6 (section 4, N3): engine-scoped, not part of RulesConfig
  // (it cannot cross postMessage). Not constructor-only/private: the engine
  // sets this on the root Game AFTER `rootGame`/`gameFromHistory` (root.ts,
  // outside this unit's allowlist) build it, so it must stay assignable from
  // outside this class. `clone()` copies it by reference, unchanged.
  scoreTerminal?: ScoreTerminalHook | undefined;
  player: number;
  turns = 0;
  progressPlies = 0;
  lastMove: LastMove = null;

  private key: Uint8Array;
  private keyHi: number;
  private keyLo: number;
  private counts = new RepetitionCounts();
  private undoStack: UndoSnapshot[] = [];

  // `initialOccurrences` seeds the key counter at construction (game.py's
  // `_history`/`_counts`, the bridge shape RT-19 needs): a plain `new
  // Game(board, rule)` is the per-FEN start (1 occurrence, no last move, no
  // progress plies, turns 0 -- game.py:426-433), and the test-only
  // bridge-shape construction reaches a count without having played it.
  constructor(
    board: Board,
    rule: RepetitionRule,
    initialOccurrences = 1,
    rules: RulesProfile | RulesConfig = DEFAULT_RULES_PROFILE,
  ) {
    this.board = board;
    this.rule = rule;
    this.rules = rules;
    this.config = toRulesConfig(rules);
    this.player = board.turn;
    this.key = keyFor(board, rule);
    const { hi, lo } = this.counts.setCount(this.key, initialOccurrences);
    this.keyHi = hi;
    this.keyLo = lo;
  }

  // KOM-1192a: how many play()s this Game could still undo() -- a search
  // that made/undid its own descent in place must leave this exactly where
  // it found it, pre-existing caller history included.
  get undoDepth(): number {
    return this.undoStack.length;
  }

  // Occurrences of the CURRENT position's key, this one included
  // (game.py:514-522).
  repetitionCount(): number {
    return this.counts.count(this.keyHi, this.keyLo, this.key);
  }

  // Verbatim port of Game.valid_moves (game.py:524-536).
  validMoves(): Uint8Array {
    const mask = new Uint8Array(ACTION_SIZE);
    if (isTerminalNonmate(this.board, this.repetitionCount(), this.config)) return mask;
    fillLegalMask(this.board, mask, this.config);
    let any = false;
    for (let i = 0; i < mask.length; i++) {
      if (mask[i]) {
        any = true;
        break;
      }
    }
    if (!any) return mask;
    if (inCheck(this.board, this.board.turn, this.config)) {
      if (isCheckmate(this.board, this.config)) mask.fill(0);
    } else if (isStalemate(this.board, this.config)) {
      mask.fill(0);
    }
    return mask;
  }

  // Verbatim port of Game.win_state (game.py:612-644): ply cap, then draft
  // gate, then is_game_over, then checkmate > stalemate > fourfold >
  // insufficient > marshal captured.
  // KOM-1203 knob 6 (plyCap): ligungi's "no ply cap" (rule 7) is now just
  // config.plyCap === null -- MaxTurns stays only as the preset's 512 value
  // and observation.ts/mlh.ts's plane/moves-left normaliser, never read here.
  winState(): [number, number, number] {
    const result: [number, number, number] = [0, 0, 0];
    let reason: TerminalReason | null = null;

    if (this.config.plyCap !== null && this.turns >= this.config.plyCap) {
      result[2] = 1;
      reason = 'ply-cap';
    } else if (this.board.draftW || this.board.draftB) {
      return result;
    } else {
      const count = this.repetitionCount();
      if (!isGameOver(this.board, count, this.config)) return result;

      if (isCheckmate(this.board, this.config)) {
        result[1 - this.board.turn] = 1;
        reason = 'checkmate';
      } else if (isStalemate(this.board, this.config)) {
        // KOM-1203 knob 4 (stalemateResult): under 'loss-for-mover' (ligungi
        // rule 4), no legal move while not in check is a LOSS for the side to
        // move, not a draw; komugi/gungijs unchanged.
        if (this.config.stalemateResult === 'loss-for-mover') {
          result[1 - this.board.turn] = 1;
        } else {
          result[2] = 1;
        }
        reason = 'stalemate';
      } else if (isRepetitionDraw(count)) {
        result[2] = 1;
        reason = 'repetition';
      } else if (isInsufficientMaterial(this.board)) {
        result[2] = 1;
        reason = 'insufficient';
      } else {
        result[1 - this.board.turn] = 1; // marshal captured
        reason = 'marshal-captured';
      }
    }

    // KOM-1203 unit 6 (section 4): every non-terminal path above already
    // returned; the hook only ever sees a position winState has already
    // decided is terminal, and can only rescore it (one-hot vector or
    // null/undefined for "keep the default"), never gate legality.
    if (this.scoreTerminal === undefined || reason === null) return result;
    const override = this.scoreTerminal(
      { board: this.board.copy(), turns: this.turns, reason },
      result as AbsoluteWdl,
    );
    if (override == null) return result;
    if (!isOneHotWdl(override)) {
      throw new Error('scoreTerminal must return a one-hot absolute [white, black, draw] vector or null');
    }
    return [...override] as [number, number, number];
  }

  // Port of Game.play_action (game.py:538-560), with an undo snapshot pushed
  // before the mutation (section 3: make/unmake instead of clone).
  play(actionId: number): void {
    const { struct } = decodeAction(this.board, actionId);
    const snapshot = this.snapshot();
    // BEFORE apply: from_tier (struct[5]) is the pre-move tower height.
    const lastMove = decodeLastMove(struct);
    applyMove(this.board, struct);

    this.progressPlies = PROGRESS_MOVE_TYPES.has(struct[0]!) ? 0 : this.progressPlies + 1;

    const key = keyFor(this.board, this.rule);
    const { hi, lo } = this.counts.increment(key);
    this.key = key;
    this.keyHi = hi;
    this.keyLo = lo;
    this.lastMove = lastMove;
    this.player = this.board.turn;
    this.turns += 1;
    this.undoStack.push(snapshot);
  }

  // Byte-exact inverse of play() (M9): restores cells, hand, the five
  // scalars, turns, progress_plies, the last move and decrements the key
  // count (deleting the entry rather than leaving a 0, so repetitionCount()
  // stays a plain map lookup).
  undo(): void {
    const snapshot = this.undoStack.pop();
    if (!snapshot) throw new Error('undo: no move to undo');

    this.counts.decrement(this.keyHi, this.keyLo, this.key);

    this.board.cells.set(snapshot.cells);
    this.board.hand.set(snapshot.hand);
    this.board.turn = snapshot.turn;
    this.board.mode = snapshot.mode;
    this.board.draftW = snapshot.draftW;
    this.board.draftB = snapshot.draftB;
    this.board.moveNumber = snapshot.moveNumber;
    this.turns = snapshot.turns;
    this.progressPlies = snapshot.progressPlies;
    this.lastMove = snapshot.lastMove;
    this.key = snapshot.key;
    this.keyHi = snapshot.keyHi;
    this.keyLo = snapshot.keyLo;
    this.player = snapshot.player;
  }

  // An independent copy (section 3: "clone() exists for the root copy and
  // tests"). Carries the board, key, repetition counts, player, turns,
  // progress plies and last move exactly; the clone starts with its own
  // empty undo stack, since every clone is used independently of the
  // source's own undo history. Private fields are set directly here (legal
  // TS across two instances of the same class) rather than reached into
  // from outside by name, which is what dropped the repetition history in
  // the first place (KOM-1079: the clone built its own fresh `RepetitionCounts`
  // via `new Game(...)` and never copied the source's).
  clone(): Game {
    // KOM-1203 N1/N3: `rules`/`config` are copied directly, by reference,
    // from the source Game below -- never re-resolved via toRulesConfig,
    // since a host-supplied RulesConfig object might be mutated between the
    // source's construction and this clone (DEFAULT_RULES_PROFILE here is
    // just a cheap, always-valid placeholder for the constructor call).
    const copy = new Game(this.board.copy(), this.rule, 1, DEFAULT_RULES_PROFILE);
    copy.rules = this.rules;
    copy.config = this.config;
    copy.scoreTerminal = this.scoreTerminal;
    copy.player = this.player;
    copy.turns = this.turns;
    copy.progressPlies = this.progressPlies;
    copy.lastMove = this.lastMove;
    copy.key = this.key.slice();
    copy.keyHi = this.keyHi;
    copy.keyLo = this.keyLo;
    copy.counts = this.counts.copy();
    copy.undoStack = [];
    return copy;
  }

  private snapshot(): UndoSnapshot {
    return {
      cells: this.board.cells.slice(),
      hand: this.board.hand.slice(),
      turn: this.board.turn,
      mode: this.board.mode,
      draftW: this.board.draftW,
      draftB: this.board.draftB,
      moveNumber: this.board.moveNumber,
      turns: this.turns,
      progressPlies: this.progressPlies,
      lastMove: this.lastMove,
      key: this.key,
      keyHi: this.keyHi,
      keyLo: this.keyLo,
      player: this.player,
    };
  }
}
