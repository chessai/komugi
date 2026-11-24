// Ports the match tree's own contracts (src/komugi/search/match/state.py)
// and the GameState adapter the match tree is actually handed
// (src/komugi/selfplay/adapter.py:317-486, src/komugi/search/match/adapter.py)
// (docs/design/clientside-engine-port.md section 2.6). One class rather than
// Python's MoverState/MatchState/MatchGameState trio: TS has no Protocol
// seam to keep the search package from importing the self-play adapter at
// runtime, so there is only one engine here and nothing to keep structurally
// apart.

import { Game, draftRootMask } from '../game.ts';
import { placementKey } from '../repetition.ts';
import { parsePlaneMapVersion } from '../rule_features.ts';
import type { PlaneMapVersion } from '../rule_features.ts';

export const WDL = 3;

// What node.ts's expand (tree.ts) needs: legality, make/unmake, the
// terminal WDL for a decided leaf, and the 243 placement-key bytes the
// fake evaluator hashes (fake_evaluator.ts). `observation()` is deliberately
// not on this interface: nothing in U7b1 reads it (the fake evaluator reads
// placementKeyBytes instead), and observation.ts is U5, not a dependency.
export interface MatchState {
  // KOM-1103: the ORT-web leaf evaluators (evaluator.ts's
  // makeOrtMlhLeafEvaluator, ort_browser.ts's makeOrtMlhLeafEvaluatorBrowser)
  // build the network observation straight off the underlying `Game`; this
  // port has exactly one state implementation (MatchGameState below), so the
  // field belongs on the interface rather than behind an unsafe cast.
  game: Game;
  // KOM-1162: the observation plane-map version every leaf evaluator encodes
  // this state under (rule_features.ts). Optional so a hand-built state
  // reads as the default, 3.
  planeMapVersion?: PlaneMapVersion;
  legalMask(): Uint8Array;
  play(action: number): void;
  // KOM-1192a: the exact inverse of the most recent play(), restoring the
  // state to exactly where it was (board, key, repetition counts) -- the
  // descent's make/undo discipline, not a clone.
  undo(): void;
  clone(): MatchState;
  isTerminal(): boolean;
  toPlay(): number;
  terminalWdl(): Float64Array;
  placementKeyBytes(): Uint8Array;
}

// WdlEval / WdlEvaluator (state.py:120-148): one network evaluation --
// policyLogits raw and unmasked (A,), wdl a (3,) [w, d, l] probability in
// the mover's perspective.
export type WdlEval = { policyLogits: Float32Array; wdl: Float64Array };
export type WdlEvaluator = (state: MatchState) => WdlEval;

// GameState (adapter.py:269-499) + MatchGameState (match/adapter.py:50-120):
// the DRAFTMIN root mask, make/unmake play, clone, and terminal_wdl's frame
// change, all over one mutable `Game`.
export class MatchGameState implements MatchState {
  game: Game;
  minPlacements: [number, number];
  // design section 2.6 S5: "the root view is depth === 0" under make/unmake.
  // 0 at construction unless `isRoot` is explicitly false (adapter.py's own
  // `is_root: bool = False` default); `play` always moves away from 0 and
  // never back, exactly as `_is_root` only ever turns false. `clone` carries
  // the value forward unchanged, exactly as Python's clone carries `_is_root`.
  depth: number;
  // KOM-1162: carried unchanged through `clone`, like `minPlacements`.
  planeMapVersion: PlaneMapVersion;

  constructor(
    game: Game,
    opts: {
      minPlacements?: number | [number, number];
      isRoot?: boolean;
      depth?: number;
      planeMapVersion?: PlaneMapVersion;
    } = {},
  ) {
    this.game = game;
    const mp = opts.minPlacements ?? 0;
    this.minPlacements = Array.isArray(mp) ? [mp[0], mp[1]] : [mp, mp];
    this.depth = opts.depth ?? (opts.isRoot ?? false ? 0 : 1);
    this.planeMapVersion = parsePlaneMapVersion(opts.planeMapVersion);
  }

  get isRoot(): boolean {
    return this.depth === 0;
  }

  // valid_moves, draft_root_mask'ed at the root for the mover's own
  // threshold (adapter.py:317-348). Off the root, or with no threshold,
  // the engine's own mask is returned unchanged.
  legalMask(): Uint8Array {
    const mask = this.game.validMoves();
    const threshold = this.minPlacements[this.game.player]!;
    if (!(this.isRoot && threshold > 0)) return mask;
    return draftRootMask(this.game.board, mask, threshold);
  }

  // play (adapter.py:350-373): drops the root view unless `keepRoot` says
  // otherwise. `keepRoot` is for the self-play loop and nothing else; the
  // search calls play(action) positionally through MatchState, which has no
  // such parameter, so it can never reach it.
  play(action: number, keepRoot = false): void {
    this.game.play(action);
    if (!keepRoot) this.depth += 1;
  }

  // KOM-1192a: inverse of play(action) (never called with keepRoot=true by
  // the search, which only ever calls play(action) positionally through
  // MatchState) -- undo the underlying game, then undo the depth bump.
  undo(): void {
    this.game.undo();
    this.depth -= 1;
  }

  // clone (adapter.py:375-388): an independent copy, carrying the root view.
  clone(): MatchGameState {
    return new MatchGameState(this.game.clone(), {
      minPlacements: this.minPlacements,
      depth: this.depth,
      planeMapVersion: this.planeMapVersion,
    });
  }

  // is_terminal (adapter.py:390-397): win_state() having any slot set.
  isTerminal(): boolean {
    const w = this.game.winState();
    return w[0] !== 0 || w[1] !== 0 || w[2] !== 0;
  }

  // to_play (adapter.py:399-406).
  toPlay(): number {
    return this.game.player;
  }

  // terminal_wdl (adapter.py:471-499, match/adapter.py:98-120): the mover's
  // one-hot (win, draw, loss) from the absolute [white, black, draw] result.
  // No draw score anywhere in it.
  terminalWdl(): Float64Array {
    const outcome = this.game.winState();
    const mover = this.game.player;
    const out = new Float64Array(WDL);
    out[0] = outcome[mover]!;
    out[1] = outcome[2]!;
    out[2] = outcome[1 - mover]!;
    return out;
  }

  // The 243-byte PLACEMENT key (design 4.3's `h = fnv1a32(placement_key_bytes)`).
  placementKeyBytes(): Uint8Array {
    return placementKey(this.game.board);
  }
}
