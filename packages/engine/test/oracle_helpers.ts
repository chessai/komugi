// KOM-1111: shared loading/canonicalisation helpers for the ligungi oracle
// fixture tests (oracle_legal.test.ts, oracle_terminal.test.ts). Fixture
// schema and canonical move format: web/engine/test/fixtures/ligungi_oracle.json's
// own `schema`/`canonicalMoveFormat` fields (read at CLEAN-ROOM time by
// inspecting the committed fixture, never ligungi source).
// loadOracleFixture() also merges in web/engine/test/fixtures/ligungi_oracle_extra.json
// (KOM-1191 fixtures2: hand-built, bot-free records restoring M10 coverage).
// CLEAN ROOM: everything here is derived from the fixture's own data, never
// from ligungi source.

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

import type { Board } from '../src/board.ts';
import { boardFromFen, boardToFen } from '../src/fen.ts';
import { Game } from '../src/game.ts';
import { generateRecords } from '../src/movegen.ts';
import { canonicalMove, RepetitionRule } from '../src/repetition.ts';
import type { RulesProfile } from '../src/repetition.ts';
import { buildSan } from '../src/san.ts';
import { isCheckmate, isInsufficientMaterial, isRepetitionDraw, isStalemate } from '../src/terminal.ts';
import { sha256Hex } from './fixture.ts';

const FIXTURE_PATH = fileURLToPath(new URL('./fixtures/ligungi_oracle.json', import.meta.url));
export const FIXTURE_SHA256 = '602d2f32cbd7520a2ad373c697db20bb494367362ea323736e7eeeae83b91010';

// KOM-1191 fixtures2: a companion fixture of hand-built, bot-free records
// (M10's only oracle-corpus catch, stripped out with the bot-match records --
// tickets/KOM-1191.md Findings). Same schema, merged into `records` by
// loadOracleFixture() so every fixture test sees one combined set.
const EXTRA_FIXTURE_PATH = fileURLToPath(new URL('./fixtures/ligungi_oracle_extra.json', import.meta.url));
export const EXTRA_FIXTURE_SHA256 = 'd28f1f9534534d014f42093cd3ea93f9453e72b6ff7ed6802d5bdcef89873ae2';

export type Terminal = { reason: string; winner: 'white' | 'black' | null } | null;

export type PositionRecord = {
  kind: 'position';
  id: string;
  mode: string;
  classes: string[];
  komugiFen: string;
  ligungiFen: string;
  komugi: { komugiOnly: string[]; ligungiOnly: string[]; terminal: Terminal };
  ligungi: { legal: string[]; standaloneDone: boolean; terminal: Terminal };
};

export type SequenceRecord = {
  kind: 'sequence';
  id: string;
  mode: string;
  classes: string[];
  komugiFen: string;
  ligungiFen: string;
  moves: string[];
  komugi: { terminal: Terminal; terminalPly: number };
  ligungi: { terminal: Terminal; terminalPly: number };
};

export type OracleRecord = PositionRecord | SequenceRecord;

export type OracleFixture = {
  schema: string;
  canonicalMoveFormat: string;
  classes: string[];
  counts: Record<string, number>;
  records: OracleRecord[];
};

// The 8 known divergence classes (KOM-1105's REVISION) plus 'none' -- a
// record belongs to zero or more of the 8 (classes are not mutually
// exclusive: a record's `classes` array can list several at once), and
// 'none' means that array is empty.
export const KNOWN_CLASSES = [
  'none',
  'self_exposure',
  'self_exposure_checkmate_claim',
  'archer_blocking',
  'betrayal_partial',
  'stalemate_scoring',
  'insufficient_material',
  'repetition_key',
  'false_checkmate',
] as const;

export type KnownClass = (typeof KNOWN_CLASSES)[number];

type ExtraFixture = {
  schema: string;
  generator: string;
  records: OracleRecord[];
};

let cachedFixture: OracleFixture | undefined;

// Loads + sha256-checks the main fixture, then merges in the KOM-1191
// companion fixture of hand-built marshal-reveal records (sha256-checked the
// same way, thrown on a ligungi_oracle_extra.json-named mismatch). Cached:
// every test file in this unit loads it once. A sha mismatch on either file
// means a committed fixture changed under us -- the pinned counts below
// would no longer mean anything, so this throws rather than silently testing
// against different data. The two record sets are concatenated then sorted
// by id (the generator's own comparison), after checking no extra id
// collides with a main-file id.
export function loadOracleFixture(): OracleFixture {
  if (cachedFixture) return cachedFixture;
  const raw = readFileSync(FIXTURE_PATH);
  const digest = sha256Hex(raw);
  if (digest !== FIXTURE_SHA256) {
    throw new Error(`ligungi_oracle.json sha256 mismatch: got ${digest}, want ${FIXTURE_SHA256}`);
  }
  const main = JSON.parse(raw.toString('utf8')) as OracleFixture;

  const extraRaw = readFileSync(EXTRA_FIXTURE_PATH);
  const extraDigest = sha256Hex(extraRaw);
  if (extraDigest !== EXTRA_FIXTURE_SHA256) {
    throw new Error(`ligungi_oracle_extra.json sha256 mismatch: got ${extraDigest}, want ${EXTRA_FIXTURE_SHA256}`);
  }
  const extra = JSON.parse(extraRaw.toString('utf8')) as ExtraFixture;

  const mainIds = new Set(main.records.map((r) => r.id));
  for (const r of extra.records) {
    if (mainIds.has(r.id)) {
      throw new Error(`ligungi_oracle_extra.json record id ${r.id} collides with a ligungi_oracle.json record`);
    }
  }

  const records = [...main.records, ...extra.records].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
  cachedFixture = { ...main, records };
  return cachedFixture;
}

export function positionRecords(fixture: OracleFixture): PositionRecord[] {
  return fixture.records.filter((r): r is PositionRecord => r.kind === 'position');
}

export function sequenceRecords(fixture: OracleFixture): SequenceRecord[] {
  return fixture.records.filter((r): r is SequenceRecord => r.kind === 'sequence');
}

// This record's classes for a per-class tally, 'none' standing in for the
// empty array (KOM-1105's own fixture note: class 'none' must tally 0 for
// every komugi/gungijs-faithful dimension).
function classesOf(record: OracleRecord): readonly string[] {
  return record.classes.length === 0 ? ['none'] : record.classes;
}

// Tallies `predicate(record)` (true = mismatch) across every class a record
// belongs to (KOM-1111 (b): "exact per-class mismatch counts"), seeded at 0
// for all KNOWN_CLASSES so a class that should read 0 is asserted, not
// merely absent from the object.
export function tallyByClass<T extends OracleRecord>(records: readonly T[], predicate: (record: T) => boolean): Record<KnownClass, number> {
  const out = Object.fromEntries(KNOWN_CLASSES.map((c) => [c, 0])) as Record<KnownClass, number>;
  for (const record of records) {
    if (!predicate(record)) continue;
    for (const c of classesOf(record)) out[c as KnownClass] += 1;
  }
  return out;
}

// The komugi/gungijs canonical legal set KOM-1105's M5 defines: ligungi's
// own legal set with the ligungi-only moves removed and the komugi-only
// moves (both precomputed by the fixture generator) added back.
export function expectedKomugiLegalSet(record: PositionRecord): Set<string> {
  const set = new Set(record.ligungi.legal);
  for (const m of record.komugi.ligungiOnly) set.delete(m);
  for (const m of record.komugi.komugiOnly) set.add(m);
  return set;
}

export function setsEqual(a: ReadonlySet<string>, b: ReadonlySet<string>): boolean {
  if (a.size !== b.size) return false;
  for (const x of a) if (!b.has(x)) return false;
  return true;
}

// Elements of `a` missing from `b`, sorted -- a readable failure message,
// never used on a hot path.
export function setDiff(a: ReadonlySet<string>, b: ReadonlySet<string>): string[] {
  return [...a].filter((x) => !b.has(x)).sort();
}

// Komugi's own verdict vocabulary (checkmate/stalemate/repetition/
// insufficient/marshal_captured), in Game.winState()'s own priority order
// (game.ts:193-216) -- the terminal dimension's komugi/gungijs expectation
// is `record.komugi.terminal`, generated by the same TS engine family in
// the same vocabulary, so this must reproduce it exactly, not merely agree
// on terminal-ness. `marshal_captured` is winState()'s final fallback
// (isCheckmate/isStalemate both false-out on a captured marshal); the
// fixture never reaches it, but the branch is reproduced for fidelity.
export function verdictOf(game: Game): Terminal {
  const [whiteWins, blackWins, draw] = game.winState();
  if (!whiteWins && !blackWins && !draw) return null;
  if (isCheckmate(game.board, game.rules)) {
    return { reason: 'checkmate', winner: whiteWins ? 'white' : 'black' };
  }
  if (isStalemate(game.board, game.rules)) {
    // KOM-1107 rule (4): under 'ligungi', no legal move while not in check
    // is a LOSS for the side to move, not a draw -- winState() already
    // scored it that way (whiteWins/blackWins one-hot), so the winner here
    // must follow suit; under komugi/gungijs winState() stays a draw, so
    // whiteWins and blackWins are both false and the winner stays null.
    return { reason: 'stalemate', winner: whiteWins ? 'white' : blackWins ? 'black' : null };
  }
  if (isRepetitionDraw(game.repetitionCount())) {
    return { reason: 'repetition', winner: null };
  }
  if (isInsufficientMaterial(game.board)) {
    return { reason: 'insufficient', winner: null };
  }
  return { reason: 'marshal_captured', winner: whiteWins ? 'white' : 'black' };
}

// A position record's Game, fresh (no replay history -- KOM-1105's fixture
// note: "position records carry no repetition history"), under `rules`.
export function gameForPosition(record: PositionRecord, rules: RulesProfile): Game {
  return new Game(boardFromFen(record.komugiFen), RepetitionRule.PLACEMENT, 1, rules);
}

// `komugiFen`/`ligungiFen` is a SAME position in each engine's own FEN
// dialect (the sequence's ply-1 start, not its end -- KOM-1111 fixture
// inspection), so `komugiFen` is `start` for a replay under ANY of the
// three profiles: we are replaying the fixture's canonical moves through
// OUR OWN engine, never ligungi's.
export function sequenceStart(record: SequenceRecord): string {
  return record.komugiFen;
}

// The legal move at `board` whose canonical form (repetition.ts's
// canonicalMove, the same one legalSet uses) equals the fixture's
// `canonical` string, as the real SAN resolveSan/rootGame need to replay it.
// Throws if none match -- a genuine illegal/unrecognised move is a fixture
// (or canonicaliser) bug, not something a test should silently skip.
export function sanForCanonical(board: Board, canonical: string): { id: number; san: string } {
  for (const { id, struct } of generateRecords(board)) {
    if (canonicalMove(struct) === canonical) return { id, san: buildSan(board, struct) };
  }
  throw new Error(`no legal move at this position canonicalises to ${canonical}`);
}

export type ReplayResult = {
  sans: string[];
  // The ply (1-based count of moves actually played) OUR engine first finds
  // terminal, or null if it never does within `record.moves`.
  terminalPly: number | null;
  finalFen: string;
};

// Replays `record.moves` (the fixture's canonical moves) one at a time
// through a fresh Game built the same way gameFromHistory builds one
// (root.ts's gameFromHistory: RepetitionRule.PLACEMENT for komugi/gungijs,
// FULL for ligungi -- KOM-1107 rule (5)), stopping at the first ply OUR
// engine's own winState() calls terminal -- so a sequence than runs past
// where we stop is simply not replayed any further, matching
// `terminalPly`'s definition (the ply the GAME ends at, not the fixture's
// move-list length).
export function replaySequence(record: SequenceRecord, rules: RulesProfile): ReplayResult {
  const rule = rules === 'ligungi' ? RepetitionRule.FULL : RepetitionRule.PLACEMENT;
  const game = new Game(boardFromFen(sequenceStart(record)), rule, 1, rules);
  const sans: string[] = [];
  let terminalPly: number | null = null;
  for (let i = 0; i < record.moves.length; i++) {
    const { id, san } = sanForCanonical(game.board, record.moves[i]);
    game.play(id);
    sans.push(san);
    const [whiteWins, blackWins, draw] = game.winState();
    if (whiteWins || blackWins || draw) {
      terminalPly = i + 1;
      break;
    }
  }
  return { sans, terminalPly, finalFen: boardToFen(game.board) };
}

// Terminal-ness + winner agreement only (never the `reason` spelling --
// komugi's and ligungi's own fixture verdicts use different vocabularies
// for the same outcome, e.g. 'checkmate' vs 'mate', so comparing the label
// text would manufacture mismatches that have nothing to do with the 8
// named divergence classes).
export function verdictsDisagree(a: Terminal, b: Terminal): boolean {
  if ((a === null) !== (b === null)) return true;
  if (a === null || b === null) return false;
  return a.winner !== b.winner;
}
