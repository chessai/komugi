// KOM-1203 unit 5b (design docs/design/kom1203-rules-config.md section 6's
// AC2 and F2(b)): engine-level tests for the host-supplied `RulesConfig`
// object that have no home in unit 1's `rules_config.test.ts` (AC2 needs a
// live `WorkerEngine`). Also covers F2(b)'s echo-rule fix: `echoRules`
// (engine.ts) must match a preset by VALUE (`rulesConfigKey`), not object
// identity -- a `RulesConfig` that crosses `postMessage` is a structured
// clone, a different object with the same fields, and `toRulesConfig`'s
// `WeakSet` fast path (rules_config.ts) never fires for a clone.

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { boardFromFen, boardToFen } from '../src/fen.ts';
import { Game } from '../src/game.ts';
import { WorkerEngine } from '../src/engine.ts';
import type { WorkerReply, WorkerRequest } from '../src/protocol.ts';
import { ADVANCED_POSITION } from '../src/constants.ts';
import { RepetitionRule } from '../src/repetition.ts';
import { HISTORY_PRESENT, rootGame } from '../src/root.ts';
import { RULES_PRESETS, RULES_PROFILES, rulesConfigKey } from '../src/rules_config.ts';
import type { RulesConfig, RulesProfile } from '../src/rules_config.ts';
import {
  loadOracleFixture,
  sanForCanonical,
  sequenceRecords,
  sequenceStart,
  verdictOf,
  verdictsDisagree,
} from './oracle_helpers.ts';
import type { SequenceRecord } from './oracle_helpers.ts';

function isOk(reply: WorkerReply): reply is WorkerReply & { action: number; san: string; stats: Record<string, unknown> } {
  return !('err' in reply);
}

// rules_profile.test.ts:30-35's own helper.
function statsOf(reply: WorkerReply): Record<string, unknown> {
  const stats = { ...(reply as { stats: Record<string, unknown> }).stats };
  delete stats.ms;
  delete stats.pondered;
  return stats;
}

// Fresh literals (section 3's table), one per profile, NOT a spread of
// `RULES_PRESETS[p]` -- AC2's stronger claim is that VALUES, not identity
// with the module's own singleton, drive behaviour.
const LITERAL_BY_PROFILE: Record<RulesProfile, RulesConfig> = {
  komugi: {
    selfExposureFilter: false,
    archerElbowBlocking: false,
    tacticianBetrayalMode: 'any-subset',
    stalemateResult: 'draw',
    marshalCaptureIsEscape: true,
    plyCap: 512,
    repetitionKey: 'placement',
  },
  gungijs: {
    selfExposureFilter: false,
    archerElbowBlocking: false,
    tacticianBetrayalMode: 'any-subset',
    stalemateResult: 'draw',
    marshalCaptureIsEscape: false,
    plyCap: 512,
    repetitionKey: 'placement',
  },
  ligungi: {
    selfExposureFilter: true,
    archerElbowBlocking: true,
    tacticianBetrayalMode: 'all-or-nothing',
    stalemateResult: 'loss-for-mover',
    marshalCaptureIsEscape: true,
    plyCap: null,
    repetitionKey: 'full',
  },
};

// ---- AC2: a custom config equal in value to a named preset -----------------

for (const profile of RULES_PROFILES) {
  test(`AC2: a ${profile} literal is indistinguishable from the named preset through a fresh WorkerEngine`, () => {
    const custom = LITERAL_BY_PROFILE[profile];
    assert.equal(
      rulesConfigKey(custom),
      rulesConfigKey(RULES_PRESETS[profile]),
      `${profile}: literal must have the same rulesConfigKey as the preset`,
    );

    // Two fresh engines -- the bias table persists across requests on the
    // same engine (engine.ts:645), so reusing one would let the string
    // run's bias entries leak into the literal run.
    const viaString = new WorkerEngine();
    const viaLiteral = new WorkerEngine();
    const request = (rules: RulesProfile | RulesConfig): WorkerRequest => ({
      id: 1,
      fen: ADVANCED_POSITION,
      sims: 16,
      rules,
    });

    const replyString = viaString.handle(request(profile));
    const replyLiteral = viaLiteral.handle(request(custom));
    assert.ok(isOk(replyString), JSON.stringify(replyString));
    assert.ok(isOk(replyLiteral), JSON.stringify(replyLiteral));
    if (!isOk(replyString) || !isOk(replyLiteral)) return;

    assert.deepEqual(statsOf(replyString), statsOf(replyLiteral));

    if (profile !== 'komugi') {
      assert.equal(replyString.stats.rules, profile, 'string run must echo the profile name');
      assert.equal(replyLiteral.stats.rules, profile, 'literal run must echo the profile name (F2(b))');
    }
  });
}

// ---- F2(b): oracle_terminal.test.ts's repetition_key-only sequences, replayed
// again with a ligungi literal rather than the string -------------------------

type SequenceReplay = {
  sans: string[];
  terminalPly: number | null;
  finalFen: string;
  // The fen just BEFORE the terminal-causing move (equal to the start fen
  // when terminalPly is null -- no terminal move exists in `record.moves`).
  // `WorkerEngine.handle` on an ALREADY-terminal root throws inside
  // `buildRoot`, which `handle()` turns into an `err` reply carrying no
  // `stats` at all (engine.ts:486-498) -- so F2(b)'s "the reply's
  // stats.rules must read 'ligungi'" claim is checked at the last
  // non-terminal ply instead, not at the fixture's own terminal position.
  preTerminalFen: string;
};

// Mirrors oracle_helpers.ts's `replaySequence` -- not reused directly, since
// that helper's `rules` parameter is pinned to `RulesProfile` and cannot take
// the ligungi `RulesConfig` literal this test needs. Every call here passes
// a ligungi-EQUIVALENT `rules` (the string, or a config with
// `repetitionKey: 'full'`), so the repetition rule is always FULL, matching
// `gameFromHistory`/`replaySequence`'s own 'ligungi' branch.
function replayLigungiEquivalent(record: SequenceRecord, rules: RulesProfile | RulesConfig): SequenceReplay {
  const game = new Game(boardFromFen(sequenceStart(record)), RepetitionRule.FULL, 1, rules);
  const sans: string[] = [];
  let terminalPly: number | null = null;
  let preTerminalFen = boardToFen(game.board);
  for (let i = 0; i < record.moves.length; i++) {
    const fenBeforeThisMove = boardToFen(game.board);
    const { id, san } = sanForCanonical(game.board, record.moves[i]);
    game.play(id);
    sans.push(san);
    const [whiteWins, blackWins, draw] = game.winState();
    if (whiteWins || blackWins || draw) {
      terminalPly = i + 1;
      preTerminalFen = fenBeforeThisMove;
      break;
    }
  }
  return { sans, terminalPly, finalFen: boardToFen(game.board), preTerminalFen };
}

test('F2(b): repetition_key-only oracle sequences replay identically under rules: "ligungi" and a ligungi literal', () => {
  const fixture = loadOracleFixture();
  const sequences = sequenceRecords(fixture).filter(
    (r) => r.classes.length === 1 && r.classes[0] === 'repetition_key',
  );
  assert.ok(sequences.length > 0, 'expected at least one repetition_key-only sequence record');

  const LIGUNGI_LITERAL = LITERAL_BY_PROFILE.ligungi;
  let preTerminalCase: { record: SequenceRecord; history: string[]; fen: string } | null = null;

  for (const record of sequences) {
    const viaString = replayLigungiEquivalent(record, 'ligungi');
    const viaLiteral = replayLigungiEquivalent(record, LIGUNGI_LITERAL);
    assert.deepEqual(viaLiteral.sans, viaString.sans, `${record.id}: sans`);
    assert.equal(viaLiteral.terminalPly, viaString.terminalPly, `${record.id}: terminal ply`);
    assert.equal(viaLiteral.finalFen, viaString.finalFen, `${record.id}: final fen`);

    const stampString = rootGame(viaString.finalFen, viaString.sans, sequenceStart(record), 'ligungi');
    const stampLiteral = rootGame(viaLiteral.finalFen, viaLiteral.sans, sequenceStart(record), LIGUNGI_LITERAL);
    assert.equal(stampString.stamp, HISTORY_PRESENT, `${record.id}: rootGame stamp (string)`);
    assert.equal(stampLiteral.stamp, stampString.stamp, `${record.id}: rootGame stamp must match between string and literal`);
    assert.equal(
      verdictsDisagree(verdictOf(stampString.game), record.ligungi.terminal),
      false,
      `${record.id}: string verdict must agree with record.ligungi.terminal`,
    );
    assert.equal(
      verdictsDisagree(verdictOf(stampLiteral.game), record.ligungi.terminal),
      false,
      `${record.id}: literal verdict must agree with record.ligungi.terminal`,
    );

    if (preTerminalCase === null && viaString.terminalPly !== null && viaString.terminalPly >= 1) {
      preTerminalCase = {
        record,
        history: viaString.sans.slice(0, viaString.terminalPly - 1),
        fen: viaString.preTerminalFen,
      };
    }
  }

  assert.ok(
    preTerminalCase !== null,
    'expected at least one repetition_key-only sequence with a terminal ply, to probe stats.rules on the preceding non-terminal root',
  );
  const { history, fen, record } = preTerminalCase!;
  const start = sequenceStart(record);
  const buildRequest = (rules: RulesProfile | RulesConfig): WorkerRequest => ({
    id: 1,
    fen,
    history,
    start,
    sims: 4,
    rules,
  });

  const replyString = new WorkerEngine().handle(buildRequest('ligungi'));
  const replyLiteral = new WorkerEngine().handle(buildRequest(LIGUNGI_LITERAL));
  assert.ok(isOk(replyString), JSON.stringify(replyString));
  assert.ok(isOk(replyLiteral), JSON.stringify(replyLiteral));
  if (!isOk(replyString) || !isOk(replyLiteral)) return;
  assert.equal(replyString.stats.rules, 'ligungi', 'string request must echo "ligungi"');
  assert.equal(replyLiteral.stats.rules, 'ligungi', 'literal request must echo "ligungi" too (F2(b))');
});

// ---- Echo cases (section 6's "F2(b)'s test" + this unit's own deviation fix)

// NOTE: both `structuredClone` cases below are regression pins for the
// fixed, `rulesConfigKey`-based (value-equal) `echoRules` comparison --
// `structuredClone` produces a different object with the same field values,
// which never hits `toRulesConfig`'s `WeakSet` fast path, so only a
// value-equal comparison (not object identity) echoes the matching preset.

test('echo: structuredClone(RULES_PRESETS.ligungi) echoes "ligungi"', () => {
  const engine = new WorkerEngine();
  const reply = engine.handle({
    id: 1,
    fen: ADVANCED_POSITION,
    sims: 4,
    rules: structuredClone(RULES_PRESETS.ligungi),
  });
  assert.ok(isOk(reply), JSON.stringify(reply));
  if (!isOk(reply)) return;
  assert.equal(reply.stats.rules, 'ligungi');
});

test('echo: structuredClone(RULES_PRESETS.komugi) is byte-identical (ms zeroed) to an absent `rules`', () => {
  const absent = new WorkerEngine();
  const cloned = new WorkerEngine();
  const replyAbsent = absent.handle({ id: 1, fen: ADVANCED_POSITION, sims: 16 });
  const replyCloned = cloned.handle({
    id: 1,
    fen: ADVANCED_POSITION,
    sims: 16,
    rules: structuredClone(RULES_PRESETS.komugi),
  });
  assert.ok(isOk(replyAbsent), JSON.stringify(replyAbsent));
  assert.ok(isOk(replyCloned), JSON.stringify(replyCloned));
  if (!isOk(replyAbsent) || !isOk(replyCloned)) return;

  const zeroMs = (reply: WorkerReply & { stats: Record<string, unknown> }): string =>
    JSON.stringify({ ...reply, stats: { ...reply.stats, ms: 0 } });
  assert.equal(zeroMs(replyCloned), zeroMs(replyAbsent));
  assert.ok(!('rules' in replyCloned.stats), 'a komugi-equivalent config must omit stats.rules, same as the default path');
});

test('echo: a genuinely custom config (one knob changed from a preset) echoes the canonical RulesConfig object', () => {
  // gungijs with archerElbowBlocking flipped -- still coherent (F3:
  // selfExposureFilter stays false, so the selfExposureFilter/
  // marshalCaptureIsEscape pairing rule never applies), and not equal in
  // value to any of the three presets.
  const custom: RulesConfig = { ...LITERAL_BY_PROFILE.gungijs, archerElbowBlocking: true };
  for (const profile of RULES_PROFILES) {
    assert.notEqual(
      rulesConfigKey(custom),
      rulesConfigKey(RULES_PRESETS[profile]),
      `custom config must not match the ${profile} preset's rulesConfigKey`,
    );
  }

  const engine = new WorkerEngine();
  const reply = engine.handle({ id: 1, fen: ADVANCED_POSITION, sims: 4, rules: custom });
  assert.ok(isOk(reply), JSON.stringify(reply));
  if (!isOk(reply)) return;
  assert.deepEqual(reply.stats.rules, custom);
});
