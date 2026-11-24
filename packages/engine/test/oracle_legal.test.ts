// KOM-1111: the ligungi oracle fixture's legal-move-set dimension (KOM-1105's
// M5), kept separate from the terminal-verdict dimension (oracle_terminal.test.ts)
// per the ticket's "keep the legal-set and verdict dimensions separate, one
// test file per dimension".
//
// (a) 'komugi' and 'gungijs' carry no legal-set rule change of their own
//     (rules (2)/(3) below are gated on `rules === 'ligungi'` at every call
//     site), so every position record's canonical legal set must equal
//     ligungi.legal - komugi.ligungiOnly + komugi.komugiOnly, exactly, with
//     no exceptions -- this is the plumbing assertion, not a pinned count.
// (b) 'ligungi' diverges on the 6 known movegen-affecting classes; the exact
//     per-class mismatch count is pinned here (class 'none' must read 0).
//     KOM-1106/1138 have now landed rules (2) (archer_blocking) and (3)
//     (betrayal_partial); the exact-match test below (KOM-1138 F2) asserts
//     the in-scope classes directly rather than through this tally.

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { boardFromFen } from '../src/fen.ts';
import { legalSet } from '../src/repetition.ts';
import type { RulesProfile } from '../src/repetition.ts';
import {
  type KnownClass,
  type PositionRecord,
  expectedKomugiLegalSet,
  FIXTURE_SHA256,
  loadOracleFixture,
  positionRecords,
  setDiff,
  setsEqual,
  tallyByClass,
} from './oracle_helpers.ts';

const fixture = loadOracleFixture();
const positions = positionRecords(fixture);

test('ligungi_oracle.json: sha256 and schema', () => {
  assert.equal(fixture.schema, 'ligungi-oracle/1');
  assert.ok(positions.length > 0, 'expected at least one position record');
  // loadOracleFixture() already threw on a mismatch; FIXTURE_SHA256 itself
  // must be the ticket's pinned value, not something that silently drifted.
  assert.equal(FIXTURE_SHA256, '602d2f32cbd7520a2ad373c697db20bb494367362ea323736e7eeeae83b91010');
});

function assertKomugiLegalSet(rules: RulesProfile): void {
  for (const record of positions) {
    const board = boardFromFen(record.komugiFen);
    const mine = legalSet(board, rules);
    const expected = expectedKomugiLegalSet(record);
    assert.ok(
      setsEqual(mine, expected),
      `${record.id} (${rules}): extra=${JSON.stringify(setDiff(mine, expected))} missing=${JSON.stringify(setDiff(expected, mine))}`,
    );
  }
}

test("komugi: every position record's canonical legal set equals ligungi.legal - ligungiOnly + komugiOnly", () => {
  assertKomugiLegalSet('komugi');
});

test("gungijs: every position record's canonical legal set equals ligungi.legal - ligungiOnly + komugiOnly", () => {
  assertKomugiLegalSet('gungijs');
});

// KOM-1111 (b): pinned per-class mismatch counts against ligungi.legal,
// computed straight from this unit's own legalSet/canonicalMove -- not from
// the fixture's precomputed komugiOnly/ligungiOnly arrays, so a canonicaliser
// bug that happened to agree with those arrays would still be caught by (a)
// above failing instead. Printed table first (ticket: "print the table, then
// hard-code it"), then the hard-coded expectation.
// KOM-1106 (SPLIT 2026-11-14): archer_blocking (35 -> 3) dropped once this
// unit's rule (2) post-filter (rules_ligungi.ts, movegen_board.ts,
// terminal.ts) applied. KOM-1138 (fix-up): betrayal_partial (19 -> 5).
// KOM-1139: the rule-(1) self-exposure filter (ligungi_filter.ts) now also
// applies, and every remaining class -- self_exposure, self_exposure_
// checkmate_claim, the last 3 archer_blocking + 5 betrayal_partial (each of
// which also carried self_exposure, per KOM-1106's own note), stalemate_
// scoring and false_checkmate -- drops to 0. No class needed a documented
// exception: the exact-match test below (now unconditional, not restricted
// to an in-scope subset) confirms every one of the 1690 position records
// matches ligungi.legal exactly, including every record with
// `ligungi.standaloneDone === true` (checked directly, see that test).
const LIGUNGI_LEGAL_MISMATCH_COUNTS: Record<KnownClass, number> = {
  none: 0,
  self_exposure: 0,
  self_exposure_checkmate_claim: 0,
  archer_blocking: 0,
  betrayal_partial: 0,
  stalemate_scoring: 0,
  insufficient_material: 0,
  repetition_key: 0,
  false_checkmate: 0,
};

test('ligungi: pinned per-class legal-set mismatch counts (none = 0)', () => {
  function mismatches(record: PositionRecord): boolean {
    const board = boardFromFen(record.komugiFen);
    const mine = legalSet(board, 'ligungi');
    return !setsEqual(mine, new Set(record.ligungi.legal));
  }
  const counts = tallyByClass(positions, mismatches);
  console.log('ligungi legal-set mismatch counts by class:', counts);
  assert.deepEqual(counts, LIGUNGI_LEGAL_MISMATCH_COUNTS);
});

// KOM-1139: with rule (1) (self-exposure, ligungi_filter.ts) now landed
// alongside rules (2)/(3) (KOM-1106/1138), EVERY position record -- every
// class, no in-scope subset -- must match ligungi.legal exactly. This
// supersedes KOM-1138's in-scope-only version (the subset restriction is no
// longer needed: LIGUNGI_LEGAL_MISMATCH_COUNTS above is all zero).
test('ligungi: every fixture position record matches ligungi.legal exactly (all classes)', () => {
  for (const record of positions) {
    const board = boardFromFen(record.komugiFen);
    const mine = legalSet(board, 'ligungi');
    const expected = new Set(record.ligungi.legal);
    assert.ok(
      setsEqual(mine, expected),
      `${record.id}: extra=${JSON.stringify(setDiff(mine, expected))} missing=${JSON.stringify(setDiff(expected, mine))}`,
    );
  }
});

// The ticket calls out "records with a stand-alone done excluded as before"
// as a documented exception to watch for. Checked directly: every one of
// the 117 position records with `ligungi.standaloneDone === true` (a draft
// position whose legal set is a lone 終-finishing placement) is already
// covered by the unconditional exact-match test above, and needs no
// exclusion -- rule (1) never applies mid-draft (ligungi_filter.ts's own
// arataFilterContext gate), so these records were never going to diverge on
// this dimension in the first place.
test('ligungi: every standaloneDone position record also matches ligungi.legal exactly (no exclusion needed)', () => {
  const standaloneDone = positions.filter((r) => r.ligungi.standaloneDone);
  assert.ok(standaloneDone.length > 0, 'expected at least one standaloneDone record');
  for (const record of standaloneDone) {
    const board = boardFromFen(record.komugiFen);
    const mine = legalSet(board, 'ligungi');
    const expected = new Set(record.ligungi.legal);
    assert.ok(setsEqual(mine, expected), `${record.id}: standaloneDone record mismatched`);
  }
});
