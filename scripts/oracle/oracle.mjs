#!/usr/bin/env node
/**
 * oracle.mjs -- regenerate the gungi.js golden corpus consumed by
 * `tests/engine/` (see `tests/engine/conftest.py::load_golden`).
 *
 * KOM-101 G1.  The original generator named in
 * `tests/engine/goldens/MANIFEST.json` was never committed; this is a NEW,
 * deterministic reimplementation.  It reproduces the committed corpus's
 * *format* exactly.  It does NOT reproduce the committed corpus byte for
 * byte -- the old PRNG and its draw order are lost, so the game/ply counts
 * this script emits are seed-dependent and are written into the MANIFEST it
 * emits rather than assumed.
 *
 * ===========================================================================
 * PRNG
 * ===========================================================================
 * mulberry32 -- a 32-bit-state generator that uses only `Math.imul`, `>>>`,
 * `^`, `+` and one float division.  Every one of those is exactly specified
 * by ECMA-262 on uint32 values, so the stream is bit-identical on every node
 * version and platform.  (`Math.random` is never called anywhere in this
 * file, and must never be.)
 *
 * ---------------------------------------------------------------------------
 * DRAW ORDER -- the determinism contract.  Changing any line of it changes
 * every ply downstream of the change, so treat it as frozen:
 *
 *   1. EXACTLY ONE mulberry32 stream exists per run, seeded with
 *      `seed >>> 0`, and it is never reseeded or forked.
 *   2. Modes are visited in `MODES` order: intro, beginner, intermediate,
 *      advanced.  The `betray` mode (see below) is visited LAST, strictly
 *      after every one of those draws.
 *   3. Within a mode, games run 0 .. gamesFor(mode)-1 in ascending order.
 *   4. Within a game, plies run in ascending order.
 *   5. EXACTLY ONE draw is consumed per RECORDED ply:
 *          idx = Math.floor(next() * legal.length)
 *      where `legal` is the code-point-sorted SAN list for that position
 *      (see `sortSan` -- sorting, not gungi.js's internal generation order,
 *      is what makes the draw stable across gungi.js patch releases).
 *   6. No other call site consumes the stream.  In particular, terminal
 *      positions -- the ones that end a game and are therefore NOT recorded
 *      -- consume nothing.
 *
 * ---------------------------------------------------------------------------
 * THE `betray` MODE (KOM-101 G2) -- an APPENDED emission, off by default.
 *
 * `--games betray=N` plays N extra advanced-rules games AFTER the four modes
 * above have finished drawing, and emits `playouts_betray.json`.  Its default
 * count is 0, so the default invocation draws nothing extra and the four
 * committed playout files plus `fen_roundtrip.json` stay bit-identical.
 *
 * It differs from the four frozen modes in exactly three ways, all of them
 * confined to `playBetrayGame`:
 *
 *   a. THE DRAW IS BIASED, deliberately.  Whenever the sorted legal list
 *      contains at least one 返 (betray) SAN, the draw domain is that subset
 *      instead of the whole list.  Uniform random play essentially never
 *      chooses a betrayal -- an advisor probe over 17,522 random plies found
 *      33 betray-LEGAL plies and 0 betray-CHOSEN ones -- so an unbiased mode
 *      cannot produce a consumable betrayal corpus at any sane size.  The
 *      bias is why this is a SEPARATE mode rather than a change to `advanced`:
 *      the meaning of the seed for the existing modes is untouched.
 *   b. ONE DRAW PER PLAYED PLY, not per recorded ply.  Every non-terminal
 *      position in a betray game consumes exactly one draw whether or not the
 *      ply is emitted.  (Terminal positions still consume nothing.)
 *   c. THE EMISSION IS FILTERED.  Only plies whose `legal` list contains a 返
 *      SAN are written out; games that contribute none are omitted entirely.
 *      Every ply is self-contained -- `fen_before` -> `chosen` -> `fen_after`
 *      -- and each recorded ply's own five fields are all any consumer reads,
 *      so a filtered game is still a valid input.  A `games` entry in this
 *      file is therefore NOT a contiguous playout and its plies must never be
 *      chained (`fen_after` of ply i is generally not `fen_before` of i+1).
 *
 * Betray plies are NOT fed into the `fen_roundtrip.json` input set, so that
 * file is unchanged by any betray count.  `playouts_betray.json` and
 * `MANIFEST.json` are the only files a betray run adds or moves.
 *
 * ===========================================================================
 * OUTPUT FORMAT -- must match the committed corpus exactly
 * ===========================================================================
 * All files are UTF-8, kanji written raw (never \u-escaped), and have NO
 * trailing newline.
 *
 * playouts_<mode>.json -- one line, `JSON.stringify(obj)`:
 *
 *   {"mode":<"intro"|"beginner"|"intermediate"|"advanced"|"betray">,
 *    "seed":<int>,
 *    "start":<the mode's start FEN>,
 *    "games":[ [ <ply>, ... ], ... ]}
 *
 *   <ply>, keys in EXACTLY this order:
 *     "fen_before" : FEN of the position before the move.
 *     "legal"      : SAN strings of every legal move in `fen_before`,
 *                    code-point-sorted, duplicates NOT removed.  Compared by
 *                    tests/engine/test_move_gen.py and test_cymovegen.py as
 *                    `sorted(m["san"] for m in generate_all_moves(fen))`.
 *     "verdict"    : object, keys in EXACTLY this order --
 *                    turn, inDraft, inCheck, isCheckmate, isStalemate,
 *                    isDraw, isGameOver.  Read by test_api.py and
 *                    test_cyengine.py.
 *     "chosen"     : the SAN drawn from `legal`.  Recorded BEFORE the move is
 *                    applied, because `Gungi.move()` mutates the returned
 *                    move's `.san` in place (it appends '#' on checkmate and
 *                    '=' on stalemate/fourfold/insufficient material).
 *     "fen_after"  : the move's `.after` FEN.
 *
 *   *** verdict and legal are computed on a FRESH `new Gungi(fen_before)`,
 *   NOT on the running game. ***  This is load-bearing, not incidental:
 *   `isDraw()` folds in `isFourfoldRepetition()`, which reads the live
 *   instance's move history.  The Python tests fresh-load every FEN and so
 *   always see an empty history (test_cyengine.py:130 spells this out).
 *   Recording from the running game would make `isDraw` disagree the first
 *   time a position repeats four times.
 *
 * fen_roundtrip.json -- one line, `JSON.stringify(arr)`:
 *
 *   [{"input":<FEN>,"canonical":<FEN>}, ...]
 *
 *   `canonical` is `encodeFEN(parseFEN(input))`, which is what
 *   tests/engine/test_fen.py and test_cyboard.py assert.  gungi.js does not
 *   re-export `parseFEN`/`encodeFEN` from its package root (see the export
 *   list at the top of its `gungi.ts`), but `new Gungi(fen).fen()` is that
 *   composition: the constructor parses and `.fen()` re-encodes.  Piece ids
 *   are not part of the FEN, so the id-assignment the constructor also does
 *   cannot affect the result.
 *
 *   Inputs, in order, deduplicated by FIRST occurrence:
 *     - the four mode start positions, in MODES order;
 *     - then, for every playout in generation order, each ply's `fen_before`
 *       followed by its `fen_after`.
 *   The `betray` mode contributes NOTHING here (see above).
 *
 * MANIFEST.json -- `JSON.stringify(obj, null, 2)`, no trailing newline; the
 *   committed schema plus `generator_sha256`, the sha256 of THIS file's
 *   bytes.  `counts` holds the ACTUAL game/ply counts of this run.  A betray
 *   run adds a `counts.betray` entry carrying, on top of the usual
 *   `games`/`plies`, `games_played`, `betray_legal_plies` and
 *   `chosen_betray`; the last two are counted independently and are equal by
 *   construction, so a disagreement is a real bug in the bias.
 *
 * ===========================================================================
 * Usage:  node oracle.mjs --seed 20260812 --out <dir>
 * See README.md.  There is no package-lock; `npm ls gungi.js` must say
 * 3.4.11 before the output of this script is trusted.
 */

import { createHash } from 'node:crypto';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import {
	ADVANCED_POSITION,
	BEGINNER_POSITION,
	Gungi,
	INTERMEDIATE_POSITION,
	INTRO_POSITION,
} from 'gungi.js';

/** Absolute path to this file -- hashed into MANIFEST.generator_sha256. */
const SELF = fileURLToPath(import.meta.url);

const require = createRequire(import.meta.url);

/** The version this corpus is defined against.  Nothing else is supported. */
const EXPECTED_GUNGI_JS_VERSION = '3.4.11';

/** Path recorded in MANIFEST.generated_by -- repo-relative, kept literal. */
const GENERATED_BY = 'scripts/oracle/oracle.mjs';

const DEFAULT_SEED = 20260812;

/**
 * Modes, in the order they are visited.  `games` is the default number of
 * playouts per mode; it matches the shape of the committed corpus (5/5/3/3).
 * The PLY counts are seed-dependent and are never hard-coded -- they are
 * measured and written to the MANIFEST.
 */
const MODES = [
	{ name: 'intro', start: INTRO_POSITION, games: 5 },
	{ name: 'beginner', start: BEGINNER_POSITION, games: 5 },
	{ name: 'intermediate', start: INTERMEDIATE_POSITION, games: 3 },
	{ name: 'advanced', start: ADVANCED_POSITION, games: 3 },
];

/**
 * The appended betray mode (KOM-101 G2).  Advanced rules and the advanced
 * start position -- it differs from `advanced` only in how the ply is drawn
 * and which plies are emitted, both of which live in `playBetrayGame`.
 *
 * It is NOT a member of `MODES`: the four entries there are the frozen
 * draw-order prefix, and everything that iterates `MODES` (the roundtrip
 * input set, the unfiltered emission loop) must keep seeing exactly those
 * four.  `games: 0` is what makes the default corpus unchanged.
 */
const BETRAY_MODE = { name: 'betray', start: ADVANCED_POSITION, games: 0 };

/** The SAN marker gungi.js writes for a betrayal: `返` + the converted types. */
const BETRAY_MARKER = '返';

/**
 * Safety valve only.  Random play nearly always terminates by marshal
 * capture well inside this; the committed corpus averaged 141-223 plies per
 * game.  A game that hits the cap is reported on stderr, because a capped
 * game is a truncated game and the reader should know.
 */
const DEFAULT_MAX_PLIES = 512;

// ---------------------------------------------------------------------------
// PRNG
// ---------------------------------------------------------------------------

/**
 * mulberry32.  Returns a function producing floats in [0, 1).
 *
 * Reference implementation (Tommy Ettinger, public domain), transcribed
 * without modification.  `a` is held as a uint32 via `>>> 0` after every
 * update so the state never leaves the exactly-representable integer range.
 */
function mulberry32(seed) {
	let a = seed >>> 0;
	return function next() {
		a = (a + 0x6d2b79f5) >>> 0;
		let t = a;
		t = Math.imul(t ^ (t >>> 15), t | 1);
		t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};
}

// ---------------------------------------------------------------------------
// Sorting
// ---------------------------------------------------------------------------

/**
 * Sort SAN strings by Unicode CODE POINT, ascending -- the ordering Python's
 * `sorted()` uses, which is what the consuming tests compare against.
 *
 * JavaScript's default `Array#sort()` compares UTF-16 code UNITS instead.
 * The two agree for every character gungi.js actually emits (CJK ideographs
 * plus ASCII digits, parentheses and hyphens are all BMP), but they diverge
 * for anything astral, so this comparator is explicit rather than relying on
 * that coincidence holding.  Returns a new array; does not mutate `sans`.
 */
function sortSan(sans) {
	return [...sans].sort((a, b) => {
		const ca = [...a];
		const cb = [...b];
		const n = Math.min(ca.length, cb.length);
		for (let i = 0; i < n; i++) {
			const pa = ca[i].codePointAt(0);
			const pb = cb[i].codePointAt(0);
			if (pa !== pb) return pa < pb ? -1 : 1;
		}
		return ca.length - cb.length;
	});
}

// ---------------------------------------------------------------------------
// gungi.js adapters
// ---------------------------------------------------------------------------

/**
 * `encodeFEN(parseFEN(fen))`, expressed with the package's public surface.
 * See the OUTPUT FORMAT note above for why this composition is the honest
 * one.
 */
function canonicalizeFen(fen) {
	return new Gungi(fen).fen();
}

/**
 * The verdict block for `probe`, with keys emitted in the committed order.
 * Object literal key order is JSON.stringify's output order in ES2015+ for
 * string keys, so writing them in order here is what pins the byte layout.
 */
function verdictOf(probe) {
	return {
		turn: probe.turn(),
		inDraft: probe.inDraft(),
		inCheck: probe.inCheck(),
		isCheckmate: probe.isCheckmate(),
		isStalemate: probe.isStalemate(),
		isDraw: probe.isDraw(),
		isGameOver: probe.isGameOver(),
	};
}

// ---------------------------------------------------------------------------
// Playouts
// ---------------------------------------------------------------------------

/**
 * Play one seeded random game from `start` and return its ply records.
 *
 * Two Gungi instances are in play per ply and the split matters:
 *
 *   `game`  -- the live instance, carrying move history.  It advances the
 *              playout and decides termination, so fourfold repetition can
 *              end a game exactly as gungi.js would end it.
 *   `probe` -- a fresh `new Gungi(fenBefore)` with empty history.  EVERY
 *              recorded field comes from it, matching how the Python tests
 *              fresh-load each FEN.
 *
 * `probe.moves()` and `game.moves()` can only disagree when the live
 * instance is already game-over (`moves()` returns [] then), and that case
 * is caught by the `game.isGameOver()` guard before the probe is built -- so
 * a SAN drawn from the probe is always legal on the live game.
 *
 * @param onFen called with every fen_before and fen_after, in order, for the
 *              fen_roundtrip input set.
 */
function playGame(start, rng, maxPlies, onFen) {
	const game = new Gungi(start);
	const plies = [];

	while (plies.length < maxPlies) {
		if (game.isGameOver()) break;

		const fenBefore = game.fen();
		const probe = new Gungi(fenBefore);
		const legal = sortSan(probe.moves());
		if (legal.length === 0) break;

		const verdict = verdictOf(probe);

		// The single stream draw for this ply (contract rule 5).
		const chosen = legal[Math.floor(rng() * legal.length)];

		// Read `chosen` out of `legal` BEFORE moving: `move()` appends '#'/'='
		// to the SAN of the move object it returns.  `.after` is untouched.
		const applied = game.move(chosen);
		const fenAfter = applied.after;

		onFen(fenBefore);
		onFen(fenAfter);

		plies.push({
			fen_before: fenBefore,
			legal,
			verdict,
			chosen,
			fen_after: fenAfter,
		});
	}

	return plies;
}

/**
 * Play one seeded betray-seeking game and return only its betray-legal plies.
 *
 * Identical to `playGame` -- same two-instance split, same probe-sourced
 * fields, same five-key ply schema, same `--max-plies` cap on plies PLAYED --
 * except for the three deliberate differences documented in the header's
 * `betray` section:
 *
 *   - whenever `legal` holds one or more 返 SANs, the draw domain is that
 *     subset, so a betray-legal ply is always a betray-CHOSEN ply;
 *   - one draw is consumed per played ply, recorded or not;
 *   - only the betray-legal plies are returned.
 *
 * The `legal` field of a returned ply is still the FULL code-point-sorted
 * list for that position, not the 返 subset -- that field is what the move
 * generator is compared against, and narrowing it would turn the strongest
 * assertion in the corpus into a weaker one.
 *
 * @returns {{plies: object[], played: number, legalPlies: number,
 *            chosenBetray: number}}
 *          `played` counts every ply applied to the board (the number the
 *          --max-plies cap is measured against); `legalPlies` and
 *          `chosenBetray` are counted independently so the invariant that
 *          they are equal stays checkable rather than assumed.
 */
function playBetrayGame(start, rng, maxPlies) {
	const game = new Gungi(start);
	const plies = [];
	let played = 0;
	let legalPlies = 0;
	let chosenBetray = 0;

	while (played < maxPlies) {
		if (game.isGameOver()) break;

		const fenBefore = game.fen();
		const probe = new Gungi(fenBefore);
		const legal = sortSan(probe.moves());
		if (legal.length === 0) break;

		const verdict = verdictOf(probe);

		const betrayals = legal.filter((san) => san.includes(BETRAY_MARKER));
		const isBetrayLegal = betrayals.length > 0;
		if (isBetrayLegal) legalPlies++;

		// The single stream draw for this PLAYED ply, over the 返 subset when
		// there is one and over the whole list otherwise.
		const domain = isBetrayLegal ? betrayals : legal;
		const chosen = domain[Math.floor(rng() * domain.length)];

		// As in `playGame`: read `chosen` before moving, because `move()`
		// mutates the returned move's `.san` in place.
		const applied = game.move(chosen);
		const fenAfter = applied.after;
		played++;

		if (!isBetrayLegal) continue;

		if (chosen.includes(BETRAY_MARKER)) chosenBetray++;

		plies.push({
			fen_before: fenBefore,
			legal,
			verdict,
			chosen,
			fen_after: fenAfter,
		});
	}

	return { plies, played, legalPlies, chosenBetray };
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

const USAGE = `usage: node oracle.mjs [--seed N] --out DIR [--max-plies N] [--games MODE=N,...]

  --seed N        PRNG seed (default ${DEFAULT_SEED})
  --out DIR       output directory; created if absent (required)
  --max-plies N   per-game safety cap (default ${DEFAULT_MAX_PLIES})
  --games SPEC    override games per mode, e.g. intro=5,advanced=3
                  modes: intro, beginner, intermediate, advanced, betray
  -h, --help      this message

Emits playouts_{intro,beginner,intermediate,advanced}.json,
fen_roundtrip.json and MANIFEST.json into DIR.

The betray mode defaults to 0 games and is drawn last, so the files above are
unaffected by it.  --games betray=N additionally emits playouts_betray.json:
N advanced-rules games that pick a 返 (betray) move whenever one is legal,
keeping ONLY the plies where one was legal.  Its plies are self-contained and
must not be chained.`;

function parseArgs(argv) {
	const opts = {
		seed: DEFAULT_SEED,
		out: null,
		maxPlies: DEFAULT_MAX_PLIES,
		// `betray` is last here too, so `--games` accepts it and its default
		// of 0 games keeps the default run identical to the pre-G2 one.
		games: new Map([...MODES, BETRAY_MODE].map((m) => [m.name, m.games])),
	};

	for (let i = 0; i < argv.length; i++) {
		const arg = argv[i];
		const need = () => {
			const v = argv[++i];
			if (v === undefined) throw new Error(`${arg} needs a value`);
			return v;
		};
		switch (arg) {
			case '-h':
			case '--help':
				console.log(USAGE);
				process.exit(0);
				break;
			case '--seed':
				opts.seed = Number.parseInt(need(), 10);
				break;
			case '--out':
				opts.out = need();
				break;
			case '--max-plies':
				opts.maxPlies = Number.parseInt(need(), 10);
				break;
			case '--games':
				for (const pair of need().split(',')) {
					const [name, n] = pair.split('=');
					if (!opts.games.has(name)) throw new Error(`unknown mode: ${name}`);
					opts.games.set(name, Number.parseInt(n, 10));
				}
				break;
			default:
				throw new Error(`unknown argument: ${arg}`);
		}
	}

	if (opts.out === null) throw new Error('--out DIR is required');
	if (!Number.isInteger(opts.seed)) throw new Error('--seed must be an integer');
	if (!Number.isInteger(opts.maxPlies) || opts.maxPlies < 1) {
		throw new Error('--max-plies must be a positive integer');
	}
	for (const [name, n] of opts.games) {
		if (!Number.isInteger(n) || n < 0) {
			throw new Error(`--games ${name}: must be a non-negative integer`);
		}
	}
	return opts;
}

/**
 * The installed gungi.js version, or null if it cannot be read.
 *
 * `createRequire(...).resolve` works on every Node that can run this script;
 * `import.meta.resolve` only became sync-and-unflagged in Node 20.6, so it is
 * the fallback rather than the primary.  Some packages omit `package.json`
 * from their `exports` map, which makes both throw -- hence the null case.
 */
function installedGungiVersion() {
	const read = (path) => JSON.parse(readFileSync(path, 'utf8')).version;
	try {
		return read(require.resolve('gungi.js/package.json'));
	} catch {
		/* fall through */
	}
	try {
		return read(fileURLToPath(import.meta.resolve('gungi.js/package.json')));
	} catch {
		return null;
	}
}

/** Write `text` as UTF-8 with no trailing newline (the committed layout). */
function writeExact(path, text) {
	writeFileSync(path, text, 'utf8');
}

function main() {
	const opts = parseArgs(process.argv.slice(2));
	const outDir = resolve(opts.out);
	mkdirSync(outDir, { recursive: true });

	const version = installedGungiVersion();
	if (version === null) {
		console.error(
			'warning: could not resolve gungi.js/package.json; recording the ' +
				`expected version ${EXPECTED_GUNGI_JS_VERSION} unverified`
		);
	} else if (version !== EXPECTED_GUNGI_JS_VERSION) {
		console.error(
			`ERROR: gungi.js ${version} is installed but this corpus is defined ` +
				`against ${EXPECTED_GUNGI_JS_VERSION}. Fix the pin (see README.md) ` +
				'and re-run; refusing to emit a mislabelled corpus.'
		);
		process.exit(1);
	}

	const rng = mulberry32(opts.seed);

	// fen_roundtrip inputs, deduplicated by first occurrence.  A Map (or Set)
	// preserves insertion order for string keys, which is the ordering the
	// format note above promises.
	const fenInputs = new Set(MODES.map((m) => m.start));
	const onFen = (fen) => fenInputs.add(fen);

	const counts = {};
	// Counted rather than hard-coded: the betray emission is optional.
	let filesWritten = 0;
	for (const mode of MODES) {
		const games = [];
		const nGames = opts.games.get(mode.name);
		for (let g = 0; g < nGames; g++) {
			const plies = playGame(mode.start, rng, opts.maxPlies, onFen);
			if (plies.length === opts.maxPlies) {
				console.error(
					`warning: ${mode.name} game ${g} hit --max-plies ` +
						`${opts.maxPlies} and is TRUNCATED, not finished`
				);
			}
			games.push(plies);
		}

		const payload = {
			mode: mode.name,
			seed: opts.seed,
			start: mode.start,
			games,
		};
		writeExact(join(outDir, `playouts_${mode.name}.json`), JSON.stringify(payload));
		filesWritten++;

		counts[mode.name] = {
			games: games.length,
			plies: games.reduce((n, plies) => n + plies.length, 0),
		};
		console.error(
			`${mode.name}: ${counts[mode.name].games} games, ` +
				`${counts[mode.name].plies} plies`
		);
	}

	// --- the appended betray emission (draw-order rule 2: strictly last) ---
	//
	// Skipped entirely at 0 games: no draws, no file, no MANIFEST entry, so
	// the default run is exactly the pre-G2 one apart from generator_sha256.
	const nBetrayGames = opts.games.get(BETRAY_MODE.name);
	if (nBetrayGames > 0) {
		const games = [];
		let played = 0;
		let legalPlies = 0;
		let chosenBetray = 0;
		let truncated = 0;

		for (let g = 0; g < nBetrayGames; g++) {
			const res = playBetrayGame(BETRAY_MODE.start, rng, opts.maxPlies);
			played += res.played;
			legalPlies += res.legalPlies;
			chosenBetray += res.chosenBetray;
			if (res.played === opts.maxPlies) truncated++;
			// Empty games are omitted: a game that never saw a betrayal has
			// nothing self-contained to contribute.
			if (res.plies.length > 0) games.push(res.plies);
		}

		const payload = {
			mode: BETRAY_MODE.name,
			seed: opts.seed,
			start: BETRAY_MODE.start,
			games,
		};
		writeExact(
			join(outDir, `playouts_${BETRAY_MODE.name}.json`),
			JSON.stringify(payload)
		);
		filesWritten++;

		counts[BETRAY_MODE.name] = {
			games: games.length,
			plies: games.reduce((n, plies) => n + plies.length, 0),
			games_played: nBetrayGames,
			betray_legal_plies: legalPlies,
			chosen_betray: chosenBetray,
		};
		console.error(
			`betray: ${nBetrayGames} games, ${legalPlies} betray-legal plies, ` +
				`${chosenBetray} chosen`
		);
		console.error(
			`betray: ${games.length} of ${nBetrayGames} games contributed a ply; ` +
				`${played} plies played, ${counts[BETRAY_MODE.name].plies} emitted`
		);
		if (legalPlies !== chosenBetray) {
			console.error(
				`ERROR: betray mode chose a betrayal on only ${chosenBetray} of ` +
					`${legalPlies} betray-legal plies; the biased draw is broken`
			);
			process.exit(1);
		}
		if (truncated > 0) {
			console.error(
				`warning: ${truncated} betray games hit --max-plies ` +
					`${opts.maxPlies} and stopped early`
			);
		}
	}

	const roundtrip = [...fenInputs].map((input) => ({
		input,
		canonical: canonicalizeFen(input),
	}));
	writeExact(join(outDir, 'fen_roundtrip.json'), JSON.stringify(roundtrip));
	filesWritten++;
	console.error(`fen_roundtrip: ${roundtrip.length} entries`);

	const manifest = {
		gungi_js_version: version ?? EXPECTED_GUNGI_JS_VERSION,
		seed: opts.seed,
		counts,
		fen_roundtrip: roundtrip.length,
		generated_by: GENERATED_BY,
		generator_sha256: createHash('sha256')
			.update(readFileSync(SELF))
			.digest('hex'),
	};
	writeExact(join(outDir, 'MANIFEST.json'), JSON.stringify(manifest, null, 2));
	filesWritten++;

	console.error(`wrote ${filesWritten} files to ${outDir}`);
}

try {
	main();
} catch (err) {
	console.error(`oracle.mjs: ${err.message}`);
	process.exit(1);
}
