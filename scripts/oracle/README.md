# `scripts/oracle` — the gungi.js golden generator

`oracle.mjs` regenerates the corpus under `tests/engine/goldens/`:

```
playouts_intro.json  playouts_beginner.json  playouts_intermediate.json
playouts_advanced.json  playouts_betray.json  fen_roundtrip.json
MANIFEST.json
```

`playouts_betray.json` is the KOM-101 G2 betrayal corpus and is **opt-in**:
it is only written when `--games betray=N` asks for it. See
"The `betray` mode" below.

Those files are the oracle the ported v1 engine suite is checked against —
`tests/engine/test_move_gen.py`, `test_cymovegen.py`, `test_api.py`,
`test_cyengine.py`, `test_cygame.py`, `test_cyboard.py`, `test_fen.py` and
`test_action_encoding.py` all read them through
`tests/engine/conftest.py::load_golden`, which *skips* when the corpus is
absent. A missing corpus is therefore silent; a wrong corpus is not.

## Status: this is a reimplementation

`MANIFEST.json` names its generator as `scripts/oracle/oracle.mjs`, but that
script was never committed. `oracle.mjs` here is a **new** generator written
against the committed corpus's format. It is not the original.

**Byte-identity with the committed corpus is out of scope and will not
happen.** The original PRNG and its draw order are gone, so a different
sequence of moves gets played and the game/ply counts come out different.
That is fine and expected: the counts are seed-dependent, so `oracle.mjs`
*measures* them and writes the real numbers into the MANIFEST it emits,
rather than asserting the old ones. Do not spend time trying to recover the
old stream — that avenue is closed.

What must hold is that a regenerated corpus is *equally valid*: same file
names, same JSON shape, same field order, same semantics per field — and the
Python engine suite passes green against it.

## Prerequisites

There is no Node in this repo's dev shell and there must not be (see
"Never wire this into CI" below). Get one ad hoc:

```sh
nix shell nixpkgs#nodejs
```

Node 18 or newer. Network access is required for `npm install`.

## Regenerating

```sh
cd scripts/oracle
npm install
npm ls gungi.js          # MUST print gungi.js@3.4.11 — see below
node oracle.mjs --seed 20260812 --out /tmp/goldens-new
```

`package.json` pins `"gungi.js": "3.4.11"` **exactly** — no caret, no tilde.
There is deliberately no `package-lock.json` (it could not be generated
offline when this was written), so `npm install` is free to resolve
gungi.js's own transitive dependencies however it likes. The pin only binds
gungi.js itself, which is why `npm ls gungi.js` is a required step and not a
suggestion. `oracle.mjs` re-checks it at runtime too: it reads the installed
`gungi.js/package.json` and **exits non-zero** rather than emit a corpus
labelled 3.4.11 that was produced by some other version.

If you do generate a lockfile later, commit it here — it makes the
`npm ls` step a formality instead of the actual guard.

Options:

| flag | default | meaning |
| --- | --- | --- |
| `--seed N` | `20260812` | PRNG seed. `20260812` is the seed recorded in the committed MANIFEST; keep it unless you mean to fork the corpus. |
| `--out DIR` | *(required)* | Output directory, created if absent. |
| `--max-plies N` | `512` | Per-game truncation cap. Purely a safety valve; a game that hits it is reported on stderr as TRUNCATED. |
| `--games SPEC` | `intro=5,beginner=5,intermediate=3,advanced=3,betray=0` | Games per mode, e.g. `--games advanced=10`. |

Progress and counts go to stderr; nothing goes to stdout except `--help`.

## The `betray` mode

Betrayal (`返`) is the one rule family the four random-play modes do not
cover. An advisor probe over 90 random advanced/intermediate games — 17,522
plies — found **33 plies where a 返 move was legal and 0 where random play
chose one**. Uniform play cannot produce a consumable betrayal corpus at any
sane corpus size, so `oracle.mjs` grows a fifth mode instead.

`betray` plays **advanced** rules from `ADVANCED_POSITION` — the same start
and mode as `advanced` — and differs from it in exactly three ways:

* **The draw is biased, deliberately.** Whenever the code-point-sorted legal
  list contains one or more `返` SANs, the ply's single draw is taken over
  *that subset* rather than the whole list. So every betray-legal ply is also
  a betray-**chosen** ply, and the two MANIFEST counts below are equal by
  construction. Everything else — start position, `--max-plies` cap, the
  fresh-probe sourcing of every recorded field, the five-key ply schema — is
  `advanced`'s.
* **One draw per *played* ply**, recorded or not (the four frozen modes
  consume one per *recorded* ply; there, the two are the same thing).
  Terminal positions still consume nothing.
* **The emission is filtered.** Only plies whose `legal` list contains a `返`
  SAN are written; a game that contributes none is omitted entirely. The
  `legal` field of a kept ply is still the FULL sorted list, not the `返`
  subset — that field is what the move generator is compared against.

Consequently **a `games` entry in `playouts_betray.json` is not a contiguous
playout**: its plies are independent positions and `fen_after` of ply *i* is
generally not `fen_before` of ply *i+1*. Every ply is self-contained, and
every consumer in `tests/engine/` reads only a ply's own five fields, so this
costs nothing — but a future test that chains plies must restrict itself to
the four full-game files. `test_move_gen.py` and `test_action_encoding.py`
spell that out in the comment above their `FULL_GAME_FILES` list.

### Why this does not break the determinism contract

`betray` defaults to **0 games** and is drawn **strictly last**, after every
draw of intro, beginner, intermediate and advanced. At 0 games it takes no
draws, writes no file and adds no MANIFEST entry, so the default invocation
is the pre-G2 one. At any count it takes its draws only after the existing
modes are finished, so the four committed playout files stay bit-identical
for a given seed. Betray plies are also deliberately **not** fed into the
`fen_roundtrip.json` input set, which keeps that file byte-identical too.

The bias is why this is a separate mode rather than a change to `advanced`:
biasing an existing mode's per-ply draw would silently change what its seed
means. This mode's seed means something different on purpose, and it is the
only mode where the draw is not uniform over the legal list.

### Regenerating with betrayals

```sh
node oracle.mjs --seed 20260812 --games betray=600 --out tests/engine/goldens
```

Writing straight into `tests/engine/goldens/` is safe **here specifically**:
by construction the four `playouts_*.json` files and `fen_roundtrip.json`
come out byte-identical to the committed ones, so the only diff is the new
`playouts_betray.json` plus `MANIFEST.json` (whose `generator_sha256` moves
whenever `oracle.mjs` is edited at all). Confirm that with `git status` /
`git diff --stat` before running the suite; if any of the five is dirty, the
contract has been broken and the run must be thrown away, not committed.

Stderr reports:

```
betray: 600 games, 214 betray-legal plies, 214 chosen
betray: 118 of 600 games contributed a ply; 84302 plies played, 214 emitted
```

(illustrative numbers). The generator **exits non-zero** if the betray-legal
and chosen counts ever disagree — that would mean the biased draw is broken,
not that the corpus is merely thin.

`MANIFEST.json` gains a `counts.betray` entry:

```json
"betray": {
  "games": 118,
  "plies": 214,
  "games_played": 600,
  "betray_legal_plies": 214,
  "chosen_betray": 214
}
```

`games`/`plies` are the *emitted* counts (what is actually in the file);
`games_played` is what `--games betray=` asked for. `betray_legal_plies` and
`chosen_betray` are counted independently so the equality above stays a
checked invariant rather than an assumption.

## Validating a regenerated corpus

Never overwrite `tests/engine/goldens/` in place and then run the suite — a
failure at that point leaves you with no known-good corpus to diff against.
Use a scratch worktree:

```sh
git worktree add /tmp/komugi-goldens HEAD
cp /tmp/goldens-new/*.json /tmp/komugi-goldens/tests/engine/goldens/
cd /tmp/komugi-goldens
python -m pytest -q tests/engine
```

What green means, test by test:

* `test_move_gen.py::test_legal_moves_match_golden` and its Cython twin —
  every `legal` list equals `sorted(m["san"] for m in generate_all_moves(fen_before))`.
  This is the move generator agreeing with gungi.js on every recorded
  position, and it is the single most load-bearing assertion in the corpus.
* `test_api.py::test_verdict_parity` / `test_cyengine.py::test_verdict_matches_golden` —
  every `verdict` block matches a **freshly loaded** `Gungi(fen_before)`.
* `test_api.py::test_replay_parity`, `test_move_gen.py::test_make_move_after_fen`,
  `test_action_encoding.py` — applying `chosen` to `fen_before` yields
  `fen_after`.
* `test_fen.py::test_fen_roundtrip_golden` / `test_cyboard.py` —
  `encode_fen(parse_fen(input)) == canonical` for all ~2.6k entries.

Only after that passes should the files move into `tests/engine/goldens/`,
and the ticket should record the new MANIFEST counts and `generator_sha256`.

Sanity checks that are cheap and worth doing before the suite:

```sh
python -m json.tool /tmp/goldens-new/MANIFEST.json     # counts look sane?
tail -c 1 /tmp/goldens-new/playouts_intro.json | xxd   # NOT 0a — no trailing newline
```

## Never wire this into CI

Per the KOM-024 amended rule, this npm pin **does not enter the CI gate and
does not enter `flake.nix`**. `ci/check.sh` must never invoke `npm`, `node`
or this directory, and no Node toolchain belongs in the flake's dev shell.

The reason is that the corpus is a *committed artifact*, not a build product.
The gate's job is to check the engine against the corpus in the tree; it is
not to re-derive the corpus, which would make every CI run depend on the npm
registry and on gungi.js's transitive dependency resolution. Regeneration is
a deliberate, human-initiated act that produces a reviewable diff.

Consequently `scripts/oracle/node_modules/` must never be committed.

## Determinism contract

The full contract lives in the header comment of `oracle.mjs`; it is
normative and this is a summary. Same seed plus gungi.js 3.4.11 must give
byte-identical output on any Node version and platform.

* **PRNG** — mulberry32, transcribed from the public-domain reference. It
  uses only `Math.imul`, `>>>`, `^`, `+` and one division by `2**32`, all of
  which ECMA-262 specifies exactly on uint32 values. `Math.random` is never
  called and must never be.
* **One stream per run**, seeded `seed >>> 0`, never reseeded, never forked.
* **Draw order** — modes in the order intro, beginner, intermediate,
  advanced, then `betray` strictly last; games ascending within a mode;
  plies ascending within a game; **exactly one draw per recorded ply**,
  `idx = Math.floor(next() * legal.length)`. Terminal (unrecorded) positions
  consume nothing. Inserting or removing a single draw anywhere renumbers
  every ply after it. (`betray` is the one exception to both halves of the
  draw rule — one draw per *played* ply, over the `返` subset when there is
  one — which is exactly why it is appended after everything else and
  defaults to 0 games. See "The `betray` mode" above.)
* **The draw domain is the code-point-sorted SAN list**, not gungi.js's
  internal generation order. Generation order is an implementation detail
  that a gungi.js patch release could reshuffle; sorted order is canonical.
  `sortSan` compares by Unicode code point, matching Python's `sorted()`
  rather than JS's default UTF-16 code-unit comparison.
* **Recorded fields come from a fresh `new Gungi(fen_before)`, never from
  the running game.** `isDraw()` folds in `isFourfoldRepetition()`, which
  reads live move history; the Python tests fresh-load every FEN and so
  always see an empty history. The live instance is used only to advance the
  playout and to decide termination.
* **`chosen` is read out of `legal` before the move is applied**, because
  `Gungi.move()` mutates the returned move's `.san` in place — appending
  `#` on checkmate and `=` on stalemate / fourfold / insufficient material.

## Output format

Every file is UTF-8, writes kanji raw (never `\uXXXX`-escaped), and has **no
trailing newline**.

`playouts_<mode>.json` — one line, `JSON.stringify(obj)`:

```json
{"mode":"intro","seed":20260812,"start":"<start FEN>","games":[[<ply>, ...], ...]}
```

`<ply>` has exactly five keys in exactly this order:

| key | value |
| --- | --- |
| `fen_before` | FEN before the move |
| `legal` | SAN of every legal move in `fen_before`, code-point-sorted, duplicates **not** removed |
| `verdict` | `{turn, inDraft, inCheck, isCheckmate, isStalemate, isDraw, isGameOver}`, in that order |
| `chosen` | the SAN drawn from `legal` |
| `fen_after` | the applied move's `.after` FEN |

`playouts_betray.json` has the same `mode`/`seed`/`start`/`games` envelope
(`"mode":"betray"`, `start` = the advanced start FEN) and the same five-key
ply schema. What differs is only which plies are present: see
"The `betray` mode".

`fen_roundtrip.json` — one line, `JSON.stringify(arr)`:

```json
[{"input":"<FEN>","canonical":"<FEN>"}, ...]
```

`canonical` is `encodeFEN(parseFEN(input))`. gungi.js does not re-export
`parseFEN`/`encodeFEN` from its package root, so `oracle.mjs` expresses that
composition as `new Gungi(fen).fen()` — the constructor parses, `.fen()`
re-encodes, and the piece ids the constructor also assigns are not part of a
FEN and so cannot affect the result. Inputs are, in order and deduplicated
by first occurrence: the four mode start positions, then each ply's
`fen_before` followed by its `fen_after`. Betray plies contribute nothing to
this set, so `fen_roundtrip.json` does not move when `--games betray=` does.

`MANIFEST.json` — `JSON.stringify(obj, null, 2)`, no trailing newline. The
committed schema plus one new field:

```json
{
  "gungi_js_version": "3.4.11",
  "seed": 20260812,
  "counts": { "intro": { "games": 5, "plies": 707 }, "...": {} },
  "fen_roundtrip": 2666,
  "generated_by": "scripts/oracle/oracle.mjs",
  "generator_sha256": "<sha256 of oracle.mjs>"
}
```

`counts` and `fen_roundtrip` are the **measured** numbers for the run, not
the committed ones (the values above are the committed corpus's, shown for
shape only). `generator_sha256` hashes `oracle.mjs`'s own bytes, so a corpus
can always be traced to the exact generator text that produced it — edit the
script and the hash moves, which is the intended signal.

## Extending it

Any extension must keep the contract above intact:

* **Betray-bearing games — done (KOM-101 G2).** Implemented as the `betray`
  mode documented above, not as "more or longer playouts": the measured rate
  of *chosen* betrayals under uniform random play is zero, so no amount of
  extra unbiased play produces them. The determinism concern the earlier plan
  raised is answered structurally rather than by avoiding the bias — the
  biased draw lives in a *new* mode that draws last and defaults to 0 games,
  so no existing mode's seed changes meaning, and the filter is applied to
  the emission rather than to the draw.
* **G2 in general.** New output files are added as new emissions from the
  same single stream, appended *after* all existing draws, so the existing
  playouts stay bit-identical for a given seed. A new emission that needs a
  different draw rule gets its own mode; it never edits an existing one.

If a change must reorder draws, say so explicitly in the ticket and treat
the whole corpus as regenerated, not amended.
