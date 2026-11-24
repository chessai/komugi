"""The bridge against a real process that really misbehaves.

Every test below spawns an actual subprocess, speaks the actual wire to it, and
plays actual games through :func:`~komugi.eval.v1bridge.play_bridged_match`. The
far end is the **fake v1** at the top of this file: a python program that speaks
§D1's protocol and can be told, per behaviour, to crash, hang, propose an
illegal move, disagree about the rules, emit garbage, breach the budget, carry a
tree across a move, or simply play.

Two things about the stub are deliberate and load-bearing:

* **It implements the wire independently.** It builds and parses its JSON by
  hand rather than importing ``komugi.eval.v1bridge``'s helpers. A stub that
  used the module's own encoder would prove only that the module agrees with
  itself; this one proves the protocol is implementable from §D1's description,
  which is the thing the real v1 adapter will have to do.
* **It keeps its own board**, synchronised only by the ``observe`` messages the
  harness sends -- exactly as §D2.2 specifies for the real adapter, which
  "constructs a v1 ``Game`` at ``reset``, applies every move record it is told
  about via ``observe``". A full clean game therefore proves the move-record
  wire form keeps two independent engine states in step for the length of a
  game, which is the assumption the whole subprocess design rests on. It uses
  Komugi's engine to do that because it is a *stand-in* for v1, not v1: nothing
  here imports, spawns or references a v1 checkout.

What each §D6 mode's test asserts, and what would fire if the bridge mis-scored
it, is the non-vacuity index at the bottom of this file.
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# the fake v1
#
# This half runs in the CHILD process. It is above the heavy imports, and the
# __main__ guard below exits before them, so a spawned stub pays for the engine
# and nothing else -- not torch, not the arena, not this file's test half.
# ---------------------------------------------------------------------------

CORRECT: str = "correct"
CRASH: str = "crash"
HANG: str = "hang"
GARBAGE: str = "garbage"
ILLEGAL: str = "illegal"
MALFORMED: str = "malformed"
DISAGREE: str = "disagree"
BREACH: str = "breach"
TREE_REUSE: str = "tree_reuse"
ZERO_EVALS: str = "zero_evals"
DECLINES_BREACH: str = "declines_breach"
DECLINES_NO_MOVE: str = "declines_no_move"
BAD_MODE: str = "bad_mode"
BAD_VERSION: str = "bad_version"

_STUB_PROTOCOL_VERSION: int = 1
"""What the stub claims to speak. Written as a literal on purpose: importing the
harness's constant would hide a version skew instead of exposing one."""


def _stub_write(payload: dict[str, Any]) -> None:
    """Write one response line, hand-rolled (see the module docstring)."""
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def _stub_wire_record(record: dict[str, Any]) -> dict[str, Any]:
    """Return §D2.3's wire form of one engine move record."""
    return {
        "from_r": int(record["from_r"]),
        "from_f": int(record["from_f"]),
        "to_r": int(record["to_r"]),
        "to_f": int(record["to_f"]),
        "to_tier": int(record["to_tier"]),
        "move_type": int(record["move_type"]),
        "piece_type_index": int(record["piece_type_index"]),
        "color": int(record["color"]),
        "captured": [
            [int(item[0]), int(item[1])] for item in (record["captured"] or ())
        ],
        "draft_finished": bool(record["draft_finished"]),
        "san": str(record.get("san", "")),
    }


def _stub_main(argv: list[str]) -> int:
    """Run the fake v1 until ``bye`` or until its behaviour ends the process.

    Args:
        argv: ``[behaviour, after_games, after_picks]``. The behaviour fires on
            the first pick of the game *after* ``after_games`` games and after
            ``after_picks`` picks within that game, so a failure can be aimed at
            a specific pair -- which is what lets the M8/M9 tests prove that a
            match-void discards the pairs that were already **completed**.

    Returns:
        The process exit status.
    """
    # Insurance for a child spawned without the repo's PYTHONPATH: tests/eval/X
    # -> repo root -> src. Harmless when the path is already there.
    src = Path(__file__).resolve().parents[2] / "src"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))

    from komugi.engine.action_encoding import (
        decode_cython_action,
        encode_cython_move,
        legal_cython_action_ids,
    )
    from komugi.engine.game import Game

    behaviour = argv[0] if argv else CORRECT
    after_games = int(argv[1]) if len(argv) > 1 else 0
    after_picks = int(argv[2]) if len(argv) > 2 else 0

    game = Game()
    games_seen = 0
    picks = 0

    def armed() -> bool:
        """Return whether this behaviour should fire on this pick."""
        return games_seen > after_games and picks > after_picks

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        message = json.loads(line)
        kind = message["type"]

        if kind == "hello":
            mode = str(message["gungi_mode"])
            _stub_write(
                {
                    "type": "hello",
                    "protocol_version": (
                        999 if behaviour == BAD_VERSION else _STUB_PROTOCOL_VERSION
                    ),
                    "checkpoint_digest": "sha256:fake-v1",
                    "action_size": int(Game.action_size()),
                    "gungi_mode": f"{mode}-not" if behaviour == BAD_MODE else mode,
                    "capabilities": {
                        "supports_no_move": behaviour != DECLINES_NO_MOVE,
                        "supports_budget_breach": behaviour != DECLINES_BREACH,
                        "memo_separable": True,
                        "memo_citation": "fake-v1: there is no memo",
                    },
                    "device": {
                        "torch_version": "none",
                        "numpy_version": "none",
                        "device": "cpu",
                        "cudnn_benchmark": False,
                        "deterministic_algorithms": True,
                    },
                }
            )
            continue

        if kind == "reset":
            game = Game()
            games_seen += 1
            picks = 0
            _stub_write({"type": "ok"})
            continue

        if kind == "observe":
            # §D2.2: the far side keeps its own board, synchronised by records.
            game.play_action(int(encode_cython_move(message["record"])))
            _stub_write({"type": "ok"})
            continue

        if kind == "probe":
            observation = game.observation()
            _stub_write(
                {
                    "type": "observation",
                    "shape": [int(dim) for dim in observation.shape],
                    "data": base64.b64encode(
                        observation.astype("float32").tobytes()
                    ).decode("ascii"),
                }
            )
            continue

        if kind == "bye":
            _stub_write({"type": "ok"})
            return 0

        if kind != "pick":
            _stub_write({"type": "ok"})
            continue

        picks += 1
        seed = int(message["seed"])
        win_state = [int(slot) for slot in game.win_state()]

        if armed():
            if behaviour == CRASH:
                sys.stderr.write("fake-v1: dying on purpose, mid-game\n")
                sys.stderr.flush()
                os._exit(9)
            if behaviour == HANG:
                time.sleep(600)
            if behaviour == GARBAGE:
                sys.stdout.write("this is not a protocol message at all\n")
                sys.stdout.flush()
                continue
            if behaviour == DISAGREE:
                _stub_write(
                    {
                        "type": "no_move",
                        "win_state": [0, 0, 1],
                        "reason": "fourfold repetition on the cells-only key",
                    }
                )
                continue
            if behaviour == BREACH:
                _stub_write(
                    {
                        "type": "budget_breach",
                        "evals_used": 258,
                        "sims_used": 256,
                        "tree_nodes_at_entry": 0,
                    }
                )
                continue

        ids = sorted(legal_cython_action_ids(game._board))
        if not ids:
            _stub_write(
                {"type": "no_move", "win_state": win_state, "reason": "no legal move"}
            )
            continue
        action = ids[seed % len(ids)]
        record = _stub_wire_record(decode_cython_action(action, game._board))
        evals = 1 if len(ids) == 1 else 1 + (seed % 256)
        sims = 0 if len(ids) == 1 else 256

        if armed():
            if behaviour == ILLEGAL:
                # Well-formed for the schema, impossible for the rules: a route
                # (move_type 0) has an origin square, and a record with none is
                # in no entry of generate_all_moves. This is M3, not M4.
                record = dict(record, move_type=0, from_r=-1, from_f=-1)
            elif behaviour == MALFORMED:
                record = dict(record)
                record["to_r"] = 42
            elif behaviour == TREE_REUSE:
                _stub_write(
                    {
                        "type": "move",
                        "record": record,
                        "evals_used": evals,
                        "sims_used": sims,
                        "tree_nodes_at_entry": 41,
                        "win_state": win_state,
                    }
                )
                continue
            elif behaviour == ZERO_EVALS:
                evals, sims = 0, 0

        _stub_write(
            {
                "type": "move",
                "record": record,
                "evals_used": evals,
                "sims_used": sims,
                "tree_nodes_at_entry": 0,
                "win_state": win_state,
            }
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(_stub_main(sys.argv[1:]))


# ---------------------------------------------------------------------------
# the harness half: everything below runs only under pytest
# ---------------------------------------------------------------------------

import pytest

pytest.importorskip("torch")

from komugi.engine.repetition import RepetitionRule
from komugi.eval.arena import ArenaConfig, game_seeds
from komugi.eval.pool_books import PoolBook
from komugi.eval.records import BLACK_WIN, DRAW, WHITE_WIN
from komugi.eval.seats import check_paired_books
from komugi.selfplay.adapter import GameState
from komugi.selfplay.records import move_seed
from komugi.eval.v1bridge import (
    ABORTED,
    COMPLETED,
    VOIDED,
    BridgedMatch,
    BridgeEntrant,
    BridgeVoid,
    HandshakeRefused,
    MoveRecord,
    legal_records,
    M1,
    M2,
    M3,
    M4,
    M5,
    M8,
    M9,
    M10,
    V1Config,
    VoidBucket,
    VoidScope,
    bridged_book_id,
    play_bridged_game,
    play_bridged_match,
    resolved_gungi_mode,
    spawn_v1,
    v1_entrant,
)

# §D3's epoch-book fixtures, shared with the pool_books nodes rather than
# re-typed here (the idiom this file is already on the other end of).
from test_pool_books import epoch_book

# KOM-236: every node here spawns the fake v1 as a real subprocess and speaks
# the wire to it, which the gate sandbox cannot run.
pytestmark = pytest.mark.sandbox_unsafe

DRAW_SCORE: float = 0.0
"""§D3.4 pre-registers ``draw_score = 0.0`` for every cross-family match."""

SHORT_DEADLINE_S: float = 3.0
"""M2's deadline, shortened for CI. The *policy* under test is "kill, void the
pair, never retry", which is independent of the pre-registered 60 s."""

STUB_PATH: str = str(Path(__file__).resolve())


def stub_config(
    behaviour: str = CORRECT,
    after_games: int = 0,
    after_picks: int = 0,
    *,
    deadline_s: float = SHORT_DEADLINE_S,
) -> V1Config:
    """Return the :class:`V1Config` that spawns this file as a fake v1.

    The argv is the caller's, which is exactly how §D1 keeps a v1 checkout out
    of this module: nothing in the bridge knows what it is starting.
    """
    return V1Config(
        argv=(sys.executable, STUB_PATH, behaviour, str(after_games), str(after_picks)),
        gungi_mode=resolved_gungi_mode(),
        checkpoint="fake-v1-0015",
        pick_deadline_s=deadline_s,
    )


def stub_entrant(
    player_id: str,
    behaviour: str = CORRECT,
    after_games: int = 0,
    after_picks: int = 0,
    *,
    deadline_s: float = SHORT_DEADLINE_S,
) -> BridgeEntrant:
    """Return a bridged entrant backed by a freshly spawned fake v1."""
    return v1_entrant(
        player_id, stub_config(behaviour, after_games, after_picks, deadline_s=deadline_s)
    )


def match_config(games: int = 2) -> ArenaConfig:
    """Return the cross-family match config, with §D4 amendment 2's book id."""
    return ArenaConfig(
        games=games,
        seed=0xC0FFEE,
        draw_score=DRAW_SCORE,
        book_id=bridged_book_id(resolved_gungi_mode()),
    )


def play(
    behaviour: str,
    after_games: int = 0,
    after_picks: int = 0,
    *,
    games: int = 2,
    deadline_s: float = SHORT_DEADLINE_S,
) -> BridgedMatch:
    """Play a match between a misbehaving fake v1 and a correct one."""
    return play_bridged_match(
        stub_entrant("v1-broken", behaviour, after_games, after_picks, deadline_s=deadline_s),
        stub_entrant("v1-correct"),
        match_config(games),
    )


def sole_void(match: BridgedMatch) -> BridgeVoid:
    """Return the match's single void, asserting that there is exactly one."""
    assert len(match.tally.voids) == 1, (
        f"expected exactly one void, got {[v.mode for v in match.tally.voids]}"
    )
    return match.tally.voids[0]


def assert_nothing_was_scored(match: BridgedMatch) -> None:
    """Assert the run produced no game record at all -- §D6's core rule.

    This is the assertion that fires if the bridge ever books a failure as a
    result: a crash, a hang, an illegal move or a rules divergence that reached
    ``MatchOutcome.records`` as a 0, a 0.5 or a 1 lands here, whatever else the
    test asserted.
    """
    if match.outcome is None:
        return
    assert match.outcome.records == (), (
        f"a §D6 failure was scored as a game: "
        f"{[(r.white, r.black, r.result) for r in match.outcome.records]}"
    )
    assert match.outcome.summary.games == 0


# ---------------------------------------------------------------------------
# the clean game: the protocol works, and §D4's pairing survives it
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def clean_match() -> BridgedMatch:
    """Play one clean pair between two fake v1s, once for the whole module.

    Module-scoped because it is the only fixture here that plays *complete*
    games -- up to the engine's 256-turn cap, three engine states in step the
    whole way -- and two tests read different properties of the same run.
    """
    return play_bridged_match(
        stub_entrant("v1-white"), stub_entrant("v1-black"), match_config(games=2)
    )


def test_a_clean_stub_match_produces_a_valid_paired_match_outcome(
    clean_match: BridgedMatch,
) -> None:
    """Two fake v1s play a whole pair, and the result is well-formed evidence.

    This is the test the whole protocol exists to make possible, and it proves
    four things at once that only a real spawned game can:

    * the move-record wire form (§D2.3) keeps **three** independent engine
      states -- the referee's and the two adapters' -- in step for a whole game,
      through every move type the position produced. A single dropped or
      narrowed field desynchronises a board, and the very next ply proposes a
      move that is illegal in the referee's position;
    * ``observe`` reaches both movers, including the one that produced the move
      (§D1: "Told every action actually played, including the opponent's");
    * §D4's pairing invariant is untouched by a cross-family match -- the pair
      is two games, mirror seats, one ``book_id`` -- which is
      ``check_paired_books`` returning empty;
    * the per-move budget triple satisfies P1 at **every** ply of a real game,
      since a single breach anywhere would have voided the pair.
    """
    match = clean_match

    assert match.status == COMPLETED, f"{match.status}: {match.reason}"
    assert match.tally.voids == (), f"unexpected voids: {match.tally.by_mode()}"
    assert match.outcome is not None
    assert len(match.outcome.records) == 2
    assert match.pairs_completed == 1
    assert match.tally.pairs_scored == 1
    assert match.admissible

    assert check_paired_books(match.outcome.match_set()) == (), (
        "§D4's pairing invariant broke on a cross-family match"
    )
    for record in match.outcome.records:
        assert record.result in {WHITE_WIN, DRAW, BLACK_WIN}
        assert {record.white, record.black} == {"v1-white", "v1-black"}
        assert record.book_id == bridged_book_id(resolved_gungi_mode())
    first, second = match.outcome.records
    assert first.seats == (second.black, second.white), "the seats must swap"
    assert first.pair_id == second.pair_id
    assert match.outcome.summary.games == 2
    assert match.outcome.summary.shortest_plies > 1


def test_the_referee_ahead_diagnostic_is_counted_and_is_not_a_void(
    clean_match: BridgedMatch,
) -> None:
    """§D6 M6: Komugi adjudicates and v1 is simply told ``reset``.

    "**Not a failure.** The referee has adjudicated." The point of the assertion
    is the second half: M6 increments a diagnostic and contributes **nothing**
    to any void bucket, so it can never reach the 2% admissibility gate.
    """
    assert clean_match.tally.voids == ()
    assert clean_match.tally.referee_ahead >= 1
    assert clean_match.tally.by_bucket() == {}
    assert clean_match.status == COMPLETED


def test_the_probe_returns_the_adapter_s_own_observation() -> None:
    """§D1's ``probe`` request, which A4's differential probe is served by.

    Rev 1's protocol had no request that returns an observation, which made A4
    -- the sole refutation of RISK C3 -- unimplementable against the
    pre-registered protocol (review F10).
    """
    connection = spawn_v1(stub_config())
    try:
        connection.reset(pair=0, seat=0)
        observation = connection.probe()
        assert observation.ndim == 3
        assert observation.shape[1:] == (9, 9)
        assert observation.dtype.name == "float32"
        assert float(observation.sum()) != 0.0, "an empty probe proves nothing"
    finally:
        connection.close()


# ---------------------------------------------------------------------------
# §D6, mode by mode, against a process that really does it
# ---------------------------------------------------------------------------


def test_a_crash_mid_game_voids_the_pair_and_is_never_a_result() -> None:
    """§D6 M1: pipe EOF on the next ``pick``, with stderr and exit status kept."""
    match = play(CRASH, after_games=0, after_picks=2)

    void = sole_void(match)
    assert void.mode == M1
    assert void.bucket is VoidBucket.CRASH
    assert void.scope is VoidScope.PAIR
    assert void.exit_status not in (None, 0), "a crash has a non-zero exit status"
    assert "dying on purpose" in void.stderr, (
        "§D1 requires v1's stderr captured verbatim into the void record"
    )
    assert void.player_id == "v1-broken"
    assert void.ply is not None and void.ply >= 2
    assert_nothing_was_scored(match)
    assert match.tally.crashes == 1


def test_three_crashes_abort_the_match_and_the_abort_keeps_nothing() -> None:
    """§D6 precedence 1 and 2: the absolute rule fires first and is terminal.

    "An aborted match is scored not at all." Its already-completed pairs are
    discarded, no ``MatchSet`` is emitted, and the run is reported as
    ``aborted(reason, pairs_completed)`` -- because voids are non-random, so the
    surviving pairs of an aborted match are exactly the biased sample the 2%
    gate exists to refuse.
    """
    match = play(CRASH, after_games=0, after_picks=0, games=8)

    assert match.status == ABORTED
    assert match.reason == VoidBucket.CRASH.value
    assert match.tally.crashes == 3, "the third crash is the terminal one"
    assert match.tally.pairs_attempted == 3, "the match stopped at the third crash"
    assert match.outcome is None, "an aborted match emits no MatchSet"
    assert not match.admissible, (
        "an aborted match is inadmissible whatever its rate: at 3 voids in 3 "
        "attempted pairs the rate gate alone would not have caught it"
    )


def test_a_hang_is_killed_and_voids_the_pair_as_a_timeout() -> None:
    """§D6 M2: the per-``pick`` deadline, fatal by policy and never retried."""
    started = time.monotonic()
    match = play(HANG, after_games=0, after_picks=1, deadline_s=SHORT_DEADLINE_S)
    elapsed = time.monotonic() - started

    void = sole_void(match)
    assert void.mode == M2
    assert void.bucket is VoidBucket.TIMEOUT
    assert void.scope is VoidScope.PAIR
    assert "deadline" in void.detail
    assert elapsed < 120.0, (
        f"the harness waited {elapsed:.1f}s on a hung process: the deadline is "
        f"what stops a hang from stopping the run"
    )
    assert_nothing_was_scored(match)


def test_an_illegal_move_voids_the_pair_and_is_never_a_loss() -> None:
    """§D6 M3: the record matches no entry of Komugi's ``generate_all_moves``.

    "**Never a loss.**" Scoring a rules disagreement as a forfeit would measure
    the harness, and it would do so in the direction that flatters whichever
    engine is not the one being validated.
    """
    match = play(ILLEGAL, after_games=0, after_picks=1)

    void = sole_void(match)
    assert void.mode == M3
    assert void.bucket is VoidBucket.ILLEGAL_MOVE
    assert void.scope is VoidScope.PAIR
    assert void.record is not None, "M3 logs the offending record"
    assert void.fen, "M3 logs the position's FEN"
    assert void.record.move_type == 0 and void.record.from_r == -1
    assert_nothing_was_scored(match)


def test_a_malformed_record_is_a_protocol_error_not_a_rules_question() -> None:
    """§D6 M4, and the distinction from M3 that §D6 draws deliberately.

    "Distinguished from M3 because M3 is a rules question and M4 is a
    version-skew question." Folding the two would point the operator's
    diagnostic at the wrong subsystem: an out-of-range square is a schema fault
    in the adapter, not a disagreement about gungi.
    """
    match = play(MALFORMED, after_games=0, after_picks=1)

    void = sole_void(match)
    assert void.mode == M4
    # The M3/M4 confusion assert comes first on purpose: it is the claim this
    # test exists for, and stated after the positive one it would be reading a
    # bucket already narrowed to PROTOCOL_ERROR -- true by construction rather
    # than by what the bridge did.
    assert void.bucket is not VoidBucket.ILLEGAL_MOVE, (
        "an out-of-range square is a schema fault in the adapter, not a "
        "disagreement about gungi"
    )
    assert void.bucket is VoidBucket.PROTOCOL_ERROR
    assert "to_r" in void.detail
    assert_nothing_was_scored(match)


def test_protocol_garbage_voids_the_pair_as_a_protocol_error() -> None:
    """§D6 M4: a line that is not a protocol message at all."""
    match = play(GARBAGE, after_games=0, after_picks=1)

    void = sole_void(match)
    assert void.mode == M4
    assert void.bucket is VoidBucket.PROTOCOL_ERROR
    assert_nothing_was_scored(match)


def test_a_rules_disagreement_is_its_own_bucket_and_not_a_crash() -> None:
    """§D6 M5, the mode the design says is **expected to occur**.

    The whole reason ``no_move`` is a first-class response type: with an
    all-zero mask a real v1's MCTS "raises, divides by zero over an empty
    policy, or returns an arbitrary index", so the harness would observe M1, M4
    or M3. The observed *total* void rate would still be right -- which is why
    this test asserts the **bucket**, not just that something was voided: a
    misattributed M5 fires M1's "three crashes abort the match" on a rules
    divergence and points the operator at the wrong subsystem.
    """
    match = play(DISAGREE, after_games=0, after_picks=1)

    void = sole_void(match)
    assert void.mode == M5
    # Same ordering, same reason as the M4 test: asserted after the positive
    # one, "not a crash" would be a statement about the narrowed bucket rather
    # than about the misattribution that would fire M1's abort rule on a rules
    # divergence.
    assert void.bucket is not VoidBucket.CRASH, (
        "a misattributed M5 fires 'three crashes abort the match'"
    )
    assert void.bucket is VoidBucket.RULES_DIVERGENCE
    assert void.scope is VoidScope.PAIR
    assert match.tally.crashes == 0, "an M5 must not count toward the crash abort"
    assert "win_state=(0, 0, 1)" in void.detail
    assert void.fen
    assert_nothing_was_scored(match)


def test_a_budget_breach_voids_the_match_and_discards_completed_pairs() -> None:
    """§D6 M8 / §D3.5's report-and-refuse, with rev 3's discard rule.

    Three separable claims, and the third is the one rev 2 got wrong:

    * a breach on a **live** connection is M8, not M1 -- the process answered
      the ``pick``, so nothing here is a crash;
    * the resolution is a **match** void, not a pair void, because 257 is the
      arithmetic maximum of the pre-registered structure: a 258th request means
      every move already played was searched under an unknown budget;
    * a match-void **is an abort** (re-review F8): the pair completed before it
      is discarded and no ``MatchSet`` is emitted. Voiding only the affected
      pair "would keep exactly the games where the discrepancy happened not to
      show".
    """
    # after_games=2: pair 0 is games 1 and 2, so the breach lands on the first
    # pick of game 3 -- pair 1 -- with a completed pair already behind it.
    match = play(BREACH, after_games=2, after_picks=0, games=4)

    void = sole_void(match)
    assert void.mode == M8
    assert void.bucket is VoidBucket.BUDGET_BREACH
    assert void.scope is VoidScope.MATCH
    assert void.invariant == "P2"
    assert void.evals_used == 258
    assert match.tally.crashes == 0, "a breach outranks and excludes crash (§D6 rule 5)"

    assert match.status == VOIDED
    assert match.reason == "P2"
    assert match.pairs_completed == 1, "one pair had been played before the breach"
    assert match.outcome is None, (
        "a match-void discards its completed pairs; keeping them would keep "
        "exactly the games where the discrepancy happened not to show"
    )
    assert not match.admissible


def test_a_carried_tree_voids_the_match_on_p3() -> None:
    """§D6 M9: ``tree_nodes_at_entry != 0`` -- the only detector there is.

    §D3.2 clause 2 is *not* behaviour-neutral: a retained subtree carries visits
    paid for out of an earlier move's budget, so "a per-move budget" stops being
    one -- retroactively, for every move already played, which is why this is a
    match void and not a pair void.
    """
    match = play(TREE_REUSE, after_games=2, after_picks=0, games=4)

    void = sole_void(match)
    assert void.mode == M9
    assert void.bucket is VoidBucket.TREE_REUSE
    assert void.scope is VoidScope.MATCH
    assert void.invariant == "P3"
    assert void.tree_nodes_at_entry == 41
    assert match.status == VOIDED
    assert match.reason == "P3"
    assert match.pairs_completed == 1
    assert match.outcome is None


def test_a_forced_move_reporting_zero_evaluations_voids_the_pair() -> None:
    """§D6 M10 / acceptance A7b's v1-side half, end to end.

    "The adapter must short-circuit identically" is readable as *zero*
    evaluations on a forced move, and §D3.4 pins that it is not: exactly one
    root evaluation through the counted evaluator, mirroring ``gumbel.py:858``.
    An adapter that reports ``sims_used == 0`` with ``evals_used == 0`` must be
    **rejected** -- and rejected as a per-move accounting failure (a pair void),
    not as the match-level fault M8 and M9 carry.
    """
    match = play(ZERO_EVALS, after_games=0, after_picks=1)

    void = sole_void(match)
    assert void.mode == M10
    assert void.bucket is VoidBucket.BUDGET_INVARIANT
    assert void.scope is VoidScope.PAIR, "M10 does not impugn earlier pairs"
    assert void.invariant == "P1"
    assert void.evals_used == 0
    assert void.sims_used == 0
    assert match.status == COMPLETED, "a pair void does not end the match"
    assert_nothing_was_scored(match)


# ---------------------------------------------------------------------------
# §D1's capability gate: refused at startup, before any game
# ---------------------------------------------------------------------------


def test_an_adapter_that_cannot_report_a_breach_is_refused_at_startup() -> None:
    """Acceptance A13, the gate that keeps re-review F1 from re-instantiating.

    Without it, a non-conforming adapter that simply dies on a budget breach
    turns a match-level, v1-only breach into an M1 ``crash`` pair void -- and
    nothing on the wire would distinguish the two. With it, that outcome is a
    **startup** failure rather than a silently mis-bucketed rated match.
    """
    with pytest.raises(HandshakeRefused, match="supports_budget_breach"):
        play_bridged_match(
            stub_entrant("v1-broken", DECLINES_BREACH),
            stub_entrant("v1-correct"),
            match_config(games=2),
        )


def test_an_adapter_that_cannot_report_no_move_is_refused_at_startup() -> None:
    """The M5 half of the same gate: an undeclared ``no_move`` is a startup fault."""
    with pytest.raises(HandshakeRefused, match="supports_no_move"):
        spawn_v1(stub_config(DECLINES_NO_MOVE))


def test_a_start_mode_mismatch_is_fatal_at_startup() -> None:
    """§D4 amendment 2: two processes, two ``GUNGI_MODE`` values, one lie.

    ``book_id="empty"`` would be shared by both games of every pair and the
    paired-book audit only checks that the two *agree* (``seats.py:143-149``),
    so this cannot be caught later. It is fatal here or it is invisible.
    """
    with pytest.raises(HandshakeRefused, match="GUNGI_MODE"):
        spawn_v1(stub_config(BAD_MODE))


def test_a_protocol_version_mismatch_is_fatal_at_startup() -> None:
    """Version skew is caught before a move record can be misread as M4."""
    with pytest.raises(HandshakeRefused, match="protocol version"):
        spawn_v1(stub_config(BAD_VERSION))


def test_an_adapter_that_never_answers_is_refused_rather_than_waited_on() -> None:
    """A command that is not an adapter fails at construction, not at ply 1."""
    config = V1Config(
        argv=(sys.executable, "-c", "import sys; sys.exit(7)"),
        gungi_mode=resolved_gungi_mode(),
        pick_deadline_s=SHORT_DEADLINE_S,
    )
    with pytest.raises(HandshakeRefused, match="handshake"):
        spawn_v1(config)


# ---------------------------------------------------------------------------
# the rated path's own gates
# ---------------------------------------------------------------------------


def test_a_rated_match_refuses_a_book_id_that_does_not_name_the_start_mode() -> None:
    """§D4 amendment 2, enforced where an anchor is actually produced."""
    with pytest.raises(ValueError, match="book_id"):
        play_bridged_match(
            stub_entrant("v1-a"),
            stub_entrant("v1-b"),
            ArenaConfig(games=2, seed=1, draw_score=DRAW_SCORE, book_id="empty"),
            rated=True,
        )


def test_a_rated_match_plays_at_the_pre_registered_budget() -> None:
    """A15's literals hold, and a rated match runs to a paired outcome."""
    match = play_bridged_match(
        stub_entrant("v1-a"), stub_entrant("v1-b"), match_config(games=2), rated=True
    )
    assert match.status == COMPLETED
    assert match.outcome is not None
    assert check_paired_books(match.outcome.match_set()) == ()


def test_two_entrants_with_one_id_are_refused_before_any_process_is_spawned() -> None:
    """``play_match``'s own refusal, kept: a record cannot name one player twice."""
    with pytest.raises(ValueError, match="both entrants"):
        play_bridged_match(
            stub_entrant("v1-same"), stub_entrant("v1-same"), match_config(games=2)
        )


# ---------------------------------------------------------------------------
# kom-011d §D3 -- the book seam, on the bridged path
# ---------------------------------------------------------------------------


def booked_config(book: PoolBook, games: int = 2) -> ArenaConfig:
    """Return a cross-family match config that declares `book` (§D4 amendment 4)."""
    return ArenaConfig(
        games=games,
        seed=0xC0FFEE,
        draw_score=DRAW_SCORE,
        book_id=book.epoch_book_id,
        book=book,
    )


def test_a_bridged_game_opens_on_the_pairs_prefix_and_tells_both_movers(
    tmp_path: Path,
) -> None:
    """kom-011d §D3.4 on the wire: the prefix is replayed **and announced**.

    The far side keeps its own board and is synchronised by ``observe`` and
    nothing else (``v1bridge.py:1159-1169``: the v1 process *"has no other way
    to learn what the opponent played"*, and the fake v1 in this file plays
    every observed record onto its own :class:`~komugi.engine.game.Game`). So
    this game can only reach a legal finish if both movers were told all
    ``plies`` of the opening: an adapter that started from the empty board would
    propose, at the very first real ply, a move that is illegal in the referee's
    position -- §D6 M3, a pair void with a FEN attached.

    That is the non-vacuity, and it is mechanical rather than argued: the
    assertions are "the moves begin with the entry" **and** "no void", and
    deleting the ``announce`` callback from the replay turns the second into an
    M3 or M5 void rather than into a subtly different game.
    """
    book = epoch_book(tmp_path, epoch=7, k=1, entries=8, seed=81)
    entry = book.entries[0]
    config = booked_config(book)

    white = stub_entrant("v1-white").mover_factory()
    black = stub_entrant("v1-black").mover_factory()
    try:
        game = play_bridged_game(
            white, black, config, game_seeds(config, 0, 0), pair=0, seat=0
        )
    finally:
        white.close()
        black.close()

    assert game.void is None, (
        f"the booked game voided as {game.void.mode if game.void else ''}: "
        f"a prefix the movers were not told about desynchronises the far side's "
        f"board, and the first real ply is then illegal in the referee's"
    )
    assert game.outcome is not None
    assert game.outcome.moves[: entry.plies] == entry.actions
    assert len(game.outcome.moves) > entry.plies, "the prefix is not the game"
    assert game.outcome.result in {WHITE_WIN, DRAW, BLACK_WIN}


class PlyRecordingMover:
    """A :class:`~komugi.eval.v1bridge.Mover` that notes what it was handed.

    It plays the lowest legal action every time -- what is under test is the
    ``ply`` index and the seed the loop passes in, not the move that comes back
    -- and it keeps them, plus every action it was told about, so the prefix
    announcement and the searched plies can both be read off one game.

    In-process rather than over the wire on purpose: the real v1 (and this
    file's fake one) never looks at ``ply``, which is precisely why the
    reviewer's mutant M2 survived the whole bridged suite.
    """

    def __init__(self) -> None:
        self.plies: list[int] = []
        self.seeds: list[int] = []
        self.observed: list[int] = []
        self.last_win_state: tuple[int, int, int] = (0, 0, 0)

    def pick(self, state: GameState, *, ply: int, seed: int) -> int:
        """Record `ply` and `seed`, and play the lowest legal action."""
        self.plies.append(int(ply))
        self.seeds.append(int(seed))
        return min(legal_records(state).values())

    def observe(self, action: int, record: MoveRecord) -> None:
        """Note one played action -- the opening's, or the opponent's."""
        del record
        self.observed.append(int(action))

    def reset(self, *, pair: int, seat: int) -> None:
        """Start a new game, forgetting the previous one."""
        del pair, seat
        self.plies.clear()
        self.seeds.clear()
        self.observed.clear()

    def close(self) -> None:
        """Release nothing: this mover holds no process and no socket."""


def test_a_bridged_booked_games_searched_plies_keep_their_true_ply_seed(
    tmp_path: Path,
) -> None:
    """Review F3, bridged half: the wire is handed the **absolute** ply.

    ``play_bridged_game`` starts ``ply`` at ``len(moves)`` after the prefix is
    replayed, so the first searched move of a booked game is asked for at ply
    ``L`` and seeded with ``move_seed(seeds, L)`` -- the same schedule the
    in-process loop uses, which is what makes a bridged game and an in-process
    one comparable at all. Nothing saw it: ``ply`` is only forwarded to the
    adapter, and neither the real v1 nor this file's fake one validates it, so
    the reviewer's mutant M2 (``ply = 0``) survived every bridged test.

    *Non-vacuity:* the plies are asserted as the exact consecutive run
    ``L, L+1, ...`` rather than merely "not zero", and each mover's recorded
    seed is re-derived from :func:`~komugi.selfplay.records.move_seed` here, so
    the node is red against a loop that offset the ply but seeded from the
    wrong index -- and the prefix announcement is asserted too, which is what
    says the ``L`` being counted is the opening that was actually played.
    """
    book = epoch_book(tmp_path, epoch=7, k=1, entries=8, seed=93)
    entry = book.entries[0]
    config = booked_config(book)
    seeds = game_seeds(config, 0, 0)

    white = PlyRecordingMover()
    black = PlyRecordingMover()
    game = play_bridged_game(white, black, config, seeds, pair=0, seat=0)

    assert game.void is None, f"the booked game voided: {game.void}"
    assert game.outcome is not None
    assert game.outcome.moves[: entry.plies] == entry.actions

    for mover in (white, black):
        assert mover.observed[: entry.plies] == list(entry.actions), (
            "both movers are told the opening, action by action (§D3.4)"
        )
        assert mover.seeds == [move_seed(seeds, ply) for ply in mover.plies], (
            "a mover's seed is move_seed of the ply it was handed"
        )

    handed = sorted(white.plies + black.plies)
    assert handed == list(range(entry.plies, entry.plies + len(handed))), (
        f"the bridged loop handed plies {handed[:4]}... for a game opening on a "
        f"{entry.plies}-ply book: the prefix is not being counted into the ply "
        f"index, so a searched move at absolute ply p carries another ply's seed "
        f"and a bridged booked game is not the in-process one"
    )


def test_a_rated_match_declares_the_epoch_book_it_was_played_on(
    tmp_path: Path,
) -> None:
    """KOM-023 §D4 amendment 4: the rated ``book_id`` check reads the epoch book.

    Amendment 2 pinned ``book_id`` to ``empty-<mode>`` because ``"empty"`` alone
    does not name *which* empty board two processes started from. kom-011d gives
    a rated match a second legal start -- a frozen epoch book -- and the
    invariant reads it: the config must declare the book it was handed, and the
    per-game ids are that book's entry ids.

    Three halves, and the first two are what the amendment changes:

    * a match handed a book while still declaring ``empty-<mode>`` is
      **refused** -- red against the pre-amendment check, which would have
      accepted exactly that config and stored records whose ``book_id`` s named
      openings the declaration denied. Since review F10 that refusal arrives one
      layer earlier, at :class:`~komugi.eval.arena.ArenaConfig` construction, so
      the undeclared config never reaches a match at all; the rated check keeps
      the same invariant behind it and both are the same sentence;
    * declaring the book's own id is accepted, and the records carry the entry
      id rather than the declaration;
    * with **no** book, amendment 2 is untouched, which the bare-``"empty"``
      refusal below pins.
    """
    book = epoch_book(tmp_path, epoch=7, k=1, entries=8, seed=91)

    with pytest.raises(ValueError, match="epoch book"):
        play_bridged_match(
            stub_entrant("v1-a"),
            stub_entrant("v1-b"),
            ArenaConfig(
                games=2,
                seed=1,
                draw_score=DRAW_SCORE,
                book_id=bridged_book_id(resolved_gungi_mode()),
                book=book,
            ),
            rated=True,
        )

    match = play_bridged_match(
        stub_entrant("v1-a"), stub_entrant("v1-b"), booked_config(book), rated=True
    )
    assert match.status == COMPLETED, f"{match.status}: {match.reason}"
    assert match.outcome is not None
    assert {record.book_id for record in match.outcome.records} == {
        book.entries[0].id
    }
    assert check_paired_books(match.outcome.match_set()) == ()

    with pytest.raises(ValueError, match="book_id"):
        play_bridged_match(
            stub_entrant("v1-a"),
            stub_entrant("v1-b"),
            ArenaConfig(games=2, seed=1, draw_score=DRAW_SCORE, book_id="empty"),
            rated=True,
        )


# ---------------------------------------------------------------------------
# NON-VACUITY (L20): the §D6 index
# ---------------------------------------------------------------------------
#
# Per §D6 mode: the behaviour that provokes it, the test that catches it, and
# the assertion that fires if the bridge mis-scored it. "Mis-scored" means the
# mode reached a GameRecord as a result, or reached the wrong bucket -- both are
# failures §D6 names explicitly, and both are covered below.
#
# M1  crash          stub `crash`        test_a_crash_mid_game_voids_the_pair...
#                    fires: sole_void(mode == M1) and assert_nothing_was_scored;
#                    a crash booked as a loss makes `records != ()` fail, and a
#                    crash booked as M4/M2 fails the mode and bucket asserts.
# M1  abort rule     stub `crash`, 4 pairs
#                    test_three_crashes_abort_the_match_and_the_abort_keeps_nothing
#                    fires: status == ABORTED, tally.crashes == 3,
#                    pairs_attempted == 3, outcome is None. Keeping the
#                    completed pairs of an aborted match fails `outcome is None`.
# M2  timeout        stub `hang`         test_a_hang_is_killed_and_voids_the_pair...
#                    fires: mode == M2, bucket TIMEOUT, and the elapsed-time
#                    assert -- a missing deadline hangs the suite instead.
# M3  illegal move   stub `illegal`      test_an_illegal_move_voids_the_pair...
#                    fires: mode == M3, bucket ILLEGAL_MOVE, void.record and
#                    void.fen present, and nothing scored. An illegal move
#                    played anyway would raise in the engine and surface as M7;
#                    scored as a loss it fails assert_nothing_was_scored.
# M4  malformed      stub `malformed`    test_a_malformed_record_is_a_protocol_error...
#                    fires: mode == M4 AND `bucket is not ILLEGAL_MOVE` --
#                    the explicit M3/M4 confusion assert.
# M4  garbage        stub `garbage`      test_protocol_garbage_voids_the_pair...
#                    fires: mode == M4, bucket PROTOCOL_ERROR.
# M5  divergence     stub `disagree`     test_a_rules_disagreement_is_its_own_bucket...
#                    fires: bucket RULES_DIVERGENCE, `bucket is not CRASH`, and
#                    tally.crashes == 0 -- the misattribution the design says
#                    would otherwise fire M1's abort rule on a rules divergence.
# M6  referee ahead  clean match         test_the_referee_ahead_diagnostic_is_counted...
#                    fires: referee_ahead >= 1 with voids == () -- booking M6 as
#                    a void (it is "not a failure") fails the second assert and
#                    would feed the 2% gate.
# M7  engine raises  covered structurally: legal_records/record_of_action wrap
#                    the engine's decoder, and the M3 test's `assert void.fen`
#                    plus test_the_record_index_... (unit) pin that a decode
#                    fault becomes a void rather than a crash of the arena.
# M8  budget breach  stub `breach`       test_a_budget_breach_voids_the_match...
#                    fires: scope MATCH, invariant "P2", crashes == 0 (the §D6
#                    rule-5 precedence), status VOIDED, pairs_completed == 1 and
#                    outcome is None -- the last two are rev 3's discard rule,
#                    and rev 2's "void the pair" reading fails all three.
# M9  tree reuse     stub `tree_reuse`   test_a_carried_tree_voids_the_match_on_p3
#                    fires: scope MATCH, invariant "P3", tree_nodes_at_entry
#                    reported, outcome is None. Treating it as a pair void fails
#                    status/outcome.
# M10 P1 breach      stub `zero_evals`   test_a_forced_move_reporting_zero_evaluations...
#                    fires: mode == M10, invariant "P1", scope PAIR (explicitly
#                    NOT match), status still COMPLETED. Accepting evals_used==0
#                    fails sole_void; escalating it to a match void fails scope.
#
# A13 capability gate and §D4 amendment 2 are covered by the four startup tests,
# each of which asserts the refusal happens with **no game played** (the
# exception escapes play_bridged_match/spawn_v1 before any pair runs).


# ---------------------------------------------------------------------------
# KOM-141 -- the bridged referee adjudicates under the config's rule
# ---------------------------------------------------------------------------


class RuleRecordingMover:
    """A mover that notes which repetition rule the referee it is handed uses.

    The :class:`PlyRecordingMover` idiom, pointed at the one thing no existing
    bridged assertion can see: ``play_bridged_game`` builds the referee itself,
    so nothing outside the loop holds a reference to it. Reading
    ``state.game.repetition_rule`` at the first ``pick`` is how the rule the
    game is actually being adjudicated under becomes observable.

    It plays the lowest legal action, like its sibling: what is under test is
    the referee, not the move that comes back.
    """

    def __init__(self) -> None:
        self.rules: list[Any] = []

    def pick(self, state: GameState, *, ply: int, seed: int) -> int:
        """Note the referee's rule, then play the lowest legal action."""
        del ply, seed
        self.rules.append(state.game.repetition_rule)
        return min(legal_records(state).values())

    def observe(self, action: int, record: MoveRecord) -> None:
        """Note nothing: this mover keeps no board."""
        del action, record

    def reset(self, *, pair: int, seat: int) -> None:
        """Start a new game, forgetting the previous one's readings."""
        del pair, seat
        self.rules.clear()

    def close(self) -> None:
        """Release nothing."""


def _rules_seen_by_a_bridged_game(rule: RepetitionRule) -> list[Any]:
    """Play one in-process bridged game at `rule`; return what white observed."""
    config = ArenaConfig(
        games=2, seed=17, draw_score=DRAW_SCORE, repetition_rule=rule
    )
    white = RuleRecordingMover()
    black = RuleRecordingMover()
    played = play_bridged_game(
        white, black, config, game_seeds(config, 0, 0), pair=0, seat=0
    )
    assert played.void is None, f"the game voided: {played.void!r}"
    assert played.outcome is not None
    assert white.rules, "white was never asked to move"
    return white.rules


def test_a_bridged_game_referees_under_the_configs_repetition_rule() -> None:
    """KOM-141: ``play_bridged_game`` builds its referee from the config.

    The bridged loop constructs the ``Game`` itself, so a config field that
    never reached it would be invisible from outside -- the match would simply
    be refereed under the wrong rule and score normally. The mover reads the
    rule off the referee it is handed, at every ply, which is the only place it
    can be seen.

    *Non-vacuity:* both values are played, and the whole game's readings are
    asserted rather than the first, so the node is red against a loop that
    built the right referee and then replaced it, and against one that
    hardcoded either rule.
    """
    full = _rules_seen_by_a_bridged_game(RepetitionRule.FULL)
    assert set(full) == {RepetitionRule.FULL}

    placement = _rules_seen_by_a_bridged_game(RepetitionRule.PLACEMENT)
    assert set(placement) == {RepetitionRule.PLACEMENT}


def test_a_bridged_game_defaults_to_the_full_repetition_rule() -> None:
    """Nothing flips silently: an undeclared config referees under FULL."""
    config = ArenaConfig(games=2, seed=17, draw_score=DRAW_SCORE)
    assert config.repetition_rule is RepetitionRule.FULL

    white = RuleRecordingMover()
    black = RuleRecordingMover()
    played = play_bridged_game(
        white, black, config, game_seeds(config, 0, 0), pair=0, seat=0
    )
    assert played.void is None
    assert set(white.rules) == {RepetitionRule.FULL}
