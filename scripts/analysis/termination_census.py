#!/usr/bin/env python3
"""KOM-142: how do arena games actually END?

``ArenaSummary`` counts draws and ply lengths and stops there -- *"a
``GameRecord`` has no length and, by design, no termination field"*
(``arena.py:469-489``). So a match that is 40% draws cannot say whether those
draws are fourfold repetitions, insufficient material, stalemates, or games
that simply ran into the env's own ply cap (rules-audit F6). Those four
have very different implications: the first says the players are shuffling,
the last says the cap is rating the players.

This script answers the question the only way that cannot drift from the
engine: it plays real arena games, then **replays each game's moves on a fresh
``Game`` and asks** :meth:`komugi.engine.game.Game.termination_reason`. Nothing
here re-derives a terminal rule, and nothing reads a result to guess a reason.

The two seams it plays through
------------------------------
* ``rawmcts:<sims>`` and ``random`` are :class:`~komugi.eval.v1bridge.Mover` s,
  played through :func:`~komugi.eval.v1bridge.play_bridged_game` -- the arena
  loop with the Gumbel search replaced by ``Mover.pick`` (``v1bridge.py:1859``).
  That is the seam a net-free player is seated at; ``ArenaPlayer`` wants an
  evaluator factory and a RawMCTS ruler has no network to build.
* ``checkpoint:<path>`` is an :class:`~komugi.eval.arena.ArenaPlayer` over
  :func:`~komugi.eval.arena.checkpoint_evaluator`, played through
  :func:`~komugi.eval.arena.play_arena_game`.

Both produce a :class:`~komugi.eval.arena.GameOutcome`, and everything
downstream of that point is one code path.

Games are played as the arena's seat-swapped pairs (``2k`` and ``2k+1`` are one
pair) under the arena's own seed derivation, :func:`~komugi.eval.arena.game_seeds`,
so a census at seed ``S`` is reproducible and a census of a self-match is not
silently a census of one seat.

Which repetition rule (KOM-141)
-------------------------------
``--rule`` names the fourfold-repetition key the games are refereed under AND
replayed under -- one value, both places, because a game refereed under
gungi.js's cells-only ``placement`` key ends at a draw a ``full`` replay cannot
see, and the replay would then run out of moves in a live position and raise.
The default is ``full``, Komugi's repaired key, which is what self-play and the
ladder play under; ``placement`` is how the site's rule is measured against the
same players at the same seed. The report carries the rule it was taken under,
so two runs cannot be confused after the fact.

Usage::

    python scripts/analysis/termination_census.py --games 64 --seed 7 \\
        --entrant rawmcts:64 --out census.json

    python scripts/analysis/termination_census.py --games 64 --seed 7 \\
        --entrant rawmcts:64 --rule placement --out census-placement.json
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Final

import numpy as np

from komugi.engine.game import (
    TERMINATION_FOURFOLD_REPETITION,
    TERMINATION_NONE,
    TERMINATION_REASONS,
    Game,
)
from komugi.engine.repetition import RepetitionRule
from komugi.eval.arena import (
    ArenaConfig,
    ArenaPlayer,
    GameOutcome,
    LADDER_FAMILY,
    checkpoint_evaluator,
    game_seeds,
    play_arena_game,
)
from komugi.eval.rawmcts import DEFAULT_C_PUCT, MctsState, RawMctsMover, legal_actions
from komugi.eval.records import DRAW, Player
from komugi.eval.v1bridge import Mover, play_bridged_game

DRAW_SCORE: Final[float] = 0.0
"""The contempt every census game is played under.

``kom-011d`` §D7.3 pins ``0.0`` for every pool game, and ``ArenaConfig`` takes
no default for it on purpose (design review D3): a census played at a different
contempt from the pool is a census of a different experiment.
"""

DEFAULT_ENTRANT: Final[str] = "rawmcts:64"
"""The default self-play opponent: RawMCTS at 64 simulations a move.

Below §D1.1's pre-registered ``{100, 800, 6400}`` because a census wants MANY
games rather than strong ones -- the distribution over termination reasons is
what is being measured, and it is estimated from the game count.
"""

RAWMCTS_SCHEME: Final[str] = "rawmcts"
RANDOM_SCHEME: Final[str] = "random"
CHECKPOINT_SCHEME: Final[str] = "checkpoint"

REPORT_KEYS: Final[tuple[str, ...]] = (
    "games",
    "by_reason",
    "draw_share",
    "fourfold_share",
    "mean_plies",
    "seed",
    "entrant",
    "rule",
)
"""Exactly the keys :func:`census` writes -- the JSON contract, in one place."""

DEFAULT_RULE: Final[RepetitionRule] = RepetitionRule.FULL
"""The repetition key a census referees under unless ``--rule`` says otherwise.

``full`` is Komugi's repaired 274-byte key (rules-audit F1-F4 / KOM-027) and is
what self-play and every ladder game is played under, so it is the census that
describes the training run. ``placement`` is gungi.js@3.4.11's cells-only key,
for measuring what the site's rule would do to the same players.
"""

_SEATS: Final[tuple[int, int]] = (0, 1)
"""The two seatings of an arena pair (``arena.py:942``)."""


class RandomMover:
    """A uniformly random legal mover, for the cheapest possible census.

    Not a rated player and never seated as one: it exists because a census of
    *random* play is the base rate every other entrant's census is read
    against, and because it is the only entrant fast enough to smoke-test this
    script's pipeline in a test suite (a RawMCTS rollout is itself a random
    playout, so even ``rawmcts:2`` costs a game hundreds of times over).

    Deterministic given the seam's seed, by exactly
    :class:`~komugi.eval.rawmcts.RawMctsMover`'s construction: a fresh
    ``Generator(PCG64(seed))`` per pick, over the seed the loop already derives
    per ply. No generator survives a ``pick``, so a replayed game is the same
    game.
    """

    def __init__(self, player_id: str = "") -> None:
        self._player_id = player_id

    @property
    def player_id(self) -> str:
        """Return the identity this mover plays as."""
        return self._player_id

    def pick(self, state: MctsState, *, ply: int, seed: int) -> int:
        """Return a uniformly random legal action in `state`."""
        del ply
        actions = legal_actions(state)
        if actions.size == 0:
            raise ValueError(
                f"{self._player_id or 'the random mover'} was asked to move in a "
                f"position with no legal action; the game loop tests "
                f"is_terminal() before it asks (v1bridge.py:1929)"
            )
        rng = np.random.Generator(np.random.PCG64(int(seed)))
        return int(actions[rng.integers(actions.size)])

    def observe(self, action: int, record: object) -> None:
        """Note a played move. A no-op: this mover carries no board."""
        del action, record

    def reset(self, *, pair: int, seat: int) -> None:
        """Start a new game: there is nothing to drop."""
        del pair, seat

    def close(self) -> None:
        """Release what this holds: nothing."""


@dataclass(frozen=True)
class EntrantSpec:
    """A parsed ``--entrant`` argument: which engine, and at what budget."""

    scheme: str
    sims: int = 0
    path: Path | None = None

    @property
    def label(self) -> str:
        """Return the spelling that goes in the report's ``entrant`` field."""
        if self.scheme == RAWMCTS_SCHEME:
            return f"{RAWMCTS_SCHEME}:{self.sims}"
        if self.scheme == CHECKPOINT_SCHEME:
            return f"{CHECKPOINT_SCHEME}:{self.path}"
        return self.scheme


def parse_entrant(spec: str) -> EntrantSpec:
    """Parse ``rawmcts:<sims>``, ``random`` or ``checkpoint:<path>``.

    Raises:
        ValueError: On an unknown scheme, a non-positive or non-integer sim
            count, or a ``checkpoint:`` with an empty path. Loudly, because the
            alternative is a census that silently measured a different player
            from the one its filename claims.
    """
    scheme, _, rest = spec.partition(":")
    if scheme == RANDOM_SCHEME:
        if rest:
            raise ValueError(f"{RANDOM_SCHEME!r} takes no argument, got {spec!r}")
        return EntrantSpec(scheme=RANDOM_SCHEME)
    if scheme == RAWMCTS_SCHEME:
        try:
            sims = int(rest)
        except ValueError as error:
            raise ValueError(
                f"{spec!r} is not a simulation budget: expected "
                f"{RAWMCTS_SCHEME}:<sims>, e.g. {DEFAULT_ENTRANT!r}"
            ) from error
        if sims <= 0:
            raise ValueError(f"a RawMCTS budget must be positive, got {sims!r}")
        return EntrantSpec(scheme=RAWMCTS_SCHEME, sims=sims)
    if scheme == CHECKPOINT_SCHEME:
        if not rest:
            raise ValueError(
                f"{spec!r} names no checkpoint: expected "
                f"{CHECKPOINT_SCHEME}:<path to a .pt>"
            )
        return EntrantSpec(scheme=CHECKPOINT_SCHEME, path=Path(rest))
    raise ValueError(
        f"unknown entrant {spec!r}: expected one of "
        f"{RAWMCTS_SCHEME}:<sims>, {RANDOM_SCHEME}, {CHECKPOINT_SCHEME}:<path>"
    )


def mover_factory(spec: EntrantSpec) -> Callable[[str], Mover]:
    """Return a factory building one `spec` mover per call.

    Fresh per game rather than per match, which is the arena's own discipline
    for a stateful mover (``arena.py:900``) and costs nothing for these two:
    neither carries a tree or a net.
    """
    if spec.scheme == RAWMCTS_SCHEME:

        def build_rawmcts(player_id: str) -> Mover:
            """Build one RawMCTS ruler at this census's budget."""
            return RawMctsMover(
                sims=spec.sims, player_id=player_id, c_puct=DEFAULT_C_PUCT
            )

        return build_rawmcts

    if spec.scheme == RANDOM_SCHEME:

        def build_random(player_id: str) -> Mover:
            """Build one uniformly random legal mover."""
            return RandomMover(player_id=player_id)

        return build_random

    raise NotImplementedError(
        f"{spec.scheme!r} is not a Mover-seam entrant; a checkpoint plays "
        f"through komugi.eval.arena.play_arena_game (see play_games)"
    )


def _checkpoint_players(spec: EntrantSpec, device: str) -> tuple[ArenaPlayer, ...]:
    """Return the two seats a ``checkpoint:`` census self-matches between.

    Two entrants over ONE evaluator factory, carrying two ids -- the way the
    arena documents self-matching a checkpoint (``arena.py:914-921``); a record
    cannot name one player in both seats.
    """
    if spec.path is None:
        raise ValueError("a checkpoint entrant must carry a path")
    factory = checkpoint_evaluator(spec.path, device)
    return tuple(
        ArenaPlayer(
            player=Player(id=f"{spec.label}#{side}", family=LADDER_FAMILY),
            evaluator_factory=factory,
        )
        for side in ("a", "b")
    )


def play_games(
    *,
    games: int,
    seed: int,
    spec: EntrantSpec,
    device: str = "cpu",
    rule: RepetitionRule = DEFAULT_RULE,
) -> tuple[GameOutcome, ...]:
    """Play `games` self-play games and return their outcomes, in play order.

    Seat-swapped pairs under the arena's seed derivation: game ``2k`` and game
    ``2k+1`` are pair ``k`` from opposite seats, seeded
    ``game_seeds(config, k, seat)``. Bookless -- ``ArenaConfig.book`` is
    ``None``, so every game starts from the engine's own start position and
    ``book_id`` is :data:`~komugi.eval.arena.EMPTY_BOOK`; a book would make the
    census a census of the book's openings.

    Args:
        games: How many games to play. Must be even (``ArenaConfig`` refuses
            an odd count: a match is a whole number of pairs).
        seed: The census's entropy.
        spec: The entrant both seats play.
        device: Where a ``checkpoint:`` entrant's net is placed. Unread by the
            net-free entrants, which have nothing to place.
        rule: The repetition key the arena's referee adjudicates on (KOM-141).
            It reaches the games through :attr:`ArenaConfig.repetition_rule`,
            and the SAME value must reach :func:`classify_outcome`, or the
            replay is refereed under a different rule from the game and ends
            live -- which :func:`classify_outcome` raises on rather than
            classifying.

    Returns:
        One :class:`~komugi.eval.arena.GameOutcome` per game, in play order.

    Raises:
        ValueError: If a bridged game came back void rather than scored. A
            void is a cross-engine failure mode (§D6) and neither of this
            script's movers can produce one, so it means the loop failed in a
            way a census must not average over.
    """
    config = ArenaConfig(
        games=games, seed=seed, draw_score=DRAW_SCORE, repetition_rule=rule
    )

    if spec.scheme == CHECKPOINT_SCHEME:
        white, black = _checkpoint_players(spec, device)
        return tuple(
            play_arena_game(white, black, config, pair=pair, seat=seat)
            for pair in range(config.pairs)
            for seat in _SEATS
        )

    build = mover_factory(spec)
    outcomes: list[GameOutcome] = []
    for pair in range(config.pairs):
        for seat in _SEATS:
            played = play_bridged_game(
                build(f"{spec.label}#w"),
                build(f"{spec.label}#b"),
                config,
                game_seeds(config, pair, seat),
                pair=pair,
                seat=seat,
            )
            if played.outcome is None:
                raise ValueError(
                    f"pair {pair} seat {seat} came back void rather than "
                    f"scored: {played.void!r}"
                )
            outcomes.append(played.outcome)
    return tuple(outcomes)


def classify_outcome(
    moves: Sequence[int], rule: RepetitionRule = DEFAULT_RULE
) -> str:
    """Return the termination reason of the game whose moves are `moves`.

    Replayed on a **fresh** ``Game`` -- the same object the arena refereed
    with, from the same start position, **under the same repetition rule** --
    and then asked :meth:`komugi.engine.game.Game.termination_reason`. The
    replay is what makes the census the engine's own answer rather than an
    inference from a result: a result of ``0.5`` says "draw" and cannot say
    which of the four draw rules ended it.

    *rule* is not optional in practice (KOM-141): a game refereed under
    ``PLACEMENT`` ends at a cells-only fourfold that a ``FULL`` replay does not
    see, so the replay runs out of moves in a live position and raises below.
    That is the intended failure -- a mis-ruled census is refused rather than
    silently reporting the wrong reasons -- and it is why :func:`play_games`
    and this function are handed the same value by :func:`main`.

    Every action is checked against ``valid_moves()`` before it is played, so a
    move list that does not belong to this engine fails here, naming the ply,
    instead of quietly classifying some other game.

    Raises:
        ValueError: If an action is illegal at its ply, or if the replayed
            game does not end in a terminal position. The second is the bug the
            ticket calls out: an arena loop exits only on ``is_terminal()``, so
            a replay that ends live means the two disagree.
    """
    game = Game(repetition_rule=rule)
    for ply, action in enumerate(moves):
        mask = np.asarray(game.valid_moves(), dtype=np.uint8)
        if not 0 <= int(action) < mask.size or mask[int(action)] != 1:
            raise ValueError(
                f"ply {ply}: action {action!r} is not legal in the replayed "
                f"position; this move list was not played by this engine"
            )
        game.play_action(int(action))

    reason = game.termination_reason()
    if reason == TERMINATION_NONE:
        raise ValueError(
            f"the replay of a {len(moves)}-ply game ends in a position the "
            f"engine has not adjudicated: the arena loop runs while "
            f"not is_terminal(), so a live final position means the replay and "
            f"the game disagree"
        )
    return reason


def census(
    outcomes: Sequence[GameOutcome],
    *,
    seed: int = 0,
    entrant: str = "",
    rule: RepetitionRule = DEFAULT_RULE,
) -> dict[str, Any]:
    """Return the census report for `outcomes`.

    Keys are exactly :data:`REPORT_KEYS`. ``by_reason`` carries **all seven**
    reasons, zeros included, so a reader never has to know whether a missing
    key means zero or means the census predates that reason; it sums to
    ``games`` by construction, because :func:`classify_outcome` returns one of
    the seven or raises.

    ``draw_share`` is read off the arena's own ``result`` and is deliberately
    NOT derived from the reasons: agreeing that the draw rate and the drawing
    reasons match is a check worth being able to run, and it is not a check if
    one is computed from the other.

    ``rule`` is reported (KOM-141) because two censuses of the same players at
    the same seed under the two repetition rules are *different measurements*,
    and a report that did not say which one it is could not be told apart from
    the other after the fact. It is the same value the replays are run under.
    """
    by_reason: dict[str, int] = {reason: 0 for reason in TERMINATION_REASONS}
    plies = 0
    draws = 0
    for outcome in outcomes:
        by_reason[classify_outcome(outcome.moves, rule)] += 1
        plies += len(outcome.moves)
        if float(outcome.result) == DRAW:
            draws += 1

    games = len(outcomes)
    return {
        "games": games,
        "by_reason": by_reason,
        "draw_share": _share(draws, games),
        "fourfold_share": _share(by_reason[TERMINATION_FOURFOLD_REPETITION], games),
        "mean_plies": _share(plies, games),
        "seed": seed,
        "entrant": entrant,
        "rule": rule.value,
    }


def _share(count: int, games: int) -> float:
    """Return ``count / games``; ``0.0`` for an empty census."""
    return 0.0 if games == 0 else count / games


def format_table(report: dict[str, Any]) -> str:
    """Return the human-readable table: reason, count, share.

    Every reason is printed, zeros included, for :func:`census`'s reason: a
    reason that never fired is a finding, and a row that is absent looks like
    a reason that does not exist.
    """
    games = int(report["games"])
    by_reason: dict[str, int] = report["by_reason"]
    width = max(len(reason) for reason in TERMINATION_REASONS)
    lines = [
        f"termination census -- {report['entrant']}, "
        f"{games} games, seed {report['seed']}, rule {report['rule']}",
        f"{'reason'.ljust(width)}  {'count':>6}  {'share':>7}",
    ]
    for reason in TERMINATION_REASONS:
        count = int(by_reason[reason])
        lines.append(
            f"{reason.ljust(width)}  {count:>6}  {_share(count, games):>7.3f}"
        )
    lines.append(
        f"{'(draws)'.ljust(width)}  {'':>6}  {float(report['draw_share']):>7.3f}"
    )
    lines.append(f"mean plies: {float(report['mean_plies']):.1f}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Return the CLI parser."""
    parser = argparse.ArgumentParser(
        prog="termination_census.py",
        description=(
            "Play arena self-play games and count how they ended, by replaying "
            "each game and asking Game.termination_reason()."
        ),
    )
    parser.add_argument(
        "--games",
        type=int,
        default=2,
        help="games to play; must be even (a match is seat-swapped pairs)",
    )
    parser.add_argument("--seed", type=int, default=0, help="the census's entropy")
    parser.add_argument(
        "--entrant",
        default=DEFAULT_ENTRANT,
        help=(
            f"{RAWMCTS_SCHEME}:<sims> (default {DEFAULT_ENTRANT}), "
            f"{RANDOM_SCHEME}, or {CHECKPOINT_SCHEME}:<path>"
        ),
    )
    parser.add_argument(
        "--rule",
        choices=[member.value for member in RepetitionRule],
        default=DEFAULT_RULE.value,
        help=(
            "the fourfold-repetition key the games are refereed AND replayed "
            f"under (default {DEFAULT_RULE.value}): 'full' is Komugi's "
            "repaired key, 'placement' is gungi.js@3.4.11's cells-only key"
        ),
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="device for a checkpoint entrant's net; unread by the net-free ones",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="where to write the JSON report; stdout when omitted",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Play the games, write the report, and return an exit status."""
    args = build_parser().parse_args(argv)
    spec = parse_entrant(args.entrant)
    # One value, read once and handed to BOTH the games and their replays: a
    # census that refereed under one rule and classified under the other would
    # raise on the first placement-drawn game rather than report it (KOM-141).
    rule = RepetitionRule(args.rule)

    outcomes = play_games(
        games=args.games,
        seed=args.seed,
        spec=spec,
        device=args.device,
        rule=rule,
    )
    report = census(outcomes, seed=args.seed, entrant=spec.label, rule=rule)

    total = sum(report["by_reason"].values())
    if total != report["games"]:
        raise ValueError(
            f"by_reason sums to {total} over {report['games']} games: every "
            f"game is classified exactly once, so this is a counting bug"
        )

    print(format_table(report))
    serialized = json.dumps(report, indent=2, sort_keys=True)
    if args.out is None:
        print(serialized)
    else:
        out: Path = args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(serialized + "\n", encoding="utf-8")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
