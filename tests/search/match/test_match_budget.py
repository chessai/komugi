"""A13 and A15: the evaluation identity, and the parity defaults as literals.

Design §B1's identity, which the first three tests pin::

    evals(move) = [1 if the root was not already expanded else 0]
                + #{ playouts that expanded a new, non-terminal node }

It is an identity rather than an estimate because one playout expands at most
one node, a playout that lands on a terminal position asks nothing, and a
forced move short-circuits to one root evaluation and zero playouts. At
``playouts = 256`` on a fresh tree that gives a ceiling of exactly
``EVAL_CEILING`` -- the same arithmetic maximum the Gumbel side is instrumented
against, reached by the same derivation rather than by coincidence, which is
what lets one counter instrument both searches.

A15 then pins the parity posture as **literals** rather than trusting a
default: reuse off, ``nodes_at_entry == 0`` on every move, and the ceiling read
from the shared symbol rather than restated as ``257``. A retained subtree
carries visits paid for out of an earlier move's budget, so a move that
searched one would search a tree its own budget did not buy -- which is why
reuse is forbidden where the per-move budget is what makes two engines
comparable.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
from planted import DeepGame, MatchToyState, constant_wdl_evaluator

from komugi.eval.v1bridge import BudgetBreach
from komugi.eval.v1runner.protocol import EVAL_CEILING
from komugi.search.match.config import MatchConfig
from komugi.search.match.player import MatchPlayer
from komugi.search.match.search import MatchSearch

WIDTH = 3

#: Deeper than any budget here, so "no playout reached a terminal" is a
#: property of the fixture and the identity's inequality never bites.
DEEP = 600


def deep_search(config: MatchConfig) -> MatchSearch:
    """Return a search over the deep fixture, with a bland evaluator."""
    return MatchSearch(
        config,
        constant_wdl_evaluator(
            [0.25, 0.5, 0.25], num_actions=config.num_actions
        ),
    )


def test_a_fresh_tree_at_256_playouts_costs_exactly_the_ceiling() -> None:
    """T9: ``1 + 256`` evaluations, no more and no fewer."""
    config = MatchConfig(num_actions=WIDTH, playouts=256)
    search = deep_search(config)
    result = search.search(MatchToyState(DeepGame(width=WIDTH, depth=DEEP)))

    assert result.playouts_used == 256
    assert result.evals_used == 1 + result.playouts_used
    assert result.evals_used == EVAL_CEILING
    assert result.nodes_at_entry == 0


def test_a_forced_move_costs_one_evaluation_and_no_playouts() -> None:
    """T9: one legal action, one root evaluation, zero playouts.

    The root evaluation is **not** skipped, for the same reason the Gumbel
    side does not skip it: something downstream -- here the retained tree and
    the reported root value -- needs an honest number for the position even
    where there is no choice to make.
    """
    config = MatchConfig(num_actions=1, playouts=256)
    search = deep_search(config)
    result = search.search(MatchToyState(DeepGame(width=1, depth=10)))

    assert result.playouts_used == 0
    assert result.evals_used == 1
    assert result.action == 0
    assert int(result.visits.sum()) == 0


def test_reuse_enters_a_move_inside_a_tree_it_did_not_pay_for() -> None:
    """T9's third clause, and the reason reuse is banned on a parity path.

    With the root already expanded the move performs at most ``playouts``
    evaluations rather than ``1 + playouts`` -- and ``nodes_at_entry`` is what
    puts the regime on the record beside the number.
    """
    config = MatchConfig(num_actions=WIDTH, playouts=32, tree_reuse=True)
    search = deep_search(config)
    state = MatchToyState(DeepGame(width=WIDTH, depth=DEEP))

    first = search.search(state)
    assert first.nodes_at_entry == 0

    played = int(first.action)
    state.play(played)
    search.observe(played)

    second = search.search(state)
    assert second.nodes_at_entry > 0
    assert second.evals_used <= second.playouts_used
    assert second.evals_used < first.evals_used
    assert second.tree_reuse is True


def test_the_parity_defaults_are_off_by_default() -> None:
    """A15: asserted on a bare constructor, as literals."""
    assert MatchConfig().tree_reuse is False
    assert MatchPlayer(MatchConfig()).tree_reuse is False
    assert MatchConfig().playouts == 256
    assert MatchConfig().contempt == 0.0
    assert MatchConfig().subtree_bias_factor == 0.45
    assert MatchConfig().variance_mode == "moment"


def test_the_fresh_tree_ceiling_is_the_shared_symbol() -> None:
    """A15: read from ``EVAL_CEILING``, never restated as ``257``.

    Restating it would let the two ceilings drift apart silently, and the
    whole point of the number is that one counter instruments both searches.
    """
    assert 1 + MatchConfig(playouts=256).playouts == EVAL_CEILING


def test_every_move_of_a_no_reuse_run_enters_an_empty_tree() -> None:
    """A15: ``nodes_at_entry == 0`` per move, asserted Komugi-side.

    The wire field this mirrors is checked on the v1 ``pick`` response and has
    had no Komugi-side counterpart until now, so the report is not enough on
    its own: the assertion is the acceptance.
    """
    config = MatchConfig(num_actions=WIDTH, playouts=16, tree_reuse=False)
    search = deep_search(config)
    state = MatchToyState(DeepGame(width=WIDTH, depth=DEEP))
    for _ in range(4):
        result = search.search(state)
        assert result.tree_reuse is False
        assert result.nodes_at_entry == 0
        assert 1 <= result.evals_used <= 1 + result.playouts_used
        assert result.evals_used <= EVAL_CEILING
        played = int(result.action)
        state.play(played)
        search.observe(played)


def test_a_player_reports_the_regime_on_every_move() -> None:
    """A15: the fields exist on every ``MatchResult`` a player produces."""
    player = MatchPlayer(
        MatchConfig(num_actions=WIDTH, playouts=8),
        constant_wdl_evaluator([0.25, 0.5, 0.25], num_actions=WIDTH),
    )
    state = DeepGame(width=WIDTH, depth=DEEP)
    for ply in range(3):
        action = player.pick(MoverDeepGame(state), ply=ply, seed=ply)
        state.play(action)
        player.observe(action, None)
    assert len(player.results) == 3
    for result in player.results:
        assert result.tree_reuse is False
        assert result.nodes_at_entry == 0
    assert player.last_seed == 2


class MoverDeepGame:
    """A :class:`DeepGame` with the seam's surface, for the player-level test.

    The seven methods ``Mover.pick`` delivers, so a player can be driven end
    to end without an engine: the wrapper the player builds around it is the
    thing under test, not the game.
    """

    __slots__ = ("_game",)

    def __init__(self, game: DeepGame) -> None:
        self._game = game

    def legal_mask(self) -> npt.NDArray[np.bool_]:
        """Return the game's legality mask."""
        return self._game.legal_mask()

    def play(self, action: int) -> None:
        """Advance the game in place."""
        self._game.play(action)

    def clone(self) -> MoverDeepGame:
        """Return an independent copy."""
        return MoverDeepGame(self._game.clone())

    def is_terminal(self) -> bool:
        """Return whether the game is over."""
        return self._game.is_terminal()

    def to_play(self) -> int:
        """Return the index of the player to move."""
        return self._game.to_play()

    def observation(self) -> npt.NDArray[np.float32]:
        """Return a placeholder encoding; this suite has no network."""
        return np.zeros(1, dtype=np.float32)

    def absolute_outcome(self) -> npt.NDArray[np.uint8]:
        """Return an all-zero outcome; this fixture never reaches a result."""
        return np.zeros(3, dtype=np.uint8)


def test_a_refused_evaluation_reaches_the_loop_as_a_budget_breach() -> None:
    """§B2 clause 3: ``BudgetRefused`` is translated at the seam.

    ``EvalCounter`` raises ``BudgetRefused``, and the bridged game loop
    catches ``BridgeVoided`` and ``BudgetBreach`` only. Untranslated, the
    refusal would escape uncaught to the arena boundary and be filed as a
    crash rather than as the budget event it is -- the same failure, in the
    wrong bucket, with the ply that overspent lost.
    """
    player = MatchPlayer(
        MatchConfig(num_actions=WIDTH, playouts=16),
        constant_wdl_evaluator([0.25, 0.5, 0.25], num_actions=WIDTH),
        player_id="match",
        ceiling=3,
    )
    with pytest.raises(BudgetBreach) as raised:
        player.pick(MoverDeepGame(DeepGame(width=WIDTH, depth=DEEP)), ply=0, seed=0)
    assert raised.value.player_id == "match"
    assert raised.value.evals_used == 4
    # The move was closed even though it raised, so the next one starts clean.
    assert player.counter.moves_counted == 1
