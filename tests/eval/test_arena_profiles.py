"""Arena-side search profiles: the two modes an entrant may be rated under.

KOM-224 gives :class:`~komugi.eval.arena.ArenaPlayer` a ``proof_mode`` and a
``discovery_mode``, wires them through :func:`~komugi.eval.arena.move_search`
into the search's own config, and builds the production tactical provider once
per match for the entrant that asked for discovery. Everything here is about
that wiring, and three claims are load-bearing:

* **The legacy player is untouched.** Both modes default to ``off``, an
  ``off``/``off`` entrant's identity is
  :data:`~komugi.search.modes.LEGACY_IDENTITY`, and a seeded match between
  default players is the match it has always been -- same records, same summary,
  same record *fields*. A mode that leaked a default would show up here as a
  changed match, not as a diff nobody read.
* **The modes reach the search, per side.** The forwarding is asserted with a
  spy over ``arena.GumbelSearch``: every search built for the profiled side saw
  its two modes *and* a provider whose ``provider_id`` is the published one, and
  every search built for the plain side saw ``off``/``off`` and no provider. The
  two sides are told apart by their budgets, so the assertion is per side rather
  than per set.
* **A rated mover refuses discovery before it does anything.**
  :class:`~komugi.eval.gumbel_mover.GumbelMover` plays under §D3.3's counted
  ceiling of 257 rows a move, and the discovery layer can ask for more than that,
  so the configuration is refused -- and refused *before* the evaluator factory
  is called, which a spy factory that records its calls is what makes checkable.

The games are real, as in :mod:`test_arena_unit`: real ``Game``, real
``GumbelSearch``, real termination, stub networks. A faked loop would fake away
exactly the seam under test -- which side's modes reach which search.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, NamedTuple, cast

import pytest

pytest.importorskip("torch")
pytest.importorskip("komugi.net")
pytest.importorskip("komugi.selfplay")

from komugi.engine.game import Game
from komugi.eval import arena
from komugi.eval.gumbel_mover import GumbelMover, komugi_entrant
from komugi.eval.records import GameRecord, Player
from komugi.eval.v1bridge import (
    BridgeEntrant,
    HelloResponse,
    Mover,
    V1Connection,
    anchor_configuration,
)
from komugi.search import modes
from komugi.search.gumbel import GumbelConfig, GumbelSearch
from komugi.search.state import Evaluator, SearchState, WaveEvaluator
from komugi.search.tactical import TacticalProvider
from komugi.selfplay.adapter import GameState, make_evaluator
from komugi.selfplay.inference import BatchEvaluator
from komugi.selfplay.tactical import GameMarshalCaptureProvider

PRODUCTION_PROVIDER_ID: str = "game-marshal-capture-v1"
"""The provider id the arena's discovery mode is allowed to consult."""

PROFILE: tuple[str, str] = ("interval_v1", "root_marshal_v1")
"""The non-legacy profile under test: both layers on, as KOM-220 requires."""


def _load_test_helpers(filename: str, alias: str) -> ModuleType:
    """Import a sibling test module's helpers by path, under `alias`.

    ``tests/`` is not a package, so the helpers are loaded by path under a name
    of this module's own choosing rather than under the test module's name --
    importing e.g. ``test_arena_unit`` a second time under *its* name would
    collide with pytest's own import of the same file in the same run.
    """
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(alias, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load the test helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_helpers = _load_test_helpers("test_arena_unit.py", "arena_unit_helpers")
_bridge_helpers = _load_test_helpers("test_v1bridge_unit.py", "v1bridge_unit_helpers")

DRAW_SCORE: float = _helpers.DRAW_SCORE
"""§D3.3's contempt, shared with the arena's own unit suite."""

TINY_M: int = _helpers.TINY_M
TINY_N: int = _helpers.TINY_N
"""The smallest budget whose first halving round carries two leaves."""

CHEAP_M: int = 1
CHEAP_N: int = 0
"""The no-simulation budget: one root evaluation a ply, real everything else.

Also the tag the forwarding spy tells the two sides apart by -- ``config.m`` is
the searching entrant's own, so a recorded search names its side without the spy
having to be told which evaluator belongs to whom.
"""


def stub_factory(seed: int = 0) -> BatchEvaluator:
    """Return a fresh deterministic stub network, in the factory shape."""
    evaluator: BatchEvaluator = _helpers.StubEvaluator(seed)
    return evaluator


def profiled_entrant(
    name: str,
    seed: int,
    *,
    proof_mode: str = "off",
    discovery_mode: str = "off",
    m: int = TINY_M,
    n: int = TINY_N,
) -> arena.ArenaPlayer:
    """One entrant over its own stub, at `m`/`n`, under the given profile."""
    return arena.ArenaPlayer(
        player=Player(id=name, family="komugi"),
        evaluator_factory=lambda: stub_factory(seed),
        m=m,
        n=n,
        proof_mode=proof_mode,
        discovery_mode=discovery_mode,
    )


def cheap_entrant(
    name: str,
    seed: int,
    *,
    proof_mode: str = "off",
    discovery_mode: str = "off",
) -> arena.ArenaPlayer:
    """An entrant that spends no simulations, for the structural assertions."""
    return profiled_entrant(
        name,
        seed,
        proof_mode=proof_mode,
        discovery_mode=discovery_mode,
        m=CHEAP_M,
        n=CHEAP_N,
    )


def match_config(games: int = 2, seed: int = 11) -> arena.ArenaConfig:
    """One seat-swapped pair at the suite's contempt."""
    return arena.ArenaConfig(games=games, seed=seed, draw_score=DRAW_SCORE)


class SearchCall(NamedTuple):
    """What one ``GumbelSearch`` construction was asked for.

    ``m`` is the tag: the two sides of a match carry different budgets, so a
    recorded call names the side that built it.
    """

    proof_mode: str
    discovery_mode: str
    m: int
    provider_id: str | None


class SearchSpy:
    """A stand-in for ``arena.GumbelSearch`` that records and then delegates.

    Delegates rather than fakes: the match still plays real games with the real
    search, so a forwarding bug cannot hide behind a stub that never searches.
    """

    def __init__(self) -> None:
        self.calls: list[SearchCall] = []

    def __call__(
        self,
        evaluator: Evaluator,
        config: GumbelConfig,
        *,
        wave_evaluator: WaveEvaluator | None = None,
        tactical_provider: TacticalProvider | None = None,
    ) -> GumbelSearch:
        """Record this construction's modes and provider, then build it."""
        provider_id = (
            None if tactical_provider is None else tactical_provider.provider_id
        )
        self.calls.append(
            SearchCall(
                proof_mode=config.proof_mode,
                discovery_mode=config.discovery_mode,
                m=config.m,
                provider_id=provider_id,
            )
        )
        return GumbelSearch(
            evaluator,
            config,
            wave_evaluator=wave_evaluator,
            tactical_provider=tactical_provider,
        )


class DefaultSpy:
    """Records each construction's modes *and whether a provider was passed*.

    Takes ``**kwargs`` rather than the named seam on purpose: a keyword-only
    parameter defaulting to ``None`` cannot tell a caller that passed ``None``
    from one that passed nothing, and "the legacy call shape is preserved" is a
    claim about the argument being absent.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, bool]] = []

    def __call__(
        self, evaluator: Evaluator, config: GumbelConfig, **kwargs: Any
    ) -> GumbelSearch:
        """Record this construction, then build the real search."""
        self.calls.append(
            (
                config.proof_mode,
                config.discovery_mode,
                "tactical_provider" in kwargs,
            )
        )
        return GumbelSearch(evaluator, config, **kwargs)


class LegacyConstruction:
    """Builds every search from a config carrying only the pre-KOM-224 fields.

    The recorded expectation for the default match: a :class:`GumbelConfig`
    rebuilt field by field from the ones that existed before the modes did, and
    no provider seam. Delegates to the real search, so the match it plays is a
    real match -- the one the arena played before this ticket.
    """

    def __call__(
        self,
        evaluator: Evaluator,
        config: GumbelConfig,
        *,
        wave_evaluator: WaveEvaluator | None = None,
        tactical_provider: TacticalProvider | None = None,
    ) -> GumbelSearch:
        """Build the legacy search, refusing a provider outright."""
        if tactical_provider is not None:
            raise AssertionError(
                "the legacy reconstruction was handed a tactical provider"
            )
        legacy = GumbelConfig(
            num_actions=config.num_actions,
            m=config.m,
            n=config.n,
            seed=config.seed,
            c_visit=config.c_visit,
            c_scale=config.c_scale,
            gumbel_scale=config.gumbel_scale,
        )
        return GumbelSearch(evaluator, legacy, wave_evaluator=wave_evaluator)


def bridge_connection() -> V1Connection:
    """Return a ``V1Connection`` stand-in carrying the v1 suite's handshake.

    :func:`~komugi.eval.v1bridge.anchor_configuration` reads nothing but
    ``connection.hello``, so the parsed handshake is the whole of what a stub
    has to be -- and it is built from ``test_v1bridge_unit``'s own conforming
    payload rather than a second copy of it that could drift from the schema.
    """

    class _Stub:
        """Only the one property the configuration block consults."""

        hello: HelloResponse = HelloResponse.from_wire(
            _bridge_helpers.conforming_hello()
        )

    return cast(V1Connection, _Stub())


class ExplodingProvider:
    """A tactical provider that cannot be built.

    Substituted for the production provider to prove the ``off`` path never
    constructs one: a mode that built a provider it would never consult is a
    cost the legacy player must not pay, and an unused object is invisible
    unless building it is fatal.
    """

    provider_id: str = PRODUCTION_PROVIDER_ID

    def __init__(self) -> None:
        raise AssertionError(
            "the arena constructed a tactical provider for a match that "
            "asked for no discovery"
        )


class CountingFactory:
    """An evaluator factory that records whether it was ever called.

    The rated refusal's whole claim is *before any side effect*: weights not
    loaded, evaluator not built, no counted row spent. This is the witness.
    """

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> BatchEvaluator:
        """Count this build and return a stub network."""
        self.calls += 1
        return stub_factory(0)


def unused_mover() -> Mover:
    """A mover factory for entrants that are never played."""
    raise AssertionError("this entrant's mover factory must not be called")


# ---------------------------------------------------------------------------
# T1: the legacy default is untouched
# ---------------------------------------------------------------------------


def test_defaults_are_the_legacy_identity() -> None:
    """Every constructor that grew the modes defaults to the legacy search.

    Built with **no mode arguments at all**: the helpers above pass ``off``
    explicitly, so a test written over them pins the *helper's* defaults and
    would survive a constructor that had changed its own. An existing call site
    passes neither mode, which is what this builds.
    """
    player = arena.ArenaPlayer(
        player=Player(id="alpha", family="komugi"),
        evaluator_factory=lambda: stub_factory(1),
    )
    assert player.search_identity == modes.LEGACY_IDENTITY

    mover = GumbelMover(
        stub_factory,
        draw_score=DRAW_SCORE,
        player_id="ckpt-test",
        m=TINY_M,
        n=TINY_N,
    )
    try:
        assert mover.search_identity == modes.LEGACY_IDENTITY
    finally:
        mover.close()

    entrant = komugi_entrant(
        "ckpt-test", stub_factory, draw_score=DRAW_SCORE, m=TINY_M, n=TINY_N
    )
    assert entrant.search_identity == modes.LEGACY_IDENTITY
    assert modes.LEGACY_IDENTITY.decision_id == "gumbel_sh_survivor_v1"
    assert modes.LEGACY_IDENTITY.target_id == "completed_q_full_legal_v1"


def test_a_default_match_never_forwards_a_mode_or_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No search of a default match was told about either layer.

    The per-search form of the claim: not "the outcome is unchanged", which a
    mode that happened to pick the same moves would also satisfy, but that every
    single ``GumbelSearch`` the match built saw ``off``/``off`` and was handed no
    provider argument at all.
    """
    spy = DefaultSpy()
    monkeypatch.setattr(arena, "GumbelSearch", spy)
    outcome = arena.play_match(
        cheap_entrant("alpha", 1), cheap_entrant("beta", 2), match_config()
    )
    assert len(outcome.records) == 2
    assert spy.calls, "the match must have built at least one search"
    assert set(spy.calls) == {("off", "off", False)}


def test_a_default_match_is_the_pre_kom_224_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default match equals one played by a search built without the modes.

    The expectation is *recorded*, not the same call twice: the second match is
    played through a stand-in that rebuilds each ``GumbelConfig`` from the fields
    that existed before KOM-224 and drops the provider seam, so it is the search
    the arena used to construct. A mode leaking into the default path moves the
    live match and not the reconstruction, and the equality fails. The field list
    is the other half -- the modes are player state, and nothing about them may
    reach a :class:`~komugi.eval.records.GameRecord`, which the rating stack
    parses by field.
    """
    config = match_config()
    played = arena.play_match(
        cheap_entrant("alpha", 1), cheap_entrant("beta", 2), config
    )
    monkeypatch.setattr(arena, "GumbelSearch", LegacyConstruction())
    expected = arena.play_match(
        cheap_entrant("alpha", 1), cheap_entrant("beta", 2), config
    )
    assert played == expected
    assert [field.name for field in dataclasses.fields(GameRecord)] == [
        "white",
        "black",
        "result",
        "book_id",
        "pair_id",
    ]


# ---------------------------------------------------------------------------
# T2: the refusals, at construction
# ---------------------------------------------------------------------------


def test_an_unknown_proof_mode_is_refused_at_construction() -> None:
    """A mode string nobody implements is a wiring error, not a fallback."""
    with pytest.raises(ValueError, match="proof_mode"):
        profiled_entrant("alpha", 1, proof_mode="bogus")


def test_an_unknown_discovery_mode_is_refused_at_construction() -> None:
    """Likewise on the discovery side."""
    with pytest.raises(ValueError, match="discovery_mode"):
        profiled_entrant(
            "alpha", 1, proof_mode="interval_v1", discovery_mode="bogus"
        )


def test_discovery_without_the_proof_layer_is_refused() -> None:
    """Discovery backs terminal results up for the proof layer to consume.

    With ``proof_mode = "off"`` there is nothing to consume them, so the pair is
    refused rather than run as a mode that silently did nothing.
    """
    with pytest.raises(ValueError, match="requires the proof layer"):
        profiled_entrant("alpha", 1, discovery_mode="root_marshal_v1")


# ---------------------------------------------------------------------------
# T3: the forwarding, per side
# ---------------------------------------------------------------------------


def test_each_side_searches_under_its_own_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every search of a match carries the modes of the side that built it.

    Side A runs both layers at the tiny budget; side B is the legacy player at
    the no-simulation budget. ``config.m`` separates them, so this is an
    assertion about *each* search rather than about the set of profiles seen --
    a forwarding bug that handed A's provider to B would pass the weaker form.
    """
    spy = SearchSpy()
    monkeypatch.setattr(arena, "GumbelSearch", spy)
    proof_mode, discovery_mode = PROFILE
    outcome = arena.play_match(
        profiled_entrant(
            "prover", 1, proof_mode=proof_mode, discovery_mode=discovery_mode
        ),
        cheap_entrant("plain", 2),
        match_config(),
    )
    assert outcome.records, "the match must have played at least one game"
    assert spy.calls, "the match must have built at least one search"
    profiled = [call for call in spy.calls if call.m == TINY_M]
    plain = [call for call in spy.calls if call.m == CHEAP_M]
    assert len(profiled) + len(plain) == len(spy.calls)
    assert profiled and plain, "both sides must have searched"
    for call in profiled:
        assert call.proof_mode == proof_mode
        assert call.discovery_mode == discovery_mode
        assert call.provider_id == PRODUCTION_PROVIDER_ID
    for call in plain:
        assert call.proof_mode == "off"
        assert call.discovery_mode == "off"
        assert call.provider_id is None


def test_move_search_forwards_both_modes_and_the_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The seam itself, called directly: neither mode may be dropped.

    The per-side test above would survive a ``move_search`` that forwarded only
    one of the two (the other would still differ between the sides for the wrong
    reason); this one pins each mode independently, and pins that a provider
    handed in reaches the search rather than being rebuilt or discarded.
    """
    spy = SearchSpy()
    monkeypatch.setattr(arena, "GumbelSearch", spy)
    evaluator = make_evaluator(stub_factory(0), DRAW_SCORE)
    wave = arena.make_batch_wave_evaluator(stub_factory(0), DRAW_SCORE)
    provider = GameMarshalCaptureProvider()

    arena.move_search(evaluator, wave, m=TINY_M, n=TINY_N, gumbel_scale=1.0, seed=0)
    arena.move_search(
        evaluator,
        wave,
        m=TINY_M,
        n=TINY_N,
        gumbel_scale=1.0,
        seed=0,
        proof_mode="interval_v1",
    )
    arena.move_search(
        evaluator,
        wave,
        m=TINY_M,
        n=TINY_N,
        gumbel_scale=1.0,
        seed=0,
        proof_mode="interval_v1",
        discovery_mode="root_marshal_v1",
        tactical_provider=provider,
    )

    assert [(call.proof_mode, call.discovery_mode) for call in spy.calls] == [
        ("off", "off"),
        ("interval_v1", "off"),
        ("interval_v1", "root_marshal_v1"),
    ]
    assert [call.provider_id for call in spy.calls] == [
        None,
        None,
        PRODUCTION_PROVIDER_ID,
    ]


# ---------------------------------------------------------------------------
# T4: the ``off`` path never builds a provider
# ---------------------------------------------------------------------------


def test_a_default_match_never_constructs_a_tactical_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With discovery off, the provider class is never touched at all."""
    monkeypatch.setattr(arena, "GameMarshalCaptureProvider", ExplodingProvider)
    outcome = arena.play_match(
        cheap_entrant("alpha", 1), cheap_entrant("beta", 2), match_config()
    )
    assert len(outcome.records) == 2


def test_a_discovery_match_does_construct_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The mirror image: with discovery on, the construction really happens.

    Proved by the same exploding stand-in -- if the wiring stopped building a
    provider the assertion above would still pass, and only this one would fail.
    """
    monkeypatch.setattr(arena, "GameMarshalCaptureProvider", ExplodingProvider)
    proof_mode, discovery_mode = PROFILE
    with pytest.raises(AssertionError, match="asked for no discovery"):
        arena.play_match(
            cheap_entrant(
                "prover", 1, proof_mode=proof_mode, discovery_mode=discovery_mode
            ),
            cheap_entrant("plain", 2),
            match_config(),
        )


def test_a_provider_with_the_wrong_id_is_refused_at_wiring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The arena checks what it built against the published id, first of all.

    The id is what a record claims the discovery layer consulted, so a provider
    reporting anything else is a provenance failure the match must not start on.
    It must also not *pay* for it: the refusal is a wiring error, and the
    counting factory is what says the checkpoint's weights were never loaded to
    discover one -- move the check after ``evaluator_factory()`` and the call
    count fails while the ``raises`` still passes.
    """

    class WrongId:
        """A provider whose id is not the one the arena publishes."""

        provider_id: str = "other-v1"

        def __init__(self) -> None:
            """Build, reporting the wrong identity."""

    factory = CountingFactory()
    proof_mode, discovery_mode = PROFILE
    entrant = arena.ArenaPlayer(
        player=Player(id="prover", family="komugi"),
        evaluator_factory=factory,
        m=CHEAP_M,
        n=CHEAP_N,
        proof_mode=proof_mode,
        discovery_mode=discovery_mode,
    )
    monkeypatch.setattr(arena, "GameMarshalCaptureProvider", WrongId)
    with pytest.raises(ValueError, match=PRODUCTION_PROVIDER_ID):
        arena._wire(entrant, DRAW_SCORE)
    assert factory.calls == 0


# ---------------------------------------------------------------------------
# T5: the rated seam refuses discovery, before anything
# ---------------------------------------------------------------------------


def test_a_rated_mover_refuses_discovery_before_building_its_evaluator() -> None:
    """The refusal costs nothing: the factory is never called.

    §D3.3's ceiling is 257 rows a move and the discovery layer can ask for more,
    so a rated discovery mover would void its own match as M8. The check is the
    constructor's first statement, and the call count is what says so -- move it
    after the factory and this test fails while the ``raises`` still passes.
    """
    factory = CountingFactory()
    proof_mode, discovery_mode = PROFILE
    with pytest.raises(ValueError, match="discovery_mode"):
        GumbelMover(
            factory,
            draw_score=DRAW_SCORE,
            player_id="ckpt-test",
            m=TINY_M,
            n=TINY_N,
            proof_mode=proof_mode,
            discovery_mode=discovery_mode,
        )
    assert factory.calls == 0


def test_a_rated_entrant_refuses_discovery_at_construction() -> None:
    """``komugi_entrant`` refuses too, rather than deferring to ``build()``.

    A match that cannot be played must not be schedulable, so the refusal is at
    the entrant and not at the first mover the harness asks it for.
    """
    factory = CountingFactory()
    proof_mode, discovery_mode = PROFILE
    with pytest.raises(ValueError, match="discovery_mode"):
        komugi_entrant(
            "ckpt-test",
            factory,
            draw_score=DRAW_SCORE,
            m=TINY_M,
            n=TINY_N,
            proof_mode=proof_mode,
            discovery_mode=discovery_mode,
        )
    assert factory.calls == 0


def test_a_rated_mover_accepts_the_proof_layer_and_reports_it() -> None:
    """``interval_v1`` costs no extra rows, so the rated seam may run it."""
    mover = GumbelMover(
        stub_factory,
        draw_score=DRAW_SCORE,
        player_id="ckpt-test",
        m=TINY_M,
        n=TINY_N,
        proof_mode="interval_v1",
    )
    try:
        assert mover.proof_mode == "interval_v1"
        assert mover.discovery_mode == "off"
        assert mover.search_identity.decision_id == "gumbel_sh_interval_v1"
        assert mover.search_identity.target_id == "completed_q_proof_support_v1"
    finally:
        mover.close()


def test_a_rated_entrant_carries_the_proof_layer_into_its_movers() -> None:
    """The entrant's modes are the modes the movers it builds search with."""
    entrant = komugi_entrant(
        "ckpt-test",
        stub_factory,
        draw_score=DRAW_SCORE,
        m=TINY_M,
        n=TINY_N,
        proof_mode="interval_v1",
    )
    assert entrant.proof_mode == "interval_v1"
    assert entrant.discovery_mode == "off"
    mover = entrant.mover_factory()
    try:
        assert isinstance(mover, GumbelMover)
        assert mover.proof_mode == "interval_v1"
        assert mover.discovery_mode == "off"
    finally:
        mover.close()


def test_a_rated_movers_pick_forwards_its_proof_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``pick`` builds the arena's search *with the mover's modes*.

    A mover that reported ``interval_v1`` and searched ``off`` would be rated
    under an algorithm it never played, and the property above cannot see that
    -- only the call can. The row count is asserted too: the proof layer is
    admitted to the rated seam on the claim that it spends no extra counted
    rows, so the move's evaluations must still fit §D3.3's shape of one root
    evaluation plus the simulations actually spent.
    """
    spy = SearchSpy()
    monkeypatch.setattr(arena, "GumbelSearch", spy)
    mover = GumbelMover(
        stub_factory,
        draw_score=DRAW_SCORE,
        player_id="ckpt-test",
        m=TINY_M,
        n=TINY_N,
        proof_mode="interval_v1",
    )
    try:
        action = mover.pick(GameState(Game(), DRAW_SCORE), ply=0, seed=7)
    finally:
        mover.close()
    assert action >= 0
    assert mover.evals_used <= 1 + mover.sims_used
    assert spy.calls, "pick must have built a search"
    for call in spy.calls:
        assert call.proof_mode == "interval_v1"
        assert call.discovery_mode == "off"
        assert call.provider_id is None


# ---------------------------------------------------------------------------
# T6: the bridge entrant carries the modes it was given
# ---------------------------------------------------------------------------


def test_a_bridge_entrants_arena_player_keeps_its_modes() -> None:
    """The serialized, report-facing side says what was played.

    Turning the modes off on the way to the ``ArenaPlayer`` would make an
    entrant's arena-facing identity disagree with the search its movers run.
    """
    entrant = BridgeEntrant(
        player=Player(id="ckpt-test", family=arena.LADDER_FAMILY),
        mover_factory=unused_mover,
        m=TINY_M,
        n=TINY_N,
        proof_mode="interval_v1",
    )
    player = entrant.arena_player()
    assert player.proof_mode == "interval_v1"
    assert player.discovery_mode == "off"
    assert player.search_identity.decision_id == "gumbel_sh_interval_v1"


def test_a_bridge_entrant_defaults_to_the_legacy_profile() -> None:
    """A v1 anchor runs neither layer, and its arena player says ``off``."""
    entrant = BridgeEntrant(
        player=Player(id="v1-anchor", family=arena.LADDER_FAMILY),
        mover_factory=unused_mover,
    )
    assert entrant.proof_mode == "off"
    assert entrant.discovery_mode == "off"
    assert entrant.arena_player().search_identity == modes.LEGACY_IDENTITY


# ---------------------------------------------------------------------------
# T7: the anchor's configuration block names the search that was played
# ---------------------------------------------------------------------------


def test_the_anchor_block_publishes_the_search_the_entrant_played() -> None:
    """§D3.6 P5's block carries both modes and the identity they name.

    ``m``/``n``/``gumbel_scale`` stopped being the whole of "what search was
    played" when the layers arrived: a reader who sees a budget and no algorithm
    attributes a proof-layer row to the legacy search, which is the misreading
    the configuration block exists to prevent.
    """
    entrant = komugi_entrant(
        "ckpt-test",
        stub_factory,
        draw_score=DRAW_SCORE,
        m=TINY_M,
        n=TINY_N,
        proof_mode="interval_v1",
    )
    block = anchor_configuration(entrant, bridge_connection())
    assert block["proof_mode"] == "interval_v1"
    assert block["discovery_mode"] == "off"
    assert block["decision_id"] == "gumbel_sh_interval_v1"
    assert block["target_id"] == "completed_q_proof_support_v1"


def test_the_anchor_block_of_a_default_entrant_is_the_legacy_search() -> None:
    """An entrant built without the modes publishes the legacy identity.

    The frozen v1 anchors were recorded before the layers existed, so the block
    a default entrant produces must say ``off``/``off`` and the survivor rule --
    anything else would relabel an archived anchor as a search it never ran.
    """
    entrant = komugi_entrant(
        "ckpt-test", stub_factory, draw_score=DRAW_SCORE, m=TINY_M, n=TINY_N
    )
    block = anchor_configuration(entrant, bridge_connection())
    assert block["proof_mode"] == "off"
    assert block["discovery_mode"] == "off"
    assert block["decision_id"] == "gumbel_sh_survivor_v1"
    assert block["target_id"] == "completed_q_full_legal_v1"


# ---------------------------------------------------------------------------
# the provider the arena builds is the one the search will accept
# ---------------------------------------------------------------------------


def test_the_arenas_provider_satisfies_the_tactical_seam() -> None:
    """Structural, checked by ``mypy --strict`` on the annotation below."""
    provider: TacticalProvider = GameMarshalCaptureProvider()
    assert provider.provider_id == PRODUCTION_PROVIDER_ID
    state: SearchState = GameState(Game(), DRAW_SCORE)
    provider.validate_state(state)
