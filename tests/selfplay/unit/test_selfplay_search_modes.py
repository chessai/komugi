"""The self-play caller path for the two search modes (KOM-223, U4).

:class:`~komugi.config.SelfPlayConfig` is where a campaign *declares* which
search its games were played with, and :class:`~komugi.selfplay.worker.
SelfPlayWorker` is the only thing that turns that declaration into searches. The
two claims this file pins are therefore:

* **Off is untouched.** At the defaults the config validates exactly as before,
  its identity is :data:`~komugi.search.modes.LEGACY_IDENTITY`, and the worker
  never constructs -- never so much as names -- a tactical provider.
* **On reaches every move.** With the modes set, both of them are forwarded on
  the fast path and on the full path, and the search is handed a provider whose
  identity is the one generator self-play is allowed to discover with.

Plus the crossing the orchestrator forces: it pickles the config into every
spawned worker, so a worker respawned after a code update unpickles an OLD
payload with NEW code. That is a *transport* contract, not an internal detail,
and it is pinned by hand-built payloads rather than only by round-tripping this
build's own bytes -- a round trip alone would pass for any pair of methods that
are inverses of each other, including a pair that cannot read yesterday's.

Everything here is CPU-only and in-process: a stub evaluator, an engine
shortened to a few plies, real shard writers under ``tmp_path``. No weights, no
GPU, no spawned worker, no campaign root.
"""

from __future__ import annotations

import dataclasses
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt
import pytest

from komugi.config import SelfPlayConfig, _config_getstate
from komugi.engine.game import Game
from komugi.engine.repetition import RepetitionRule
from komugi.search.gumbel import GumbelConfig, GumbelSearch
from komugi.search.modes import LEGACY_IDENTITY
from komugi.search.state import AnyEvaluator, AnyWaveEvaluator, SearchState
from komugi.search.tactical import TacticalProvider
from komugi.selfplay.inference import NetEval
from komugi.selfplay.records import SCHEMA, num_actions
from komugi.selfplay.worker import SelfPlayWorker, WorkerConfig
from komugi.train.shards import ShardWriter

PROVIDER_ID: Final[str] = "game-marshal-capture-v1"
"""The identity the production self-play provider must carry (KOM-220)."""

DISCOVERY_MODES: Final[tuple[str, ...]] = ("root_marshal_v1", "root_win_v1")
"""Every discovery mode a campaign may declare, other than ``off`` (KOM-235 F3).

The caller path is mode-agnostic by construction -- the config validates a name,
the worker forwards it, the search selects on it -- so every pin from the config
through the worker to the search runs under **both**, and a mode that the
campaign could select but the worker silently dropped fails here rather than in
a generation's provenance."""

_SHORT_GAME_PLIES: Final[int] = 6
"""Turn cap for the in-test engine subclass, as in ``test_worker_unit.py``.

Shortening the *engine* rather than teaching the worker a test-only ply cap:
termination is the engine's concern, and the worker has no cap of its own (F15).
"""

_RUN_SEED: Final[int] = 20260917

LEGACY_VALUES: Final[tuple[object, ...]] = (
    3,  # workers
    5,  # process_batch_size
    16,  # gumbel_m_full
    64,  # gumbel_n_full
    4,  # gumbel_m_fast
    8,  # gumbel_n_fast
    0.5,  # prob_fast
    0.0,  # draw_score
    2,  # games_in_flight
    "",  # book_path
    0.0,  # book_fraction
    0.5,  # fast_gumbel_scale
    2.0,  # draft_gumbel_scale
    RepetitionRule.FULL,  # repetition_rule
    4,  # min_placements
    0.05,  # mlh_utility_lambda
    0.2,  # mlh_gate_tau0
    0.4,  # mlh_gate_tau1
    24,  # mlh_slope_plies
)
"""A payload in the shape a pre-KOM-223 parent process sends: 19 fields.

Deliberately non-default in every field that has a default, so "every
transported value survived" is a claim about the values rather than about a
config that happens to agree with the constructor. ``draw_score`` is 0.0 because
this payload turns the moves-left term on, which is pre-registered at contempt 0.
"""


# --------------------------------------------------------------------------
# Rig: a deterministic stub evaluator and a four-ply engine
# --------------------------------------------------------------------------


class _StubEvaluator:
    """A deterministic batch evaluator: uniform logits, fixed (w, d, l)."""

    def __init__(self, actions: int) -> None:
        self.actions = actions
        self.calls = 0

    def evaluate(
        self, obs: npt.NDArray[np.float32], legal: npt.NDArray[np.bool_]
    ) -> NetEval:
        """Return one evaluation per row of `obs`."""
        rows = int(obs.shape[0])
        self.calls += 1
        return NetEval(
            policy_logits=np.zeros((rows, self.actions), dtype=np.float32),
            wdl=np.tile(np.asarray((0.4, 0.2, 0.4), dtype=np.float32), (rows, 1)),
            moves_left=np.zeros(rows, dtype=np.float32),
        )


class _ShortGame(Game):
    """The real engine, adjudicating a draw after :data:`_SHORT_GAME_PLIES`."""

    @staticmethod
    def max_turns() -> int:
        """Return the shortened turn cap."""
        return _SHORT_GAME_PLIES


def _selfplay(
    prob_fast: float, proof_mode: str = "off", discovery_mode: str = "off"
) -> SelfPlayConfig:
    """The smallest search that still has a shape, with the given modes."""
    return SelfPlayConfig(
        workers=1,
        process_batch_size=1,
        gumbel_m_full=2,
        gumbel_n_full=2,
        gumbel_m_fast=2,
        gumbel_n_fast=2,
        prob_fast=prob_fast,
        draw_score=0.2,
        games_in_flight=1,
        proof_mode=proof_mode,
        discovery_mode=discovery_mode,
    )


def _worker(directory: Path, selfplay: SelfPlayConfig) -> SelfPlayWorker:
    """Build a worker over a real shard writer in `directory`."""
    directory.mkdir(parents=True, exist_ok=True)
    writer = ShardWriter(directory, SCHEMA)
    return SelfPlayWorker(
        _StubEvaluator(num_actions()),
        selfplay,
        WorkerConfig(
            worker_id=1,
            incarnation=0,
            run_seed=_RUN_SEED,
            games=1,
            shard_dir=str(directory),
        ),
        writer,
    )


def _play(
    directory: Path, selfplay: SelfPlayConfig, monkeypatch: pytest.MonkeyPatch
) -> int:
    """Play game 0 into `directory`, returning the number of rows written."""
    monkeypatch.setattr("komugi.selfplay.worker.Game", _ShortGame)
    worker = _worker(directory, selfplay)
    try:
        return worker.play_game(0)
    finally:
        worker._writer.close()


@dataclass(frozen=True)
class _Seen:
    """What one :class:`GumbelSearch` construction was asked for."""

    proof_mode: str
    discovery_mode: str
    provider: TacticalProvider | None


def _spy_on_search(monkeypatch: pytest.MonkeyPatch) -> list[_Seen]:
    """Record every search construction, then build the real search.

    A spy that *delegates* rather than a stub that returns a canned result: the
    worker must still play a real game through the real search, or "the modes
    were forwarded" would be true of a worker whose searches never ran.
    """
    seen: list[_Seen] = []
    real_search = GumbelSearch

    def spy(
        evaluator: AnyEvaluator,
        config: GumbelConfig,
        *,
        wave_evaluator: AnyWaveEvaluator | None = None,
        tactical_provider: TacticalProvider | None = None,
    ) -> GumbelSearch:
        """Record what the worker asked for, then build the real search."""
        seen.append(
            _Seen(config.proof_mode, config.discovery_mode, tactical_provider)
        )
        return real_search(
            evaluator,
            config,
            wave_evaluator=wave_evaluator,
            tactical_provider=tactical_provider,
        )

    monkeypatch.setattr("komugi.selfplay.worker.GumbelSearch", spy)
    return seen


class _ExplodingProvider:
    """A provider that cannot be constructed -- the off-path tripwire."""

    provider_id: str = PROVIDER_ID

    def __init__(self) -> None:
        raise AssertionError(
            "a tactical provider was constructed with discovery off"
        )


class _MisnamedProvider:
    """A provider whose identity is not the one self-play is allowed to use."""

    provider_id: str = "other-v1"

    def validate_state(self, state: SearchState) -> None:
        """Unreachable: construction of this worker is refused."""
        raise AssertionError("misnamed provider was consulted")


# --------------------------------------------------------------------------
# T_CONFIG: the fields, their validation, and the derived identity
# --------------------------------------------------------------------------


def test_the_default_config_is_the_legacy_search() -> None:
    """Off/off, and the identity of the search this project has always run."""
    config = SelfPlayConfig(workers=1, process_batch_size=1)
    assert config.proof_mode == "off"
    assert config.discovery_mode == "off"
    assert config.search_identity == LEGACY_IDENTITY


def test_an_unknown_proof_mode_is_refused_at_construction() -> None:
    """A typo is a ValueError here, not a campaign that ran something else."""
    with pytest.raises(ValueError, match="proof_mode"):
        SelfPlayConfig(workers=1, process_batch_size=1, proof_mode="bogus")


def test_discovery_without_the_proof_layer_is_refused() -> None:
    """Discovery backs terminal results up for the proof layer to consume.

    With ``proof_mode = "off"`` there is nothing to consume them, so the pair is
    refused where it is configured rather than recorded as a layer that ran and
    changed nothing.
    """
    with pytest.raises(ValueError, match="discovery_mode"):
        SelfPlayConfig(
            workers=1,
            process_batch_size=1,
            discovery_mode="root_marshal_v1",
        )


@pytest.mark.parametrize("discovery_mode", DISCOVERY_MODES)
def test_both_modes_on_name_the_proof_search(discovery_mode: str) -> None:
    """The identity is derived from the pair -- decision and target together."""
    config = SelfPlayConfig(
        workers=1,
        process_batch_size=1,
        proof_mode="interval_v1",
        discovery_mode=discovery_mode,
    )
    identity = config.search_identity
    assert identity.proof_mode == "interval_v1"
    assert identity.discovery_mode == discovery_mode
    assert identity.decision_id == "gumbel_sh_interval_v1"
    assert identity.target_id == "completed_q_proof_support_v1"


def test_replacing_the_proof_mode_preserves_every_other_field() -> None:
    """``dataclasses.replace`` is how a campaign turns the layer on."""
    config = _selfplay(0.25)
    turned_on = dataclasses.replace(config, proof_mode="interval_v1")
    assert turned_on.proof_mode == "interval_v1"
    assert turned_on.discovery_mode == "off"
    for field in dataclasses.fields(config):
        if field.name == "proof_mode":
            continue
        assert getattr(turned_on, field.name) == getattr(config, field.name), (
            f"replace() changed {field.name}"
        )


# --------------------------------------------------------------------------
# T_PICKLE: the payload a parent process sends a respawned worker
# --------------------------------------------------------------------------


@pytest.mark.parametrize("discovery_mode", DISCOVERY_MODES)
def test_a_pickle_round_trip_preserves_explicit_modes(discovery_mode: str) -> None:
    """The ordinary transport: this build's bytes, read by this build."""
    config = _selfplay(0.25, "interval_v1", discovery_mode)
    restored = pickle.loads(pickle.dumps(config))
    assert restored == config
    assert restored.proof_mode == "interval_v1"
    assert restored.discovery_mode == discovery_mode
    assert restored.search_identity == config.search_identity


def test_a_legacy_payload_is_read_as_the_search_it_was_produced_by() -> None:
    """A 19-field payload predates the modes, so it is off/off -- exactly right.

    This is the respawn crossing: NEW code, OLD payload. Every transported value
    must survive, because the alternative is a worker playing a generation under
    a configuration nobody chose.
    """
    config = SelfPlayConfig.__new__(SelfPlayConfig)
    config.__setstate__(LEGACY_VALUES)

    assert config.proof_mode == "off"
    assert config.discovery_mode == "off"
    assert config.search_identity == LEGACY_IDENTITY

    names = [f.name for f in dataclasses.fields(SelfPlayConfig)]
    for name, value in zip(names, LEGACY_VALUES, strict=False):
        assert getattr(config, name) == value, f"{name} did not survive transport"


@pytest.mark.parametrize("discovery_mode", DISCOVERY_MODES)
def test_a_21_field_payload_reads_as_cap_adjudication_off(discovery_mode: str) -> None:
    """A KOM-223-era payload predates KOM-249's four, so it carries their defaults.

    The second respawn crossing (KOM-245 review F1). A parent running the code
    that appended the two search modes -- and nothing after them -- sends 21
    values; this build has 25 fields, and padding the gap with anything but the
    defaults would be a worker playing a cap-adjudication rule its parent never
    configured. The modes it *does* carry must still arrive intact, which is
    what pins that the four were appended rather than inserted.
    """
    config = SelfPlayConfig.__new__(SelfPlayConfig)
    config.__setstate__((*LEGACY_VALUES, "interval_v1", discovery_mode))

    assert (config.proof_mode, config.discovery_mode) == ("interval_v1", discovery_mode)
    assert config.cap_adjudication == "off"
    assert config.cap_adjudication_p_max == 0.8
    assert config.cap_adjudication_d_full == 8
    assert config.min_placements_max == 0

    names = [f.name for f in dataclasses.fields(SelfPlayConfig)]
    for name, value in zip(names, LEGACY_VALUES, strict=False):
        assert getattr(config, name) == value, f"{name} did not survive transport"


@pytest.mark.parametrize("discovery_mode", DISCOVERY_MODES)
def test_a_25_field_payload_reads_as_seed_openings_off(discovery_mode: str) -> None:
    """A KOM-249-era payload predates KOM-254's two, so it carries their defaults.

    The third respawn crossing. A parent running the code that appended
    KOM-249's four -- and nothing after them -- sends 25 values; this build has
    27 fields, and ``("", 0.0)`` is not a chosen default but the configuration
    that parent was actually running: no seed file, no seeded share. Padding
    with anything else would hand a respawned worker a conversion curriculum
    nobody configured, and at ``prob > 0`` with a path the worker would even
    load a file the parent never named.
    """
    config = SelfPlayConfig.__new__(SelfPlayConfig)
    config.__setstate__(
        (*LEGACY_VALUES, "interval_v1", discovery_mode, "off", 0.8, 8, 0)
    )

    assert (config.proof_mode, config.discovery_mode) == ("interval_v1", discovery_mode)
    assert config.seed_openings_path == ""
    assert config.seed_openings_prob == 0.0

    names = [f.name for f in dataclasses.fields(SelfPlayConfig)]
    for name, value in zip(names, LEGACY_VALUES, strict=False):
        assert getattr(config, name) == value, f"{name} did not survive transport"


@pytest.mark.parametrize("discovery_mode", DISCOVERY_MODES)
def test_a_current_payload_restores_the_modes_it_carries(discovery_mode: str) -> None:
    """The 27-field payload is positional too, and validated on arrival.

    Built by ``_config_getstate`` rather than written out as a literal: the
    payload this build *sends* is the one it must be able to read, and a literal
    would only pin the length a previous author believed in.
    """
    source = dataclasses.replace(
        _selfplay(0.25, "interval_v1", discovery_mode),
        cap_adjudication="material",
        cap_adjudication_p_max=0.7,
        cap_adjudication_d_full=6,
        min_placements_max=0,
    )
    payload = _config_getstate(source)
    assert len(payload) == 27, "the transport's current shape is the field count"

    config = SelfPlayConfig.__new__(SelfPlayConfig)
    config.__setstate__(payload)
    assert config == source
    assert config.proof_mode == "interval_v1"
    assert config.discovery_mode == discovery_mode
    assert (config.cap_adjudication, config.cap_adjudication_p_max) == (
        "material",
        0.7,
    )
    assert config.cap_adjudication_d_full == 6


@pytest.mark.parametrize(
    "state",
    [
        pytest.param((*LEGACY_VALUES, "off"), id="twenty"),
        pytest.param(LEGACY_VALUES[:-1], id="eighteen"),
        pytest.param((*LEGACY_VALUES, "off", "off", "off"), id="twenty-two"),
        pytest.param((*LEGACY_VALUES, "off", "off", "off", 0.8), id="twenty-three"),
        pytest.param((*LEGACY_VALUES, "off", "off", "off", 0.8, 8), id="twenty-four"),
        pytest.param(
            (*LEGACY_VALUES, "off", "off", "off", 0.8, 8, 0, 0), id="twenty-six"
        ),
    ],
)
def test_a_payload_of_an_unknown_length_is_refused(state: tuple[object, ...]) -> None:
    """Not padded, not truncated: a shape this build cannot read is an error.

    A 20-tuple is the tempting one -- "one mode is missing, default it" -- and it
    is exactly what must not happen: the fields are positional, so a payload one
    short is a payload whose *every* value past the gap may be misassigned. 22,
    23 and 24 are the same temptation for KOM-249's four: they are one lever
    each way, so a partial arrival is a payload built by a build nobody here
    knows, not a KOM-249 payload with fields to default. 26 is the same rule in
    the other direction -- a payload from a LATER build, whose extra field this
    one cannot name and must not silently drop.

    The message is pinned, not just the exception: it is what an operator reads
    when a respawn fails, and the three lengths it names are the whole transport
    contract. ``\\d+`` rather than a literal so the pin is on the arms, not on
    which case of this table produced it.
    """
    config = SelfPlayConfig.__new__(SelfPlayConfig)
    with pytest.raises(
        ValueError, match=r"27 field values \(or the legacy 25, 21 or 19\), got \d+"
    ):
        config.__setstate__(state)


@pytest.mark.parametrize(
    ("state", "field"),
    [
        pytest.param(
            (*LEGACY_VALUES, "off", "off", "material", 1.5, 8, 0),
            "cap_adjudication_p_max",
            id="p-above-one",
        ),
        pytest.param(
            (*LEGACY_VALUES, "off", "off", "off", 0.8, 8, 3),
            "min_placements_max",
            id="max-below-low-edge",
        ),
    ],
)
def test_a_bad_cap_knob_is_refused_on_arrival(
    state: tuple[object, ...], field: str
) -> None:
    """A legacy-25 payload is still validated: the wire is not an exemption.

    25 is the KOM-249 legacy arm, so these payloads are padded to 27 and *then*
    validated -- ``__setstate__`` ends in ``__post_init__`` for exactly that
    reason: a worker respawned with a ramp nobody can play must die at
    construction rather than at its first capped game, hours into a campaign,
    with a shard of mislabelled rows already on disk. The second case is the
    cross-field rule: ``min_placements_max=3`` is below this payload's own
    ``min_placements`` of 4, an empty draw range that only a check reading both
    fields can catch.
    """
    config = SelfPlayConfig.__new__(SelfPlayConfig)
    assert len(state) == 25, (
        "this node is the legacy-25 arm reaching validation; an unknown length "
        "would be refused for its length before any field was validated"
    )
    with pytest.raises(ValueError, match=field):
        config.__setstate__(state)


def test_a_payload_carrying_an_invalid_mode_is_refused() -> None:
    """Arriving over the wire does not exempt a pair from validation."""
    config = SelfPlayConfig.__new__(SelfPlayConfig)
    with pytest.raises(ValueError, match="proof_mode"):
        config.__setstate__((*LEGACY_VALUES, "bogus", "off"))


def test_a_non_tuple_payload_is_refused() -> None:
    """Including the dict a default slotless dataclass would have sent."""
    config = SelfPlayConfig.__new__(SelfPlayConfig)
    with pytest.raises(ValueError, match="tuple"):
        config.__setstate__({"workers": 1})


# --------------------------------------------------------------------------
# T_WORKER_FORWARDS / T_MUTATION_GUARDS: what the worker hands the search
# --------------------------------------------------------------------------


@pytest.mark.parametrize("discovery_mode", DISCOVERY_MODES)
@pytest.mark.parametrize(
    "prob_fast", [1.0, 0.0], ids=["fast-path", "full-path"]
)
def test_the_worker_forwards_both_modes_and_the_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prob_fast: float,
    discovery_mode: str,
) -> None:
    """Both modes and the provider reach EVERY move, fast schedule and full.

    Parametrised over the two schedules rather than trusting one: ``prob_fast``
    of 1.0 makes every ply fast and 0.0 makes every ply full, so a mode dropped
    from either branch of ``_gumbel_config`` fails here with the branch named.
    These are the KOM-223 mutation guards (a) and (b); (c) is the provider
    assertion below, which no forwarded-mode check would catch.
    """
    seen = _spy_on_search(monkeypatch)
    rows = _play(
        tmp_path / "on",
        _selfplay(prob_fast, "interval_v1", discovery_mode),
        monkeypatch,
    )

    assert rows > 0
    assert len(seen) >= rows, "the spy saw fewer searches than the game had plies"
    for index, call in enumerate(seen):
        assert call.proof_mode == "interval_v1", f"move {index} lost proof_mode"
        assert call.discovery_mode == discovery_mode, (
            f"move {index} lost discovery_mode"
        )
        # Guard (c): the modes can be forwarded perfectly and the seam still be
        # empty, which the search would then refuse -- but only at run time.
        assert call.provider is not None, f"move {index} got no tactical provider"
        assert call.provider.provider_id == PROVIDER_ID


def test_a_default_worker_hands_the_search_no_modes_and_no_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same rig at the defaults: off, off, and nothing attached."""
    seen = _spy_on_search(monkeypatch)
    rows = _play(tmp_path / "off", _selfplay(0.5), monkeypatch)

    assert rows > 0
    assert seen, "the worker ran no searches"
    assert all(
        (call.proof_mode, call.discovery_mode, call.provider) == ("off", "off", None)
        for call in seen
    ), f"a default worker asked for something other than the legacy search: {seen}"


# --------------------------------------------------------------------------
# T_OFF_NEVER_CONSTRUCTS / T_WRONG_PROVIDER_ID
# --------------------------------------------------------------------------


def test_the_off_path_never_constructs_a_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Off does not merely ignore the provider: it never builds one.

    The tripwire is the provider's own ``__init__``, so this is a claim about
    the worker's control flow rather than about what it stored -- "constructed
    it and dropped it" and "never constructed it" are indistinguishable from the
    attribute alone, and only the second keeps the off path free.
    """
    monkeypatch.setattr(
        "komugi.selfplay.worker.GameMarshalCaptureProvider", _ExplodingProvider
    )
    worker = _worker(tmp_path / "off", _selfplay(0.5))
    worker._writer.close()

    with pytest.raises(AssertionError, match="discovery off"):
        _worker(tmp_path / "on", _selfplay(0.5, "interval_v1", "root_marshal_v1"))


def test_a_provider_with_the_wrong_identity_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The worker checks the generator's identity rather than assuming it.

    The identity travels into every search's diagnostics and from there into a
    run's provenance, so a provider nobody recognises must stop the worker at
    construction rather than be recorded as the one that was expected.
    """
    monkeypatch.setattr(
        "komugi.selfplay.worker.GameMarshalCaptureProvider", _MisnamedProvider
    )
    with pytest.raises(ValueError, match="other-v1"):
        _worker(tmp_path / "wrong", _selfplay(0.5, "interval_v1", "root_marshal_v1"))


@pytest.mark.parametrize("discovery_mode", DISCOVERY_MODES)
def test_the_worker_holds_the_production_provider_when_discovery_is_on(
    tmp_path: Path, discovery_mode: str
) -> None:
    """Unmonkeypatched: the real class, with the identity KOM-220 published.

    Under either discovery mode: ``root_win_v1`` asks the provider nothing at the
    root, but it still verifies terminals through it and still records it as the
    generator, so a worker that built one only for ``root_marshal_v1`` would fail
    at the first search rather than here.
    """
    worker = _worker(tmp_path / "real", _selfplay(0.5, "interval_v1", discovery_mode))
    try:
        provider: Any = worker._tactical_provider
        assert provider is not None
        assert provider.provider_id == PROVIDER_ID
    finally:
        worker._writer.close()


# -- review finding F1 (2026-09-17): the REAL legacy payload is a list, built here with
# the exact opcode shape the old class produced (NEWOBJ, then BUILD with a list).
import copyreg  # noqa: E402
import io  # noqa: E402
import pickle  # noqa: E402


def _legacy_values() -> list[object]:
    cfg = SelfPlayConfig(workers=3, process_batch_size=2, gumbel_n_full=64, draw_score=0.1)
    names = [f.name for f in dataclasses.fields(SelfPlayConfig)][:19]
    return [getattr(cfg, n) for n in names]


class _LegacyPickler(pickle.Pickler):
    def reducer_override(self, obj: object) -> Any:  # the Pickler hook is loosely typed
        if isinstance(obj, SelfPlayConfig):
            newobj = getattr(copyreg, "__newobj__")  # typeshed hides pickle's NEWOBJ helper
            return (newobj, (SelfPlayConfig,), _legacy_values())
        return NotImplemented


def test_a_list_shaped_legacy_payload_from_the_old_class_restores_as_off_off() -> None:
    """A pre-KOM-223 parent pickles a 19-element LIST; a respawned worker must read it."""
    buf = io.BytesIO()
    _LegacyPickler(buf, protocol=4).dump(SelfPlayConfig(workers=1, process_batch_size=1))
    restored = pickle.loads(buf.getvalue())
    assert (restored.proof_mode, restored.discovery_mode) == ("off", "off")
    names = [f.name for f in dataclasses.fields(SelfPlayConfig)][:19]
    assert all(getattr(restored, n) == v for n, v in zip(names, _legacy_values()))
    assert restored.workers == 3 and restored.gumbel_n_full == 64
