"""KOM-234: ``discovery_mode = "root_win_v1"``, the exhaustive root win scan.

The KOM-232 census of the first post-resume iteration is what this module exists
for. With ``root_marshal_v1`` live, every available marshal **capture** was taken
(31/31) -- and 100% of the immediate wins still missed were **checkmates**: 51 by
drops, 16 by ordinary moves, 4 by stacking. The engine adjudicates checkmate as a
win, but the KOM-222 provider enumerates ordinary captures only, so no
checkmating move was ever a root candidate and the survey never verified one.
``root_win_v1`` is the repair: at the ROOT, every offered legal action is
verified by clone/play, so a win of any move type is found. The reply
(anti-decisive) half stays provider-based, because an exhaustive reply scan is
``|C| x |replies|`` transitions per candidate.

The fixtures are the KOM-219/KOM-220 toys, loaded by path so there is exactly one
copy of them, and the claims are:

* **T1** a toy whose only immediate win is invisible to the provider -- the
  stand-in for a checkmate -- is found, played and witnessed under
  ``root_win_v1`` at every seed, prior, budget and schedule, with **no provider
  call at the root**; and on the same toy ``root_marshal_v1`` does not find it.
  That pair is the whole ticket: the second half is what makes the first half a
  difference rather than a restatement;
* **T2/T3** where the provider already sees everything -- the C1 capture toy, the
  C2 interior refutation -- the two modes return the *same search*: same action,
  same visits, same target, same diagnostics apart from the three fields that
  describe how the root list was built;
* **T4** the two schedules agree under the new mode as they do under the old one;
* **T5** the scan is free in network rows and bounded in transitions;
* **T6** ``off`` and proof-only are untouched, and a bypassed root still names the
  rule it would have run under;
* **T7** a poisoned ``terminal_winner`` is refused before a single backup, in the
  exhaustive scan too -- the scan widens what is *verified*, never what is
  *trusted*.

Nothing here loads a checkpoint, touches the engine or leaves the CPU.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Final, cast

import numpy as np
import numpy.typing as npt
import pytest

from komugi.search import GumbelConfig, GumbelSearch, SearchResult, SearchState
from komugi.search import gumbel


def _load_discovery_fixtures() -> ModuleType:
    """Import the KOM-220 discovery module by path, under its own module name.

    ``tests/`` is not a package, so there is no import path to it; loading it
    under a name of its own leaves pytest's collection of the real file alone.
    That module in turn loads the KOM-219 coverage fixtures the same way, so
    this single hop reaches both toys and both providers -- there is still
    exactly one definition of each.
    """
    path = Path(__file__).with_name("test_gumbel_tactical_discovery.py")
    spec = importlib.util.spec_from_file_location("_kom220_discovery_fixtures", path)
    assert spec is not None and spec.loader is not None, f"cannot load {path}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_DISCOVERY: Final[ModuleType] = _load_discovery_fixtures()

_CoverageState: Any = _DISCOVERY._CoverageState
"""The C1/C2 toy position (``Any``: it arrives through a by-path import)."""

_CoverageEvaluator: Any = _DISCOVERY._CoverageEvaluator
"""The two-prior network stub, with its call counter."""

_QuietState: Any = _DISCOVERY._QuietState
"""The no-tactics toy, whose ``"forced"`` kind has one legal root action."""

_QuietEvaluator: Any = _DISCOVERY._QuietEvaluator
"""The quiet toy's network stub."""

ToyProvider: Any = _DISCOVERY.ToyProvider
"""The provider that knows where each toy's win is."""

EmptyProvider: Any = _DISCOVERY.EmptyProvider
"""The provider with nothing to say, for the quiet toy."""

PoisonProvider: Any = _DISCOVERY.PoisonProvider
"""The provider that breaks exactly one clause of the contract."""

_BackupLog: Any = _DISCOVERY._BackupLog
"""Every backup of a search, as ``(root edge, delta)``."""

_WIDTH: Final[int] = int(_DISCOVERY._WIDTH)
_ROOT_LEGAL: Final[int] = int(_DISCOVERY._ROOT_LEGAL)
_ROOT_WIN: Final[int] = int(_DISCOVERY._ROOT_WIN)
_INTERIOR_WIN: Final[int] = int(_DISCOVERY._INTERIOR_WIN)
_BUDGETS: Final[tuple[tuple[int, int], ...]] = tuple(
    (int(m), int(n)) for m, n in _DISCOVERY._BUDGETS
)
_SEEDS: Final[tuple[int, ...]] = tuple(int(seed) for seed in _DISCOVERY._SEEDS)
_PRIORS: Final[tuple[str, ...]] = tuple(str(policy) for policy in _DISCOVERY._PRIORS)

_INTERIOR_LEGAL: Final[int] = 2
"""Legal actions at the C2 interior toy's root, which is where the scan runs."""


# --------------------------------------------------------------------------
# The one new provider: a win the provider cannot name
# --------------------------------------------------------------------------


class SilentRootProvider:
    """``ToyProvider`` with its ROOT answer removed: the KOM-232 checkmate.

    The census's missed wins were checkmates -- legal, immediately winning, and
    *not captures*, so ``GameMarshalCaptureProvider`` never lists them. This toy
    provider reproduces exactly that shape without inventing a new position:
    action :data:`_ROOT_WIN` still wins on the spot and ``terminal_winner`` still
    reads the fixture's frame correctly, but ``candidate_actions`` returns
    nothing at the root. A provider-driven root survey therefore has nothing to
    verify; an exhaustive one plays the win anyway.

    It delegates rather than subclasses (the by-path import gives ``Any``, and a
    class hierarchy built on one is a hierarchy the type checker cannot read), so
    its replies and its winner derivation are the honest ones and the child
    survey is the same survey in both modes -- which is the point of leaving it
    alone.
    """

    provider_id = "toy-silent-root-v1"

    def __init__(self) -> None:
        self._inner: Any = ToyProvider()

    def validate_state(self, state: SearchState) -> None:
        """Refuse anything that is not a KOM-219 coverage fixture."""
        self._inner.validate_state(state)

    def candidate_actions(
        self, state: SearchState, legal_mask: npt.NDArray[np.bool_]
    ) -> tuple[int, ...]:
        """Say nothing about a C1 root position; answer honestly anywhere else."""
        self.validate_state(state)
        if str(cast(Any, state).kind) == "root":
            return ()
        return cast(
            "tuple[int, ...]", self._inner.candidate_actions(state, legal_mask)
        )

    def terminal_winner(self, terminal_state: SearchState) -> int | None:
        """Read the winner off the fixture, exactly as ``ToyProvider`` does."""
        return cast("int | None", self._inner.terminal_winner(terminal_state))


# --------------------------------------------------------------------------
# Wiring
# --------------------------------------------------------------------------


def _config(
    *,
    m: int,
    n: int,
    seed: int,
    proof: str = "interval_v1",
    discovery: str = "root_win_v1",
) -> GumbelConfig:
    """Return a toy-sized config with both optional layers set explicitly."""
    return GumbelConfig(
        num_actions=_WIDTH,
        m=m,
        n=n,
        seed=seed,
        proof_mode=proof,
        discovery_mode=discovery,
    )


def _search_toy(
    kind: str,
    *,
    policy: str,
    m: int,
    n: int,
    seed: int,
    wave: bool,
    discovery: str = "root_win_v1",
    provider: Any = None,
) -> tuple[SearchResult, Any]:
    """Search a fresh coverage fixture under `discovery`, returning the evaluator."""
    evaluator = _CoverageEvaluator(policy)
    search = GumbelSearch(
        evaluator,
        _config(m=m, n=n, seed=seed, discovery=discovery),
        wave_evaluator=evaluator.wave if wave else None,
        tactical_provider=ToyProvider() if provider is None else provider,
    )
    return search.search(_CoverageState(kind)), evaluator


def _diagnostics(result: SearchResult) -> gumbel.SearchDiagnostics:
    """Return `result`'s diagnostics, refusing the ``None`` an active layer cannot give."""
    assert result.diagnostics is not None, "an active layer reported no diagnostics"
    return result.diagnostics


_ROOT_LIST_FIELDS: Final[frozenset[str]] = frozenset(
    {"identity", "root_scan", "provider_calls", "candidate_verification_transitions"}
)
"""The diagnostics fields that describe HOW the root list was built.

Everything outside this set is a fact about the search the two discovery modes
ran -- what it attached, what it spent on the network, what it played -- and
T2/T3 assert those are equal. Inside it: ``identity`` carries the mode name by
construction; ``root_scan`` is the new field; ``provider_calls`` drops the root
call under ``root_win_v1``; and ``candidate_verification_transitions`` differs
because the scan reaches the *same* witness after a different number of probes
(the provider named it directly, the scan walked down to it in score order).
"""


def _mode_invariant(diagnostics: gumbel.SearchDiagnostics) -> dict[str, object]:
    """Return the diagnostics fields the two discovery modes must agree on."""
    return {
        field.name: getattr(diagnostics, field.name)
        for field in dataclasses.fields(diagnostics)
        if field.name not in _ROOT_LIST_FIELDS
    }


def _matrix() -> list[tuple[str, int, int, int, bool]]:
    """Return the audited sweep: priors x budgets x seeds x schedules."""
    return [
        (policy, m, n, seed, wave)
        for policy in _PRIORS
        for m, n in _BUDGETS
        for seed in _SEEDS
        for wave in (False, True)
    ]


# --------------------------------------------------------------------------
# The fixture's own frame, pinned before anything is claimed about it
# --------------------------------------------------------------------------


def test_the_silent_provider_hides_the_root_win_and_nothing_else() -> None:
    """SANITY: the T1 toy differs from the C1 toy in exactly one answer.

    If ``candidate_actions`` were silenced everywhere, T3's "the reply survey is
    untouched" would be vacuous; if ``terminal_winner`` were silenced too, the
    exhaustive scan could not certify anything and T1 would be testing the
    fixture rather than the search. So: the root answer is empty, the reply
    answer is the honest one, and the winner derivation is unchanged.
    """
    provider = SilentRootProvider()
    root = _CoverageState("root")
    assert provider.candidate_actions(root, root.legal_mask()) == ()
    assert ToyProvider().candidate_actions(root, root.legal_mask()) == (_ROOT_WIN,)

    won = root.clone()
    won.play(_ROOT_WIN)
    assert won.is_terminal()
    assert provider.terminal_winner(won) == 0, "the root mover won that line"

    reply = _CoverageState("interior")
    reply.play(0)
    assert provider.candidate_actions(reply, reply.legal_mask()) == (_INTERIOR_WIN,)

    # The two widths the transition bounds below are written against, read off
    # the fixtures rather than assumed: a toy that changed shape would otherwise
    # loosen T3 and T5 silently.
    assert int(root.legal_mask().sum()) == _ROOT_LEGAL
    assert int(_CoverageState("interior").legal_mask().sum()) == _INTERIOR_LEGAL


# --------------------------------------------------------------------------
# T1 -- a win no provider names is found, and only by the exhaustive scan
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("policy", "m", "n", "seed", "wave"), _matrix())
def test_a_win_the_provider_cannot_name_is_scanned_verified_and_played(
    policy: str, m: int, n: int, seed: int, wave: bool
) -> None:
    """T1: the KOM-232 checkmate case -- the root scan finds what no candidate
    list contained, at every seed, prior, budget and schedule.

    Every claim here is about a *provider that said nothing*: the survey asked it
    for no root candidates (``provider_calls == 0``), reached the win by walking
    the offered legal set in score order, verified it by clone/play, attached it
    as a real terminal node and stopped. The proof layer then plays it, which is
    why an active discovery mode requires one.
    """
    result, _ = _search_toy(
        "root",
        policy=policy,
        m=m,
        n=n,
        seed=seed,
        wave=wave,
        provider=SilentRootProvider(),
    )
    diagnostics = _diagnostics(result)

    assert result.action == _ROOT_WIN, "the scan did not play the immediate win"
    assert int(result.visit_counts[_ROOT_WIN]) >= 1, "the witness was not backed up"
    assert diagnostics.discovery_status == "root_witness"
    assert diagnostics.root_scan == "exhaustive"
    # Zero *in total*: a root witness ends the survey, so the child survey --
    # the one half that is still provider-based -- never runs either.
    assert diagnostics.provider_calls == 0, "the scan asked a provider anyway"
    assert diagnostics.terminal_witnesses == 1
    assert diagnostics.candidate_verification_transitions <= _ROOT_LEGAL, (
        "the scan verified more actions than the root offers"
    )
    assert diagnostics.survey_nn_rows == 0, "a verified candidate cost a network row"


@pytest.mark.parametrize(("m", "n"), _BUDGETS)
@pytest.mark.parametrize("seed", _SEEDS)
@pytest.mark.parametrize("wave", (False, True))
def test_the_provider_driven_mode_misses_the_same_win(
    m: int, n: int, seed: int, wave: bool
) -> None:
    """T1 (the other half): ``root_marshal_v1`` on the same toy does NOT find it.

    Under the ``misleading`` prior the draw never samples action 63 (the KOM-219
    C1 baseline), and the provider names nothing at the root, so the survey has
    nothing to verify and completes with the candidate set it was given. This is
    what makes the test above a difference between the two modes rather than a
    property of the fixture: same position, same provider, same seeds -- one mode
    plays the win and the other cannot see it at any budget.
    """
    result, _ = _search_toy(
        "root",
        policy="misleading",
        m=m,
        n=n,
        seed=seed,
        wave=wave,
        discovery="root_marshal_v1",
        provider=SilentRootProvider(),
    )
    diagnostics = _diagnostics(result)

    assert diagnostics.discovery_status == "candidate_survey_complete"
    assert diagnostics.root_scan == "provider"
    assert diagnostics.terminal_witnesses == 0
    assert result.action != _ROOT_WIN
    assert int(result.visit_counts[_ROOT_WIN]) == 0


# --------------------------------------------------------------------------
# T2 -- where the provider already sees the win, the modes agree
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("policy", "m", "n", "seed", "wave"), _matrix())
def test_on_the_capture_toy_both_modes_return_the_same_search(
    policy: str, m: int, n: int, seed: int, wave: bool
) -> None:
    """T2: on the C1 fixture the provider lists 63, so the scan adds reach the
    search did not need -- and changes nothing it produces.

    The value of this test is the negative: widening the root list is not allowed
    to move the action, the visit counts, the target, the row counts or the
    witness count on a position where the old list was already sufficient. What
    may differ is only :data:`_ROOT_LIST_FIELDS` -- how the list was built --
    and the transition count inside it is a *cost*, asserted as a bound in T5.
    """
    marshal, marshal_eval = _search_toy(
        "root",
        policy=policy,
        m=m,
        n=n,
        seed=seed,
        wave=wave,
        discovery="root_marshal_v1",
    )
    win, win_eval = _search_toy(
        "root", policy=policy, m=m, n=n, seed=seed, wave=wave, discovery="root_win_v1"
    )

    assert win.action == marshal.action == _ROOT_WIN
    assert np.array_equal(win.visit_counts, marshal.visit_counts)
    assert np.array_equal(win.policy_target, marshal.policy_target)
    assert np.array_equal(win.root_prior_logits, marshal.root_prior_logits)
    assert win.root_value == marshal.root_value
    assert win.sims_used == marshal.sims_used
    assert _mode_invariant(_diagnostics(win)) == _mode_invariant(_diagnostics(marshal))
    assert win_eval.calls == marshal_eval.calls, "the scan spent a network row"

    marshal_diagnostics = _diagnostics(marshal)
    win_diagnostics = _diagnostics(win)
    assert (marshal_diagnostics.root_scan, win_diagnostics.root_scan) == (
        "provider",
        "exhaustive",
    )
    assert marshal_diagnostics.provider_calls == 1
    assert win_diagnostics.provider_calls == 0
    assert win_diagnostics.identity.discovery_mode == "root_win_v1"
    assert marshal_diagnostics.identity.discovery_mode == "root_marshal_v1"


# --------------------------------------------------------------------------
# T3 -- the reply survey is untouched
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("policy", "m", "n", "seed", "wave"), _matrix())
def test_on_the_interior_toy_the_reply_survey_is_unchanged(
    policy: str, m: int, n: int, seed: int, wave: bool
) -> None:
    """T3: the C2 refutation fixture behaves identically under both modes.

    ``root_win_v1`` changes the ROOT half only. The interior toy's root holds no
    immediate win at all -- the scan verifies its two legal actions, attaches
    nothing, and hands sequential halving exactly the tree ``root_marshal_v1``
    would have -- while the anti-decisive half below each candidate is the same
    provider-driven survey, refuting the same move with the same witness. If a
    future change folded the reply scan into the exhaustive one, this is the
    assertion that would fail.
    """
    marshal, _ = _search_toy(
        "interior",
        policy=policy,
        m=m,
        n=n,
        seed=seed,
        wave=wave,
        discovery="root_marshal_v1",
    )
    win, _ = _search_toy(
        "interior",
        policy=policy,
        m=m,
        n=n,
        seed=seed,
        wave=wave,
        discovery="root_win_v1",
    )

    assert win.action == marshal.action
    assert np.array_equal(win.visit_counts, marshal.visit_counts)
    assert np.array_equal(win.policy_target, marshal.policy_target)
    assert _mode_invariant(_diagnostics(win)) == _mode_invariant(_diagnostics(marshal))

    win_diagnostics = _diagnostics(win)
    marshal_diagnostics = _diagnostics(marshal)
    # The reply survey is what the surviving counts are made of, and it ran
    # identically: the only extra cost is the root's own two probes.
    assert win_diagnostics.candidate_verification_transitions == (
        marshal_diagnostics.candidate_verification_transitions + _INTERIOR_LEGAL
    )
    assert win_diagnostics.provider_calls == marshal_diagnostics.provider_calls - 1


# --------------------------------------------------------------------------
# T4 -- one search, two schedules
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ("root", "interior"))
@pytest.mark.parametrize(("m", "n"), _BUDGETS)
@pytest.mark.parametrize("policy", _PRIORS)
def test_the_wave_schedule_returns_the_same_result_under_root_win(
    kind: str, m: int, n: int, policy: str
) -> None:
    """T4: the scan is a prologue both schedules share verbatim.

    It runs before halving and entirely through the scalar ``_make_node`` seam,
    so batching the leaves cannot move a number. Full ``SearchResult`` equality,
    diagnostics included -- the assertion that would fail if a future scheduler
    tried to fold the verification playouts into a wave.
    """
    for seed in _SEEDS:
        sequential, _ = _search_toy(
            kind, policy=policy, m=m, n=n, seed=seed, wave=False
        )
        waved, _ = _search_toy(kind, policy=policy, m=m, n=n, seed=seed, wave=True)
        assert sequential == waved, f"{kind} seed {seed} diverged between schedules"


# --------------------------------------------------------------------------
# T5 -- the scan is free in rows and bounded in transitions
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("policy", "m", "n", "seed", "wave"), _matrix())
@pytest.mark.parametrize(
    ("kind", "num_legal"), (("root", _ROOT_LEGAL), ("interior", _INTERIOR_LEGAL))
)
def test_the_root_scan_costs_no_network_row_and_a_bounded_number_of_transitions(
    kind: str, num_legal: int, policy: str, m: int, n: int, seed: int, wave: bool
) -> None:
    """T5: the published bounds, checked on both toys.

    A verified candidate is either non-terminal (nothing is attached) or terminal
    (the value is read off the game), and neither is handed to the evaluator --
    so widening the root list from a handful of captures to the whole legal set
    cannot cost a single network row, and ``1 + K + n`` still holds. What it does
    cost is rules work, bounded by ``num_legal + 2K``: the scan, one reply probe
    per candidate, and one transition per candidate played at the root.
    """
    result, evaluator = _search_toy(
        kind, policy=policy, m=m, n=n, seed=seed, wave=wave
    )
    diagnostics = _diagnostics(result)
    k = min(m, num_legal)

    rows = (
        diagnostics.root_nn_rows
        + diagnostics.survey_nn_rows
        + diagnostics.ordinary_nn_rows
    )
    assert rows == int(evaluator.calls), "a row was spent outside the accounting"
    assert diagnostics.root_nn_rows == 1
    assert rows <= 1 + k + n, f"{rows} rows exceeds the 1 + K + n bound"

    transitions = (
        diagnostics.candidate_verification_transitions
        + diagnostics.survey_root_transitions
    )
    assert transitions <= num_legal + 2 * k, (
        f"{transitions} transitions exceeds the num_legal + 2K bound"
    )
    assert diagnostics.tactical_sims_used <= 2 * k


# --------------------------------------------------------------------------
# T6 -- the untouched paths
# --------------------------------------------------------------------------


@pytest.mark.parametrize("wave", (False, True))
def test_discovery_off_and_proof_only_are_untouched(wave: bool) -> None:
    """T6: a new discovery mode adds a value, it does not move a default.

    ``proof_mode = "off"`` still reports no diagnostics at all, and a proof-only
    search reports ``root_scan = "provider"`` -- the default, which is what "no
    root was scanned exhaustively" means; there is no third word for "nothing
    happened", because :attr:`discovery_status` already is one.
    """
    evaluator = _CoverageEvaluator("uniform")
    plain = GumbelSearch(
        evaluator,
        _config(m=8, n=32, seed=0, proof="off", discovery="off"),
        wave_evaluator=evaluator.wave if wave else None,
    )
    assert plain.search(_CoverageState("interior")).diagnostics is None

    proof_evaluator = _CoverageEvaluator("uniform")
    proof_only = GumbelSearch(
        proof_evaluator,
        _config(m=8, n=32, seed=0, proof="interval_v1", discovery="off"),
        wave_evaluator=proof_evaluator.wave if wave else None,
    )
    diagnostics = _diagnostics(proof_only.search(_CoverageState("interior")))
    assert diagnostics.root_scan == "provider"
    assert diagnostics.discovery_status == "off"
    assert diagnostics.discovery_scope == "off"
    assert diagnostics.provider_id is None
    assert diagnostics.candidate_verification_transitions == 0


@pytest.mark.parametrize("wave", (False, True))
def test_a_forced_root_bypasses_the_scan_and_still_names_the_rule(wave: bool) -> None:
    """T6: a root with one legal action has nothing to scan, and says so.

    ``root_scan`` is the rule the root half runs under, which the mode fixes;
    ``discovery_status`` is what says whether it ran. A bypassed root is the one
    place those two come apart, so it is pinned rather than left to a reader.
    """
    evaluator = _QuietEvaluator("uniform")
    search = GumbelSearch(
        evaluator,
        _config(m=8, n=32, seed=0),
        wave_evaluator=evaluator.wave if wave else None,
        tactical_provider=EmptyProvider(),
    )
    diagnostics = _diagnostics(search.search(_QuietState("forced")))

    assert diagnostics.discovery_status == "bypassed_forced_root"
    assert diagnostics.root_scan == "exhaustive"
    assert diagnostics.candidate_verification_transitions == 0
    assert diagnostics.terminal_witnesses == 0


@pytest.mark.parametrize("wave", (False, True))
def test_a_zero_budget_bypasses_the_scan(wave: bool) -> None:
    """T6: ``n = 0`` leaves the scan unrun, exactly as it leaves the survey unrun.

    The scan is a prologue to sequential halving; with no halving to run there is
    nothing for it to prepare, and spending ``num_legal`` transitions on a search
    that will not simulate would be work nobody asked for.
    """
    evaluator = _CoverageEvaluator("uniform")
    search = GumbelSearch(
        evaluator,
        _config(m=8, n=0, seed=0),
        wave_evaluator=evaluator.wave if wave else None,
        tactical_provider=SilentRootProvider(),
    )
    diagnostics = _diagnostics(search.search(_CoverageState("root")))

    assert diagnostics.discovery_status == "bypassed_n0"
    assert diagnostics.root_scan == "exhaustive"
    assert diagnostics.candidate_verification_transitions == 0
    assert diagnostics.terminal_witnesses == 0


# --------------------------------------------------------------------------
# T7 -- the scan widens what is verified, never what is trusted
# --------------------------------------------------------------------------


@pytest.mark.parametrize("poison", ("winner_bool", "winner_out_of_range"))
def test_a_poisoned_winner_is_refused_before_any_witness_is_attached(
    poison: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T7: ``terminal_winner`` is still checked, and still checked first.

    The exhaustive scan hands the provider far more terminal states than a
    candidate list ever did, so the one answer it must keep refusing is a
    malformed winner: ``True`` compares equal to player ``1`` and would attach a
    backed-up ``+1`` on a move that does not win. The refusal is a ``ValueError``
    out of the search, and the backup log proves nothing was attached before it
    -- a half-surveyed tree returned as a result would be worse than a crash.
    """
    log = _BackupLog(monkeypatch)
    evaluator = _CoverageEvaluator("uniform")
    search = GumbelSearch(
        evaluator,
        _config(m=8, n=32, seed=0),
        tactical_provider=PoisonProvider(poison),
    )
    with pytest.raises(ValueError, match="terminal_winner must return 0, 1 or None"):
        search.search(_CoverageState("root"))
    assert log.entries == [], "a witness was backed up before the answer was checked"
