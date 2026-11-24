# mypy: disable-error-code="no-untyped-def, no-untyped-call, misc, name-defined"
# The fixtures here are loaded from the sibling test modules by path (tests/ is not
# a package), so their classes are dynamically typed; the assertions are the point.
"""Proposed KOM-220a pins for the survey branches no shipped fixture exercises.

Loads the branch's discovery test module by path and borrows its fixtures.
Each test names the mutant it exists to kill (see the review's mutant table).
"""
from __future__ import annotations
import importlib.util, sys
from pathlib import Path
from typing import Any
import numpy as np
import numpy.typing as npt
import pytest

_TESTS = Path(__file__).parent

def _load(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(f"_review_{name}", _TESTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod; spec.loader.exec_module(mod)
    return mod

D = _load("test_gumbel_tactical_discovery")
from komugi.search import GumbelConfig, GumbelSearch  # noqa: E402

ROOT_WIN, INTERIOR_WIN, ROOT_LEGAL, WIDTH = D._ROOT_WIN, D._INTERIOR_WIN, D._ROOT_LEGAL, D._WIDTH


class FalsePositiveProvider(D.ToyProvider):
    """Conservative provider: names a drawing/quiet action BEFORE the real win."""
    provider_id = "toy-false-positive-v1"
    def candidate_actions(self, state, legal_mask):
        self.validate_state(state)
        wanted = (0, ROOT_WIN) if D._coverage_kind(state) == "root" else (0, INTERIOR_WIN)
        return tuple(a for a in wanted if bool(legal_mask[a]))


# ---- a 4-action toy whose child has a drawn reply, two winning replies and a quiet one ----
class _ReplyToy:
    NUM_ACTIONS = 4
    def __init__(self, history=()): self.history = history
    def clone(self): return _ReplyToy(self.history)
    def to_play(self): return len(self.history) % 2
    def is_terminal(self):
        h = self.history
        if len(h) == 1 and h[0] in (2, 3):   # root actions 2 and 3 win outright for the root mover
            return True
        return len(h) >= 2 and h[0] in (0, 1) and h[1] in (0, 1, 3)  # reply 0 draws; replies 1 and 3 win for the replier
    def terminal_value(self):
        h = self.history
        if len(h) == 1: return -1.0          # mover here is the opponent, who lost
        return 0.0 if h[1] == 0 else -1.0    # mover here is the root mover, who lost (or drew)
    def terminal_wdl(self):
        v = self.terminal_value()
        return np.array([0.0, 1.0, 0.0] if v == 0.0 else [0.0, 0.0, 1.0])
    def legal_mask(self):
        mask = np.zeros(4, dtype=np.bool_)
        if not self.is_terminal(): mask[:] = True
        return mask
    def play(self, a):
        assert bool(self.legal_mask()[a]), f"illegal {a} at {self.history}"
        self.history = (*self.history, a)


class _ReplyEvaluator:
    def __init__(self, root_logits=None):
        self.calls = 0; self.root_logits = root_logits
    def __call__(self, state):
        assert isinstance(state, _ReplyToy) and not state.is_terminal()
        self.calls += 1
        logits = np.zeros(4, dtype=np.float32)
        if self.root_logits is not None and not state.history:
            logits[:] = self.root_logits
        return logits, 0.0
    def wave(self, states): return [self(s) for s in states]


class _ReplyProvider:
    """Root: names both wins (2, 3). Child: names the draw, a win, and the second win (0, 1, 3)."""
    provider_id = "toy-reply-v1"
    def __init__(self, at_root=(), at_child=(0, 1, 3)):
        self.at_root, self.at_child = at_root, at_child
    def validate_state(self, state):
        if not isinstance(state, _ReplyToy): raise TypeError("unsupported")
    def candidate_actions(self, state, legal_mask):
        self.validate_state(state)
        wanted = self.at_root if not state.history else self.at_child
        return tuple(a for a in wanted if bool(legal_mask[a]))
    def terminal_winner(self, s):
        self.validate_state(s)
        if not s.is_terminal(): raise ValueError("non-terminal")
        v = float(s.terminal_value()); mover = int(s.to_play())
        return None if v == 0.0 else (mover if v == 1.0 else 1 - mover)


def _cfg(m, n, seed, A=4, discovery="root_marshal_v1"):
    return GumbelConfig(num_actions=A, m=m, n=n, seed=seed, proof_mode="interval_v1", discovery_mode=discovery)


DISCOVERY_MODES = ("root_marshal_v1", "root_win_v1")
"""Both root rules. The root half differs (provider list vs. every legal action);
every rule the survey applies to that list -- ordering, verification, what is
attached -- is shared code and must hold under either (KOM-235 F3)."""


# M9b: a terminal root candidate that is NOT a win for the root mover is skipped, not attached
@pytest.mark.parametrize("wave", (False, True))
def test_a_drawing_root_candidate_is_verified_and_skipped(wave, monkeypatch):
    log = D._BackupLog(monkeypatch)
    ev = D._CoverageEvaluator("misleading")      # logits[63] = -30, so action 0 is verified FIRST
    s = D._SurveySpy(ev, D._config(m=8, n=32, seed=0), wave_evaluator=ev.wave if wave else None,
                     tactical_provider=FalsePositiveProvider())
    s.log = log
    r = s.search(D._CoverageState("root"))
    d = D._diagnostics(r)
    assert d.discovery_status == "root_witness"
    assert d.candidate_verification_transitions == 2, "both candidates were played"
    assert d.terminal_witnesses == 1
    assert np.flatnonzero(s.post_survey_visits).tolist() == [ROOT_WIN], "the draw was attached as a witness"
    assert s.survey_backups == [(ROOT_WIN, 1.0)]
    assert r.action == ROOT_WIN


# M10: a NON-terminal root candidate is skipped without an evaluator call
@pytest.mark.parametrize("wave", (False, True))
def test_a_non_terminal_root_candidate_costs_one_transition_and_no_row(wave):
    ev = D._CoverageEvaluator("uniform")
    s = D._SurveySpy(ev, D._config(m=8, n=32, seed=0), wave_evaluator=ev.wave if wave else None,
                     tactical_provider=FalsePositiveProvider())   # at the interior root it names action 0, which is quiet
    r = s.search(D._CoverageState("interior"))
    d = D._diagnostics(r)
    assert d.discovery_status == "candidate_survey_complete"
    assert d.candidate_verification_transitions == 1 + 2, "root probe of 0, then replies 0 and 76 under candidate 0"
    assert d.terminal_witnesses == 1
    assert d.survey_nn_rows == 1 and ev.calls == 1 + d.survey_nn_rows + d.ordinary_nn_rows
    assert int(s.post_survey_visits.sum()) == d.tactical_sims_used == 3


# M11: several winning root candidates -> the one the search itself rates highest is the witness
# (KOM-235 F3: under BOTH root rules. The exhaustive list is built differently and ordered by the
# same score, so an id-order mutant of either branch attaches 2 where 3 is expected.)
@pytest.mark.parametrize("discovery", DISCOVERY_MODES)
@pytest.mark.parametrize("seed", (0, 1, 2, 3))
@pytest.mark.parametrize("wave", (False, True))
def test_the_root_witness_is_the_highest_scoring_winning_candidate(seed, wave, discovery):
    logits = np.array([0.0, 0.0, 0.0, 10.0], dtype=np.float32)
    ev = _ReplyEvaluator(logits)
    cfg = _cfg(m=2, n=8, seed=seed, discovery=discovery)
    g = np.random.default_rng(seed).gumbel(size=4)
    expected = max((2, 3), key=lambda a: g[a] + float(logits[a]))
    s = D._SurveySpy(ev, cfg, wave_evaluator=ev.wave if wave else None, tactical_provider=_ReplyProvider(at_root=(2, 3)))
    r = s.search(_ReplyToy())
    d = D._diagnostics(r)
    assert d.discovery_status == "root_witness" and d.terminal_witnesses == 1
    if discovery == "root_marshal_v1":
        assert d.candidate_verification_transitions == 1, "the best-scoring win is verified first and ends the survey"
        assert d.provider_calls == 1
    else:
        # The scan also probes the quiet actions 0 and 1 where they outscore the wins, but never
        # more than the four legal ids, and it asks the provider nothing at the root.
        assert 1 <= d.candidate_verification_transitions <= 4
        assert d.provider_calls == 0
    assert np.flatnonzero(s.post_survey_visits).tolist() == [expected]


# M12 + M13: a drawn reply is skipped; the first winning reply is attached; the second is never verified
@pytest.mark.parametrize("wave", (False, True))
def test_a_drawn_reply_is_skipped_and_only_one_witness_is_attached_per_candidate(wave):
    ev = _ReplyEvaluator()
    s = D._SurveySpy(ev, _cfg(m=2, n=8, seed=0), wave_evaluator=ev.wave if wave else None,
                     tactical_provider=_ReplyProvider(at_child=(0, 1, 3)))
    # keep the draw sample on the two quiet root actions so every candidate has a reply survey
    ev.root_logits = np.array([0.0, 0.0, -30.0, -30.0], dtype=np.float32)
    r = s.search(_ReplyToy())
    d = D._diagnostics(r)
    K = s.candidate_count
    assert d.discovery_status == "candidate_survey_complete" and K == 2
    for a in (0, 1):
        child = s.node.children[a]
        assert 0 not in child.children, "a drawn reply was attached as a witness"
        assert 1 in child.children and child.children[1].is_terminal
        assert 3 not in child.children, "a second witness was attached after the first"
    assert d.terminal_witnesses == K
    assert d.candidate_verification_transitions == 2 * K, "replies 0 and 1 per candidate, never 3"
    assert d.tactical_sims_used == 2 * K
    assert r.sims_used == 8 + 2 * K == int(r.visit_counts.sum())


# M16: a root witness the draw missed is appended to the candidate set (K -> K + 1)
@pytest.mark.parametrize("seed", D._SEEDS)
@pytest.mark.parametrize("wave", (False, True))
def test_a_root_witness_the_draw_missed_widens_the_candidate_set_by_one(seed, wave):
    ev = D._CoverageEvaluator("misleading")
    s = D._SurveySpy(ev, D._config(m=8, n=32, seed=seed), wave_evaluator=ev.wave if wave else None,
                     tactical_provider=D.ToyProvider())
    r = s.search(D._CoverageState("root"))
    assert s.candidate_count == 8 + 1, "the witness was not appended to the halving set"
    assert D._diagnostics(r).rescue_scope != "full_root"
