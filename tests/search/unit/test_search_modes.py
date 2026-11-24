"""KOM-221a: the frozen search-mode vocabulary and its identity derivation."""

from __future__ import annotations

import dataclasses
import os
import subprocess
import sys
from typing import Any

import pytest

from komugi.search.modes import (
    DISCOVERY_MODES,
    LEGACY_IDENTITY,
    PROOF_MODES,
    SearchAlgorithmIdentity,
    search_algorithm_identity,
    validate_search_modes,
)


# --- T1: validation -------------------------------------------------------


@pytest.mark.parametrize(
    ("proof_mode", "discovery_mode"),
    [
        ("off", "off"),
        ("interval_v1", "off"),
        ("interval_v1", "root_marshal_v1"),
        ("interval_v1", "root_win_v1"),
    ],
)
def test_allowed_combinations_validate(proof_mode: str, discovery_mode: str) -> None:
    validate_search_modes(proof_mode, discovery_mode)
    identity = search_algorithm_identity(proof_mode, discovery_mode)
    assert identity.proof_mode in PROOF_MODES


def test_mode_tuples_are_the_published_vocabulary() -> None:
    assert PROOF_MODES == ("off", "interval_v1")
    assert DISCOVERY_MODES == ("off", "root_marshal_v1", "root_win_v1")


@pytest.mark.parametrize("discovery_mode", ["root_marshal_v1", "root_win_v1"])
def test_discovery_requires_the_proof_layer(discovery_mode: str) -> None:
    with pytest.raises(ValueError, match="requires the proof layer"):
        validate_search_modes("off", discovery_mode)
    with pytest.raises(ValueError, match="requires the proof layer"):
        search_algorithm_identity("off", discovery_mode)


@pytest.mark.parametrize("bad", ["on", "", "INTERVAL_V1", " off"])
def test_unknown_proof_mode_strings_raise(bad: str) -> None:
    with pytest.raises(ValueError, match="unknown proof_mode"):
        validate_search_modes(bad, "off")


@pytest.mark.parametrize("bad", ["on", "", "INTERVAL_V1", " off"])
def test_unknown_discovery_mode_strings_raise(bad: str) -> None:
    with pytest.raises(ValueError, match="unknown discovery_mode"):
        validate_search_modes("interval_v1", bad)


@pytest.mark.parametrize("bad", [None, 1, True])
def test_non_str_proof_mode_raises(bad: Any) -> None:
    with pytest.raises(ValueError, match="proof_mode must be a str"):
        validate_search_modes(bad, "off")


@pytest.mark.parametrize("bad", [None, 1, True])
def test_non_str_discovery_mode_raises(bad: Any) -> None:
    with pytest.raises(ValueError, match="discovery_mode must be a str"):
        validate_search_modes("interval_v1", bad)


def test_error_messages_name_the_offending_value() -> None:
    with pytest.raises(ValueError, match="'INTERVAL_V1'"):
        validate_search_modes("INTERVAL_V1", "off")
    with pytest.raises(ValueError, match="'root_marshal'"):
        validate_search_modes("interval_v1", "root_marshal")


# --- T2: identity ---------------------------------------------------------


def test_legacy_identity_is_the_off_off_search() -> None:
    identity = search_algorithm_identity("off", "off")
    assert identity.decision_id == "gumbel_sh_survivor_v1"
    assert identity.target_id == "completed_q_full_legal_v1"
    assert LEGACY_IDENTITY == identity


@pytest.mark.parametrize("discovery_mode", ["off", "root_marshal_v1", "root_win_v1"])
def test_proof_identity_is_independent_of_discovery(discovery_mode: str) -> None:
    identity = search_algorithm_identity("interval_v1", discovery_mode)
    assert identity.decision_id == "gumbel_sh_interval_v1"
    assert identity.target_id == "completed_q_proof_support_v1"


@pytest.mark.parametrize(
    ("proof_mode", "discovery_mode"),
    [
        ("off", "off"),
        ("interval_v1", "off"),
        ("interval_v1", "root_marshal_v1"),
        ("interval_v1", "root_win_v1"),
    ],
)
def test_identity_carries_the_modes_it_was_derived_from(
    proof_mode: str, discovery_mode: str
) -> None:
    identity = search_algorithm_identity(proof_mode, discovery_mode)
    assert identity.proof_mode == proof_mode
    assert identity.discovery_mode == discovery_mode


def test_the_two_discovery_modes_are_distinguishable_identities() -> None:
    """KOM-234: the ids are proof-derived, so ``discovery_mode`` is what separates
    a ``root_win_v1`` row from a ``root_marshal_v1`` one -- and it must."""
    marshal = search_algorithm_identity("interval_v1", "root_marshal_v1")
    win = search_algorithm_identity("interval_v1", "root_win_v1")
    assert (marshal.decision_id, marshal.target_id) == (win.decision_id, win.target_id)
    assert marshal != win
    assert len({marshal, win}) == 2


def test_identity_is_frozen_hashable_and_compares_by_value() -> None:
    identity = search_algorithm_identity("interval_v1", "root_marshal_v1")
    with pytest.raises(dataclasses.FrozenInstanceError):
        identity.proof_mode = "off"  # type: ignore[misc]
    assert hash(identity) == hash(
        search_algorithm_identity("interval_v1", "root_marshal_v1")
    )
    assert identity == SearchAlgorithmIdentity(
        proof_mode="interval_v1",
        discovery_mode="root_marshal_v1",
        decision_id="gumbel_sh_interval_v1",
        target_id="completed_q_proof_support_v1",
    )
    assert identity != LEGACY_IDENTITY
    assert len({identity, search_algorithm_identity("interval_v1", "root_marshal_v1")}) == 1


# --- T3: purity -----------------------------------------------------------


# ``komugi.search.__init__`` eagerly imports :mod:`komugi.search.gumbel` (and so
# numpy), so ``import komugi.search.modes`` can never be free of them while that
# package import exists. The property this module actually promises is that
# *modes.py itself* pulls in nothing heavy, which is what a caller who imports it
# from a lighter package layout, or vendors it, gets. Load it by file path in a
# fresh interpreter and assert on ``sys.modules`` afterwards.

_PURITY_PROBE = """
import importlib.util
import sys

spec = importlib.util.spec_from_file_location("_komugi_modes_probe", sys.argv[1])
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

for heavy in ("komugi.search.gumbel", "komugi.search", "torch", "numpy"):
    assert heavy not in sys.modules, heavy

module.search_algorithm_identity("interval_v1", "root_marshal_v1")
"""


def test_modes_module_pulls_in_no_heavy_dependency() -> None:
    source = getattr(sys.modules["komugi.search.modes"], "__file__", None)
    assert source is not None
    env = dict(os.environ)
    src = os.path.dirname(os.path.dirname(os.path.dirname(source)))
    env["PYTHONPATH"] = (
        src + os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else src
    )
    proc = subprocess.run(
        [sys.executable, "-c", _PURITY_PROBE, source],
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr


def test_modes_module_imports_only_the_standard_library() -> None:
    source = getattr(sys.modules["komugi.search.modes"], "__file__", None)
    assert source is not None
    with open(source, encoding="utf-8") as handle:
        lines = [line.strip() for line in handle]
    imports = [line for line in lines if line.startswith(("import ", "from "))]
    assert imports == [
        "from __future__ import annotations",
        "from dataclasses import dataclass",
        "from typing import Final, Literal",
    ]
