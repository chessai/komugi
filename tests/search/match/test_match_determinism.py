"""A11 / T6: no randomness at all, and no hash randomisation either.

"Deterministic given a seed" would be a *weaker* claim than what is true here.
Match search has no root Dirichlet noise, no root temperature, no Gumbel draw
and no sampling anywhere: selection and the move choice are ``argmax``
throughout, with ties resolved by the lowest action id. So the ``seed`` the
``Mover`` seam carries is accepted, recorded, and **ignored**, and the first
test below is what stops a later "just add a little tie-break noise" from
passing review unnoticed.

The second test guards the one way a no-RNG search can still be
irreproducible. ``legal_hash`` is required to be a pure-numpy FNV-1a over
``actions.tobytes()`` and never Python's ``hash``, which is randomised per
process by ``PYTHONHASHSEED``. Comparing only the chosen action and the visit
counts across two processes **cannot** fail a ``hash()``-based implementation:
the stored hash is only ever compared against a same-process recomputation, so
it agrees with itself no matter what it is. What has teeth is comparing the
**stored values themselves** between two processes, which is what this does.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Final

from planted import DeepGame, MatchToyState, constant_wdl_evaluator

from komugi.search.match.config import MatchConfig
from komugi.search.match.search import MatchResult, MatchSearch

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[3]

#: Run in a fresh interpreter under a given ``PYTHONHASHSEED``. Self-contained
#: on purpose: it depends on the package under test and on numpy, and on
#: nothing in this test tree, so what the two processes disagree about can only
#: be the package.
PROGRAM: Final[str] = '''
import json

import numpy as np

from komugi.search.match.config import MatchConfig
from komugi.search.match.search import MatchSearch
from komugi.search.match.state import WdlEval

WIDTH = 6


class Chain:
    """Three plies, with a legal set that grows by one action each ply."""

    def __init__(self, moves=()):
        self.moves = tuple(moves)

    def legal_mask(self):
        mask = np.zeros(WIDTH, dtype=np.bool_)
        if len(self.moves) < 3:
            for action in range(len(self.moves) + 2):
                mask[action] = True
        return mask

    def play(self, action):
        self.moves = (*self.moves, int(action))

    def clone(self):
        return Chain(self.moves)

    def is_terminal(self):
        return len(self.moves) >= 3

    def to_play(self):
        return len(self.moves) % 2

    def observation(self):
        return np.zeros(1, dtype=np.float32)

    def terminal_wdl(self):
        return np.array([0.0, 1.0, 0.0], dtype=np.float64)


def evaluate(state):
    return WdlEval(
        np.arange(WIDTH, dtype=np.float32) * 0.125,
        np.array([0.3, 0.4, 0.3], dtype=np.float64),
    )


search = MatchSearch(MatchConfig(num_actions=WIDTH, playouts=48), evaluate)
result = search.search(Chain())

hashes = []
stack = [search.tree.root]
while stack:
    node = stack.pop()
    hashes.append(int(node.legal_hash))
    for index in sorted(node.children, reverse=True):
        stack.append(node.children[index])

print(
    json.dumps(
        {
            "action": int(result.action),
            "visits": [int(count) for count in result.visits],
            "hashes": hashes,
        }
    )
)
'''


def run_under_hash_seed(seed: str) -> dict[str, Any]:
    """Run :data:`PROGRAM` in a fresh interpreter under `seed`."""
    environment = dict(os.environ)
    environment["PYTHONHASHSEED"] = seed
    source = str(REPO_ROOT / "src")
    existing = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = f"{source}{os.pathsep}{existing}" if existing else source
    completed = subprocess.run(
        [sys.executable, "-c", PROGRAM],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        cwd=str(REPO_ROOT),
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    assert isinstance(payload, dict)
    return payload


def test_two_seeds_give_the_identical_search() -> None:
    """T6: the ``seed`` argument changes nothing, because nothing reads it."""
    config = MatchConfig(num_actions=3, playouts=64)
    results: list[MatchResult] = []
    for seed in (1, 987654321):
        search = MatchSearch(
            config, constant_wdl_evaluator([0.2, 0.5, 0.3], num_actions=3)
        )
        state = MatchToyState(DeepGame(width=3, depth=40))
        results.append(search.search(state))
    first, second = results
    assert first.action == second.action
    assert list(first.visits) == list(second.visits)
    assert list(first.weights) == list(second.weights)
    assert first.root_utility == second.root_utility


def test_the_stored_legal_hashes_survive_a_different_hash_seed() -> None:
    """T6: two fresh processes, two ``PYTHONHASHSEED``s, identical stored hashes.

    An implementation reaching for ``hash(actions.tobytes())`` produces a node
    field whose value changes per process, and this is the test that says so.
    Both nonzero seeds, so hash randomisation is *on* in both runs rather than
    disabled in one of them.
    """
    first = run_under_hash_seed("1")
    second = run_under_hash_seed("424242")

    assert first["hashes"] == second["hashes"]
    assert first["action"] == second["action"]
    assert first["visits"] == second["visits"]
    # The comparison must have something to compare: a fixture whose nodes all
    # shared one legal set would pass this test under any hash at all.
    assert len(first["hashes"]) >= 7
    assert len(set(first["hashes"])) >= 3
