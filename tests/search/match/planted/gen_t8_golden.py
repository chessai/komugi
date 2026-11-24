"""Generate the T8 lambda-0 golden from the BIAS-FREE search (design §A4's
two-commit ordering: this ran at a commit where the bias table has no call
sites, so the golden records a bias-free search, not the feature under test).

Run by the ADVISOR (workers cannot execute):
  PYTHONPATH=src:tests/search/match:tests/search/oracle \
      python tests/search/match/planted/gen_t8_golden.py

Writes goldens/t8_lambda0.json: for each fixture, the chosen action and the
per-edge visit counts of a MatchSearch at the frozen config, with the
generating commit recorded. T8 (stage B) asserts a lambda=0 search still
matches this byte-for-byte and a lambda=0.45 search differs.
"""
from __future__ import annotations

import json
import pathlib
import subprocess

from planted import MatchToyState, solver_wdl_evaluator
from toygames import TreeGame

from komugi.search.match.config import MatchConfig
from komugi.search.match.search import MatchSearch

#: (name, payoff table, logits, playouts) — table 1 is the T1 sweep's first
#: found fixture; table 2 a plain asymmetric table; both depth-two TreeGames.
FIXTURES = [
    ("t1_first", ((1.0, 1.0), (-1.0, -1.0)), (0.0, 1.0), 48),
    ("asym", ((0.5, -0.2, 0.1), (-0.6, 0.9, 0.4), (-0.3, 0.7, -0.8)), (0.0, 0.0, 0.0), 48),
    # Advisor probe 2026-09-04: these two DIVERGE at lambda = 0.45 on the wired
    # branch (weights [45,3]->[44,4] and [35,13]->[29,19]), so T8's divergence
    # half has a fixture that can actually fire. Those two observed splits were
    # measured BEFORE the mover-frame fix in MatchTree._update_bias, which
    # reversed the sign the correction enters selection with, so the lambda=0.45
    # splits have moved since; what the probe establishes and the test asserts
    # is that this table is lambda-SENSITIVE at this budget, not which visit
    # split it lands on. The recorded lambda=0 numbers below are unaffected --
    # nothing multiplied by zero has a frame.
    ("div1", ((0.9, 0.8), (-0.9, -0.8)), (0.0, 1.0), 48),
    ("div2", ((0.9, 0.8), (-0.9, -0.8)), (0.0, 2.0), 48),
]


def main() -> None:
    out = {}
    for name, table, logits, playouts in FIXTURES:
        width = len(table)
        import numpy as np
        config = MatchConfig(num_actions=width, playouts=playouts)
        evaluator = solver_wdl_evaluator(
            num_actions=width, logits=np.asarray(logits, dtype=np.float32)
        )
        search = MatchSearch(config, evaluator)
        result = search.search(MatchToyState(TreeGame(table)), ply=0)
        root = search.tree.root
        assert root is not None
        weights = [float(w) for w in root.weight]
        out[name] = {
            "table": table, "logits": logits, "playouts": playouts,
            "action": int(result.action), "root_weights": weights,
            "evals_used": int(result.evals_used),
        }
        print(name, out[name]["action"], weights, flush=True)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    payload = {"generated_at_commit": sha, "note": "bias-free search (no SubtreeValueBias call sites at this commit)", "fixtures": out}
    p = pathlib.Path(__file__).parent / "goldens" / "t8_lambda0.json"
    p.write_text(json.dumps(payload, indent=2) + "\n")
    print("T8_GOLDEN_DONE", p)


if __name__ == "__main__":
    main()
