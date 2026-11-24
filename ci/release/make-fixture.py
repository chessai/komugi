#!/usr/bin/env python3
"""make-fixture.py -- write a small, checkpoint-free candidate for
build-release.sh --fixture (KOM-1221 design docs/design/kom1221-release-ci.md
sections 3.2 and 9 unit 2).

Builds ``onnx_export.fresh_small_net`` (the same helper
``tools/test_release_model.py:50-59`` uses), saves it as an EMA checkpoint
under CANDIDATE_DIR, and writes a self-consistent ``candidate.json`` next to
it: ``campaign: "fixture"``, ``rulesProfile: "komugi"``,
``modesTrained: ["advanced"]`` (Advanced, so the e2e page can still select
through ``recommended.advanced``).

``expectedId``/``onnxSha256`` are derived by running the exact export
``tools/release_model.py`` itself performs (``onnx_export.export_net`` with
``batch="dynamic"``) against the same net, so the values recorded here are
what ``build-release.sh``'s own later run of ``release_model.py`` actually
produces -- otherwise section 3.2 step 4's three-way id/sha check would fail
on its own fixture. This relies on ``build-release.sh`` having already run
``ensure_ext`` and exported ``PYTHONPATH`` before invoking this file -- a
Python file cannot call a bash function (G4d).

Usage: make-fixture.py CANDIDATE_DIR
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Final

_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
for _extra in (_REPO_ROOT / "src", _REPO_ROOT / "tools"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import release_model  # noqa: E402  (tools/release_model.py)

from komugi.net import onnx_export  # noqa: E402
from komugi.train import checkpoint, ema  # noqa: E402

#: Non-zero seed (tools/test_release_model.py's own convention, review F3).
SEED: Final[int] = 20260923
STEP: Final[int] = 100
ITERATION: Final[int] = 1
CAMPAIGN: Final[str] = "fixture"
RULES_PROFILE: Final[str] = "komugi"
MODES_TRAINED: Final[list[str]] = ["advanced"]


def write_fixture(candidate_dir: Path) -> Path:
    """Write a checkpoint plus a self-consistent candidate.json into
    `candidate_dir`; return the candidate.json path."""
    candidate_dir.mkdir(parents=True, exist_ok=True)

    net = onnx_export.fresh_small_net(SEED)
    state = {key: value.clone() for key, value in net.state_dict().items()}
    meta = ema.ema_meta(
        iteration=ITERATION, source_step=STEP,
        train_sha256="0" * 64, ema_sha256="1" * 64,
    )
    meta.update(checkpoint.plane_meta(state))
    checkpoint_path = checkpoint.save_checkpoint(candidate_dir, STEP, state, {}, meta)
    source_sha256 = release_model.sha256_hex(checkpoint_path.read_bytes())

    with tempfile.TemporaryDirectory(prefix="make-fixture-") as staging:
        staged_onnx = Path(staging) / "fixture.onnx"
        onnx_export.export_net(net, staged_onnx, batch="dynamic")
        onnx_sha256 = release_model.sha256_hex(staged_onnx.read_bytes())

    expected_id = release_model.model_id(CAMPAIGN, STEP, onnx_sha256)
    candidate = {
        "checkpointAsset": checkpoint_path.name,
        "sourceCheckpointSha256": source_sha256,
        "campaign": CAMPAIGN,
        "rulesProfile": RULES_PROFILE,
        "modesTrained": MODES_TRAINED,
        "expectedId": expected_id,
        "onnxSha256": onnx_sha256,
    }
    candidate_path = candidate_dir / "candidate.json"
    candidate_path.write_text(
        json.dumps(candidate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return candidate_path


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: make-fixture.py CANDIDATE_DIR", file=sys.stderr)
        return 2
    candidate_path = write_fixture(Path(argv[0]))
    print(f"make-fixture: wrote {candidate_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
