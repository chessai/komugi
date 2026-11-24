"""tools/release_model.py (KOM-1188 unit 1b; design section 3, 4, 6, 7).

Every test builds its EMA checkpoint from ``onnx_export.fresh_small_net(_SEED)``
(non-zero seed, review F3) in a fresh temp dir: never a real campaign
checkpoint, never the live campaign root, and nothing is uploaded. Oracles are
nets built independently of the tool (a second ``fresh_small_net(_SEED)``),
never the module the tool itself built.
"""

from __future__ import annotations

import glob
import hashlib
import io
import json
import os
import socket
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Final

import numpy as np
import onnx
import pytest
import torch
from onnx import numpy_helper
from onnx.reference import ReferenceEvaluator

import release_model
import schema_check

from komugi.campaign import factory
from komugi.engine import observation
from komugi.net import KomugiNet, onnx_export
from komugi.train import checkpoint, ema

#: Seed this module's own fixture net; non-zero (review F3's convention).
_SEED: Final[int] = 20261111
#: Element-wise tolerance for the ONNX-reference-vs-torch parity check.
_PARITY_ATOL: Final[float] = 1e-4

_REPO_ROOT = Path(__file__).resolve().parent.parent
_STEP = 1200
_ITERATION = 4
_CAMPAIGN = "c6"


def _make_ema(run_dir: Path, *, emptied: bool = False) -> Path:
    """Save a fake EMA checkpoint of the seeded fixture net into `run_dir`."""
    net = onnx_export.fresh_small_net(_SEED)
    state = {} if emptied else {k: v.clone() for k, v in net.state_dict().items()}
    meta = ema.ema_meta(
        iteration=_ITERATION, source_step=_STEP,
        train_sha256="0" * 64, ema_sha256="1" * 64,
    )
    meta.update(checkpoint.plane_meta(state or net.state_dict()))
    return checkpoint.save_checkpoint(run_dir, _STEP, state, {}, meta)


def _argv(src: Path, out: Path, **overrides: Any) -> list[str]:
    opts: dict[str, Any] = {
        "--ema-checkpoint": str(src), "--campaign": _CAMPAIGN,
        "--rules-profile": "komugi", "--modes-trained": ["advanced", "beginner"],
        "--out": str(out), "--eval-pointer": "evals/c6-gauntlet.json",
    }
    opts.update(overrides)
    argv: list[str] = []
    for key, value in opts.items():
        if value is None:
            continue
        argv.append(key)
        argv.extend(value if isinstance(value, list) else [value])
    return argv


def _fixture_net() -> torch.nn.Module:
    return onnx_export.fresh_small_net(_SEED)


def _heads(net: torch.nn.Module, obs: np.ndarray) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        out = net(torch.from_numpy(obs))
    return {"policy": out.policy_logits, "wdl": out.wdl_logits,
            "moves_left": out.moves_left}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _written(out: Path) -> list[str]:
    return sorted(p.name for p in out.iterdir()) if out.exists() else []


def _leaks(out: Path, needles: list[str]) -> list[tuple[str, str]]:
    found = []
    for path in sorted(out.iterdir()):
        data = path.read_bytes()
        for needle in needles:
            if needle and needle.encode() in data:
                found.append((path.name, needle))
    return found


def _leak_needles(*dirs: Path) -> list[str]:
    needles = {tempfile.gettempdir(), str(Path.home()), str(_REPO_ROOT)}
    for d in dirs:
        needles.add(str(d))
        needles.add(os.path.realpath(d))
    return sorted(needles)


@pytest.fixture(scope="module")
def obs() -> np.ndarray:
    return release_model.bench_observations(release_model.PARITY_N)


@pytest.fixture(scope="module")
def release(tmp_path_factory: pytest.TempPathFactory) -> SimpleNamespace:
    root = tmp_path_factory.mktemp("release")
    src = _make_ema(root / "run")
    out = root / "out"
    assert release_model.main(_argv(src, out)) == 0
    cards = [p for p in out.iterdir() if p.name.endswith(".card.json")]
    assert len(cards) == 1
    card = json.loads(cards[0].read_text())
    return SimpleNamespace(
        src=src, out=out, card=card, card_path=cards[0],
        index=json.loads((out / "models.json").read_text()),
        pt=out / card["assets"]["pt"]["path"],
        onnx=out / card["assets"]["onnx"]["path"],
    )


# --- AC1 ---------------------------------------------------------------------


def test_assets_written_and_sha256_match(release: SimpleNamespace) -> None:
    card = release.card
    assert _written(release.out) == sorted(
        [release.pt.name, release.onnx.name, release.card_path.name, "models.json"]
    )
    assert _sha(release.pt) == card["assets"]["pt"]["sha256"]
    assert _sha(release.onnx) == card["assets"]["onnx"]["sha256"]
    assert card["assets"]["pt"]["sha256"] != card["assets"]["onnx"]["sha256"]
    entry = release.index["models"][card["id"]]
    assert entry == {"card": release.card_path.name,
                     "onnxSha256": card["assets"]["onnx"]["sha256"]}
    assert release.index["recommended"] == {}
    assert set(release.index["recommended"].values()) <= set(release.index["models"])


def test_release_card_stamps_asset_bytes(release: SimpleNamespace) -> None:
    card = release.card
    assert card["assets"]["pt"]["bytes"] == release.pt.stat().st_size
    assert card["assets"]["onnx"]["bytes"] == release.onnx.stat().st_size
    assert card["assets"]["pt"]["bytes"] > 0
    assert card["assets"]["onnx"]["bytes"] > 0


def test_id_and_metadata_derivation(release: SimpleNamespace) -> None:
    card = release.card
    onnx_sha = _sha(release.onnx)
    assert card["id"] == f"komugi-{_CAMPAIGN}-s{_STEP}-{onnx_sha[:8]}"
    assert release.pt.name == f"{card['id']}.pt"
    assert release.onnx.name == f"{card['id']}.onnx"
    assert release.card_path.name == f"{card['id']}.card.json"
    model = onnx.load(str(release.onnx))
    props = {p.key: p.value for p in model.metadata_props}
    key = card["compatibilityKey"]
    assert key["planeMapVersion"] == int(props["komugi.plane_map_version"])
    assert key["planeMapVersion"] == observation.PLANE_MAP_VERSION
    assert key["actionEncodingVersion"] == 2
    assert key["rulesProfile"] == "komugi"
    small = factory.SMALL_CONFIG.trunk
    assert key["architectureTag"] == f"b{small.blocks}c{small.channels}"
    assert (card["campaign"], card["step"], card["iteration"]) == (
        _CAMPAIGN, _STEP, _ITERATION)
    assert card["modesTrained"] == ["advanced", "beginner"]
    assert card["sourceCheckpoint"] == {"name": release.src.name,
                                        "sha256": _sha(release.src)}
    assert card["exportOpset"] == onnx_export.OPSET_VERSION
    assert card["evalPointers"] == ["evals/c6-gauntlet.json"]
    assert card["license"] == "MIT, Copyright (c) 2026 chessai"
    schema_check.validate(card, schema_check.load_schema(release_model.CARD_SCHEMA))
    schema_check.validate(
        release.index, schema_check.load_schema(release_model.INDEX_SCHEMA))


# --- AC2 ---------------------------------------------------------------------


def test_onnx_matches_torch_reference(release: SimpleNamespace) -> None:
    net = _fixture_net()
    positions = onnx_export.parity_positions(8, seed=release_model.PARITY_SEED)
    batch = np.stack([position.observation for position in positions])
    evaluator = ReferenceEvaluator(str(release.onnx))
    produced = evaluator.run(None, {"obs": batch})
    expected = _heads(net, batch)
    for name, got in zip(onnx_export.OUTPUT_NAMES, produced):
        want = expected[name].numpy()
        assert got.shape == want.shape, name
        delta = float(np.abs(got - want).max())
        assert delta <= _PARITY_ATOL, f"{name}: max|delta|={delta}"


def test_selfcheck_recorded_in_card(release: SimpleNamespace) -> None:
    parity = release.card["exportParity"]
    assert parity["n"] == release_model.PARITY_N == 4
    assert parity["rtol"] == pytest.approx(release_model.parity_rtol(_fixture_net()))
    assert 0 < parity["atol"]
    assert 0 <= parity["maxAbsDelta"] <= parity["atol"] * 2


def test_parity_bound_below_1e4() -> None:
    net = _fixture_net()
    rtol = release_model.parity_rtol(net)
    assert 0 < rtol < 1e-4


def test_parity_selfcheck_detects_bias_shift(
    release: SimpleNamespace, tmp_path: Path, obs: np.ndarray
) -> None:
    net = _fixture_net()
    clean = release_model.check_export_parity(release.onnx, net, obs)
    assert clean.passed, clean.per_head

    model = onnx.load(str(release.onnx))
    targets = [i for i in model.graph.initializer
               if i.name.endswith("wdl.out_linear.bias")]
    assert len(targets) == 1, [i.name for i in model.graph.initializer]
    shifted = numpy_helper.to_array(targets[0]) + np.float32(1e-3)
    targets[0].CopyFrom(numpy_helper.from_array(
        shifted.astype(np.float32), targets[0].name))
    mutated = tmp_path / "mutated.onnx"
    onnx.save(model, str(mutated))

    red = release_model.check_export_parity(mutated, net, obs)
    assert not red.passed
    assert red.per_head["wdl"]["maxAbsDelta"] > red.per_head["wdl"]["atol"]


# --- AC3 ---------------------------------------------------------------------


def test_release_weights_bit_exact(release: SimpleNamespace, obs: np.ndarray) -> None:
    weights = release_model.load_release_weights(release.pt)
    assert weights.meta["role"] == "release"
    assert weights.meta["sourceCheckpoint"] == release.card["sourceCheckpoint"]
    assert weights.step == _STEP
    loaded = release_model.build_net(weights.model_state)
    got = _heads(loaded, obs)
    want = _heads(_fixture_net(), obs)
    for name in ("policy", "wdl", "moves_left"):
        assert torch.equal(got[name], want[name]), name


def _rewrite_pt(src: Path, dst: Path, edit: Any) -> Path:
    raw = torch.load(src, map_location="cpu", weights_only=True)
    edit(raw)
    buffer = io.BytesIO()
    torch.save(raw, buffer)
    dst.write_bytes(buffer.getvalue())
    return dst


def test_release_weights_refuses_ema_role(
    release: SimpleNamespace, tmp_path: Path
) -> None:
    def to_ema(raw: dict) -> None:
        meta = json.loads(raw["meta_json"])
        meta["role"] = "ema"
        raw["meta_json"] = json.dumps(meta)

    bad = _rewrite_pt(release.pt, tmp_path / "ema-role.pt", to_ema)
    with pytest.raises(ValueError, match="role"):
        release_model.load_release_weights(bad)


def test_release_weights_refuses_nan(release: SimpleNamespace, tmp_path: Path) -> None:
    def poison(raw: dict) -> None:
        key = next(k for k, v in raw["model"].items() if v.is_floating_point())
        raw["model"][key] = raw["model"][key].clone()
        raw["model"][key].view(-1)[0] = float("nan")

    bad = _rewrite_pt(release.pt, tmp_path / "nan.pt", poison)
    with pytest.raises(ValueError, match="non-finite"):
        release_model.load_release_weights(bad)


# --- AC6 ---------------------------------------------------------------------


class _Recorder:
    def __init__(self) -> None:
        self.dirs: set[str] = set()

    def add(self, target: Any) -> None:
        if target is None:
            target = "."
        if isinstance(target, int):
            target = os.readlink(f"/proc/self/fd/{target}")
        self.dirs.add(os.path.realpath(os.fsdecode(os.fspath(target))))


def _raise(*_a: Any, **_k: Any) -> Any:
    raise AssertionError("network/subprocess access during release_model.main")


def test_lists_only_out_and_staging_and_leaks_no_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = _make_ema(tmp_path / "campaign" / "run")
    out = tmp_path / "out"
    rec = _Recorder()
    staging: list[str] = []

    real_tempdir = release_model.tempfile.TemporaryDirectory

    class _Capture(real_tempdir):  # type: ignore[misc, valid-type]
        def __init__(self, *a: Any, **k: Any) -> None:
            super().__init__(*a, **k)
            staging.append(os.path.realpath(self.name))

    real_glob, real_iterdir = Path.glob, Path.iterdir
    real_scandir, real_listdir = os.scandir, os.listdir
    real_gglob, real_list = glob.glob, checkpoint.list_checkpoints

    def p_glob(self: Path, *a: Any, **k: Any) -> Any:
        rec.add(self)
        return real_glob(self, *a, **k)

    def p_iterdir(self: Path) -> Any:
        rec.add(self)
        return real_iterdir(self)

    def p_scandir(path: Any = None) -> Any:
        rec.add(path)
        return real_scandir(".") if path is None else real_scandir(path)

    def p_listdir(path: Any = None) -> Any:
        rec.add(path)
        return real_listdir(".") if path is None else real_listdir(path)

    def p_gglob(pathname: Any, *a: Any, **k: Any) -> Any:
        rec.add(os.path.dirname(os.fspath(pathname)) or ".")
        return real_gglob(pathname, *a, **k)

    def p_list(directory: Any) -> Any:
        rec.add(directory)
        return real_list(directory)

    monkeypatch.setattr(release_model.tempfile, "TemporaryDirectory", _Capture)
    monkeypatch.setattr(Path, "glob", p_glob)
    monkeypatch.setattr(Path, "iterdir", p_iterdir)
    monkeypatch.setattr(os, "scandir", p_scandir)
    monkeypatch.setattr(os, "listdir", p_listdir)
    monkeypatch.setattr(glob, "glob", p_gglob)
    monkeypatch.setattr(checkpoint, "list_checkpoints", p_list)
    monkeypatch.setattr(socket.socket, "connect", _raise)
    monkeypatch.setattr(subprocess, "Popen", _raise)
    monkeypatch.setattr(urllib.request, "urlopen", _raise)

    code = release_model.main(_argv(src, out))
    monkeypatch.undo()

    assert code == 0
    assert len(staging) == 1
    assert rec.dirs, "recorder saw nothing: the hooks are vacuous"
    assert staging[0] in rec.dirs
    allowed = {os.path.realpath(out), staging[0]}
    assert rec.dirs <= allowed, rec.dirs - allowed
    assert len(_written(out)) == 4
    needles = _leak_needles(src.parent, Path(staging[0]), out)
    assert _leaks(out, needles) == []


def test_absolute_asset_path_mutant_red(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = _make_ema(tmp_path / "run")
    out = tmp_path / "out"
    seen: list[dict] = []
    real_validate = release_model._validate

    def spy(instance: Any, schema_path: Path, what: str) -> None:
        if what == "card":
            seen.append(instance)
        real_validate(instance, schema_path, what)

    monkeypatch.setattr(release_model, "_asset_name", lambda path: str(out / path.name))
    monkeypatch.setattr(release_model, "_validate", spy)
    code = release_model.main(_argv(src, out))
    monkeypatch.undo()

    # The 4.2 name pattern refuses the mutant card; nothing is written.
    assert code != 0
    assert _written(out) == []
    assert len(seen) == 1
    card = seen[0]
    with pytest.raises(schema_check.ValidationError, match="pattern"):
        schema_check.validate(card, schema_check.load_schema(release_model.CARD_SCHEMA))
    # The output grep is red on the mutant's card bytes.
    out.mkdir()
    (out / "mutant.card.json").write_text(json.dumps(card))
    assert _leaks(out, _leak_needles(src.parent, out)) != []


def test_cli_refusals(tmp_path: Path) -> None:
    src = _make_ema(tmp_path / "run")
    cases = {
        "uppercase-campaign": {"--campaign": "C6"},
        "no-rules-profile": {"--rules-profile": None},
        "no-modes-trained": {"--modes-trained": None},
    }
    for label, override in cases.items():
        out = tmp_path / f"out-{label}"
        assert release_model.main(_argv(src, out, **override)) != 0, label
        assert not out.exists(), label


# --- F3 vacuity pair and N10 -------------------------------------------------


def test_fixture_differs_from_fresh_small_config(obs: np.ndarray) -> None:
    fixture = _heads(_fixture_net(), obs)
    fresh_net = KomugiNet(factory.SMALL_CONFIG)
    fresh_net.eval()
    fresh = _heads(fresh_net, obs)
    for name in ("policy", "wdl", "moves_left"):
        assert not torch.allclose(fixture[name], fresh[name]), name


def test_emptied_state_dict_refused(tmp_path: Path) -> None:
    src = _make_ema(tmp_path / "run", emptied=True)
    out = tmp_path / "out"
    assert release_model.main(_argv(src, out)) != 0
    assert not out.exists()


def test_campaign_net_env_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sentinel = "sentinel-not-a-net"
    monkeypatch.setenv("KOMUGI_CAMPAIGN_NET", sentinel)
    src = _make_ema(tmp_path / "run")
    assert release_model.main(_argv(src, tmp_path / "out-ok")) == 0
    assert os.environ.get("KOMUGI_CAMPAIGN_NET") == sentinel
    assert release_model.main(
        _argv(src, tmp_path / "out-refused", **{"--campaign": "C6"})) != 0
    assert os.environ.get("KOMUGI_CAMPAIGN_NET") == sentinel
    empty = _make_ema(tmp_path / "run-empty", emptied=True)
    assert release_model.main(_argv(empty, tmp_path / "out-empty")) != 0
    assert os.environ.get("KOMUGI_CAMPAIGN_NET") == sentinel
    assert not (tmp_path / "out-refused").exists()
    assert not (tmp_path / "out-empty").exists()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
