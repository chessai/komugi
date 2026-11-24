#!/usr/bin/env python3
"""release_model.py -- turn an EMA checkpoint into a model release (KOM-1188).

Design: docs/design/kom1188-model-releases.md sections 3 and 4. Given one EMA
checkpoint FILE, writes under ``--out``:

* ``<id>.pt``        -- weights-only payload, ``meta.role == "release"``;
* ``<id>.onnx``      -- dynamic-batch export via ``onnx_export.export_net``;
* ``<id>.card.json`` -- the model card (schemas/model-card.schema.json);
* ``models.json``    -- read if present, one ``models`` entry appended.

where ``<id> = komugi-<campaign>-s<step>-<sha256(onnx)[:8]>``.

The source bytes are staged into a fresh temporary directory first and
every later step reads only the staged copy, so the caller's checkpoint
directory is never listed (review F4). The net's architecture is matched from
the staged state dict's own shapes against ``{CONFIG, SMALL_CONFIG}``; the
``KOMUGI_CAMPAIGN_NET`` environment variable is neither read nor written
(review N10). The exported ``.onnx`` must pass an N=4 export-parity
self-check against the loaded torch net (derived float32 bound, review F10/N7)
before the card or the ``models.json`` entry is written. Nothing is uploaded.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import io
import json
import math
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
_SRC: Final[Path] = _REPO_ROOT / "src"
_TOOLS: Final[Path] = _REPO_ROOT / "tools"
for _extra in (_SRC, _TOOLS):
    if _extra.is_dir() and str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from onnx.reference import ReferenceEvaluator  # noqa: E402

from komugi.campaign import factory  # noqa: E402
from komugi.engine import observation  # noqa: E402
from komugi.engine.action_encoding import ACTION_ENCODING_VERSION  # noqa: E402
from komugi.net import KomugiNet, NetConfig  # noqa: E402
from komugi.net import onnx_export  # noqa: E402
from komugi.train import checkpoint, ema  # noqa: E402

import schema_check  # noqa: E402  (tools/schema_check.py)

#: ``meta["role"]`` of a release ``.pt`` (distinct from ``ema.EMA_ROLE``).
RELEASE_ROLE: Final[str] = "release"
#: The card's fixed license literal (tickets/KOM-1188.md Findings).
LICENSE: Final[str] = "MIT, Copyright (c) 2026 chessai"
#: ``--rules-profile`` choices: exactly rules_config.ts's ``RulesProfile``.
RULES_PROFILES: Final[tuple[str, ...]] = ("komugi", "gungijs", "ligungi")
#: ``--modes-trained`` choices: the card schema's ``modesTrained`` enum.
MODES: Final[tuple[str, ...]] = ("intro", "beginner", "intermediate", "advanced")
#: ``--campaign`` grammar (section 2.1).
CAMPAIGN_RE: Final[re.Pattern[str]] = re.compile(r"[a-z0-9]+")
#: Batch size of the export-parity self-check.
PARITY_N: Final[int] = 4
#: Seed :func:`bench_observations` plays its random playout from (non-zero,
#: review F3's convention).
PARITY_SEED: Final[int] = 20261124
#: float32 unit roundoff used by the derived parity bound.
EPS32: Final[float] = 2.0**-23
#: Name of the index file under ``--out``.
MODELS_INDEX: Final[str] = "models.json"
CARD_SCHEMA: Final[Path] = _REPO_ROOT / "schemas" / "model-card.schema.json"
INDEX_SCHEMA: Final[Path] = _REPO_ROOT / "schemas" / "models-index.schema.json"
#: The configs a staged state dict may match, by shape (section 3.2 step 2).
CANDIDATE_CONFIGS: Final[tuple[NetConfig, ...]] = (factory.CONFIG, factory.SMALL_CONFIG)


class ReleaseError(Exception):
    """A refusal: the tool exits non-zero and records no release."""


@dataclass(frozen=True)
class ReleaseWeights:
    """What :func:`load_release_weights` returns."""

    step: int
    model_state: dict[str, torch.Tensor]
    meta: dict[str, Any]


@dataclass(frozen=True)
class ParityResult:
    """Outcome of :func:`check_export_parity`."""

    passed: bool
    n: int
    max_abs_delta: float
    atol: float
    rtol: float
    per_head: dict[str, dict[str, float]]

    def card_record(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "maxAbsDelta": self.max_abs_delta,
            "atol": self.atol,
            "rtol": self.rtol,
        }


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def model_id(campaign: str, step: int, onnx_sha256: str) -> str:
    """Section 2.1: ``komugi-<campaign>-s<step>-<sha8>``."""
    return f"komugi-{campaign}-s{step}-{onnx_sha256[:8]}"


def architecture_tag(config: NetConfig) -> str:
    """Section 2.2.4: ``b<blocks>c<channels>``."""
    return f"b{config.trunk.blocks}c{config.trunk.channels}"


def _asset_name(path: Path) -> str:
    """The card's asset ``path`` value: a bare basename, never a path (N8)."""
    return path.name


def _shape_map(state: Mapping[str, torch.Tensor]) -> dict[str, tuple[int, ...]]:
    return {key: tuple(value.shape) for key, value in state.items()}


@functools.lru_cache(maxsize=None)
def _candidate_shapes(index: int) -> dict[str, tuple[int, ...]]:
    return _shape_map(KomugiNet(CANDIDATE_CONFIGS[index]).state_dict())


def match_config(model_state: Mapping[str, torch.Tensor]) -> NetConfig:
    """Return the one config in :data:`CANDIDATE_CONFIGS` whose net has exactly
    `model_state`'s key set and shapes, or raise :class:`ReleaseError`."""
    shapes = _shape_map(model_state)
    matches = [
        cfg for index, cfg in enumerate(CANDIDATE_CONFIGS)
        if _candidate_shapes(index) == shapes
    ]
    if len(matches) != 1:
        raise ReleaseError(
            f"state dict shape mismatch: {len(model_state)} tensors match "
            f"{len(matches)} of the known net configs (exactly one required)"
        )
    return matches[0]


def build_net(model_state: Mapping[str, torch.Tensor]) -> KomugiNet:
    """Build ``KomugiNet(cfg)`` for the config matching `model_state`'s shapes
    and load it strictly; eval mode. Never consults ``KOMUGI_CAMPAIGN_NET``."""
    cfg = match_config(model_state)
    net = KomugiNet(cfg)
    net.load_state_dict(dict(model_state))
    net.eval()
    return net


def stage_checkpoint(source: Path, staging: Path) -> Path:
    """Copy `source`'s bytes (read once) into `staging` under its basename."""
    data = source.read_bytes()
    staged = staging / source.name
    staged.write_bytes(data)
    return staged


def load_staged_ema(staged: Path) -> checkpoint.Checkpoint:
    loaded = checkpoint.load_checkpoint_file(staged)
    if loaded is None:
        raise ReleaseError(f"{staged.name} is not a checkpoint that verifies")
    role = loaded.meta.get("role")
    if role != ema.EMA_ROLE:
        raise ReleaseError(
            f"{staged.name} has role {role!r}, expected {ema.EMA_ROLE!r}"
        )
    return loaded


def release_payload_bytes(
    *, step: int, model_state: Mapping[str, torch.Tensor], in_planes: int,
    source_name: str, source_sha256: str,
) -> bytes:
    """The weights-only ``.pt`` bytes: checkpoint.py's payload shape with an
    empty optimiser and ``role: "release"`` (section 3.2 step 3)."""
    meta = {
        checkpoint.PLANE_MAP_VERSION_KEY: observation.PLANE_MAP_VERSION,
        checkpoint.IN_PLANES_KEY: in_planes,
        "action_encoding_version": ACTION_ENCODING_VERSION,
        "role": RELEASE_ROLE,
        "sourceCheckpoint": {"name": source_name, "sha256": source_sha256},
    }
    payload = {
        "format_version": checkpoint.FORMAT_VERSION,
        "step": step,
        "meta_json": json.dumps(meta, sort_keys=True, separators=(",", ":")),
        "model": dict(model_state),
        "optim": {},
    }
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return buffer.getvalue()


def load_release_weights(path: str | Path) -> ReleaseWeights:
    """Read a release ``.pt`` back (review F11).

    ``weights_only=True``; requires ``format_version`` and
    ``meta["role"] == "release"``; refuses any non-finite tensor; checks the
    plane meta (``require_matching_plane_count`` against the weights' own
    matched config) and the action encoding version.
    """
    target = Path(path)
    raw: object = torch.load(target, map_location="cpu", weights_only=True)
    if not isinstance(raw, dict):
        raise ValueError(f"release payload is a {type(raw).__name__}, expected a dict")
    if raw.get("format_version") != checkpoint.FORMAT_VERSION:
        raise ValueError(
            f"release format version {raw.get('format_version')!r}, "
            f"expected {checkpoint.FORMAT_VERSION}"
        )
    step = raw.get("step")
    if not isinstance(step, int) or isinstance(step, bool):
        raise ValueError(f"release step is {step!r}, expected an int")
    meta_json = raw.get("meta_json")
    if not isinstance(meta_json, str):
        raise ValueError("release meta_json is not a string")
    meta = json.loads(meta_json)
    if not isinstance(meta, dict):
        raise ValueError("release meta is not a JSON object")
    if meta.get("role") != RELEASE_ROLE:
        raise ValueError(
            f"release meta role is {meta.get('role')!r}, expected {RELEASE_ROLE!r}"
        )
    model = raw.get("model")
    if not isinstance(model, dict) or not all(
        isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in model.items()
    ):
        raise ValueError("release model state is not a str -> tensor dict")
    checkpoint._refuse_nonfinite_state(model, {})
    if meta.get("action_encoding_version") != ACTION_ENCODING_VERSION:
        raise ValueError(
            f"release action_encoding_version {meta.get('action_encoding_version')!r}, "
            f"expected {ACTION_ENCODING_VERSION}"
        )
    cfg = match_config(model)
    checkpoint.require_matching_plane_count(
        checkpoint.Checkpoint(
            step=step, model_state=model, optim_state={}, meta=meta, path=target
        ),
        cfg.trunk.in_planes,
    )
    return ReleaseWeights(step=step, model_state=dict(model), meta=meta)


def bench_observations(n: int = PARITY_N) -> np.ndarray:
    """``(n, IN_PLANES, 9, 9)`` float32 observations from a seeded random
    playout (:func:`komugi.net.onnx_export.parity_positions`), split evenly
    between the draft and the play phase."""
    positions = onnx_export.parity_positions(n, seed=PARITY_SEED)
    return np.stack([position.observation for position in positions])


def parity_rtol(net: torch.nn.Module) -> float:
    """``rtol = 4*sqrt(K*L)*eps32`` (section 3.2 step 4), computed from `net`:
    K is the widest Conv2d/Linear fan-in, L the count of such modules in
    ``net.trunk`` plus the largest count in any one head."""

    def layers(module: torch.nn.Module) -> list[torch.nn.Module]:
        return [
            m for m in module.modules()
            if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear))
        ]

    every = layers(net)
    fan_in = max(int(m.weight[0].numel()) for m in every)
    heads = [child for name, child in net.named_children() if name != "trunk"]
    depth = len(layers(net.trunk)) + max(len(layers(h)) for h in heads)
    return 4.0 * math.sqrt(fan_in * depth) * EPS32


def _torch_heads(net: torch.nn.Module, obs: np.ndarray) -> dict[str, np.ndarray]:
    with torch.no_grad():
        out = net(torch.from_numpy(obs))
    return {
        "policy": out.policy_logits.numpy(),
        "wdl": out.wdl_logits.numpy(),
        "moves_left": out.moves_left.numpy(),
    }


def check_export_parity(
    onnx_path: str | Path, net: torch.nn.Module, obs: np.ndarray | None = None,
) -> ParityResult:
    """Run `onnx_path` against `net` per exported head; pass iff
    ``|onnx - torch| <= atol + rtol*|torch|`` elementwise for every head, with
    ``atol = rtol*max|torch|`` per head."""
    batch = bench_observations() if obs is None else obs
    rtol = parity_rtol(net)
    expected = _torch_heads(net, batch)
    produced = ReferenceEvaluator(str(onnx_path)).run(
        None, {onnx_export.INPUT_NAME: batch}
    )
    passed = True
    max_delta = 0.0
    max_atol = 0.0
    per_head: dict[str, dict[str, float]] = {}
    for name, got in zip(onnx_export.OUTPUT_NAMES, produced):
        want = expected[name]
        got = np.asarray(got)
        if got.shape != want.shape:
            passed = False
            per_head[name] = {"maxAbsDelta": math.inf, "atol": 0.0}
            max_delta = math.inf
            continue
        atol = rtol * float(np.abs(want).max())
        delta = np.abs(got.astype(np.float64) - want.astype(np.float64))
        if not bool(np.all(delta <= atol + rtol * np.abs(want))):
            passed = False
        head_max = float(delta.max())
        per_head[name] = {"maxAbsDelta": head_max, "atol": atol}
        max_delta = max(max_delta, head_max)
        max_atol = max(max_atol, atol)
    return ParityResult(
        passed=passed, n=int(batch.shape[0]), max_abs_delta=max_delta,
        atol=max_atol, rtol=rtol, per_head=per_head,
    )


def build_card(
    *, release_id: str, campaign: str, iteration: int, step: int,
    config: NetConfig, rules_profile: str, modes_trained: Sequence[str],
    pt_path: Path, pt_sha256: str, pt_bytes: int,
    onnx_path: Path, onnx_sha256: str, onnx_bytes: int,
    source_name: str, source_sha256: str, parity: ParityResult,
    eval_pointers: Sequence[str], created_at: str,
) -> dict[str, Any]:
    """The model card of section 4.1."""
    return {
        "id": release_id,
        "campaign": campaign,
        "iteration": iteration,
        "step": step,
        "compatibilityKey": {
            "planeMapVersion": observation.PLANE_MAP_VERSION,
            "actionEncodingVersion": ACTION_ENCODING_VERSION,
            "rulesProfile": rules_profile,
            "architectureTag": architecture_tag(config),
        },
        "modesTrained": list(modes_trained),
        "assets": {
            "pt": {"path": _asset_name(pt_path), "sha256": pt_sha256, "bytes": pt_bytes},
            "onnx": {"path": _asset_name(onnx_path), "sha256": onnx_sha256, "bytes": onnx_bytes},
        },
        "sourceCheckpoint": {"name": source_name, "sha256": source_sha256},
        "exportOpset": onnx_export.OPSET_VERSION,
        "exportParity": parity.card_record(),
        "evalPointers": list(eval_pointers),
        "license": LICENSE,
        "createdAt": created_at,
    }


def _validate(instance: Any, schema_path: Path, what: str) -> None:
    try:
        schema_check.validate(instance, schema_check.load_schema(schema_path))
    except schema_check.ValidationError as exc:
        raise ReleaseError(f"{what} fails {schema_path.name}: {exc}") from exc


def _dump(obj: Any) -> str:
    return json.dumps(obj, indent=2, sort_keys=True) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--ema-checkpoint", type=Path, required=True,
                        help="EMA checkpoint FILE (never a directory)")
    parser.add_argument("--campaign", required=True, help="e.g. c6; [a-z0-9]+")
    parser.add_argument("--rules-profile", required=True, choices=RULES_PROFILES)
    parser.add_argument("--modes-trained", required=True, nargs="+", choices=MODES)
    parser.add_argument("--out", type=Path, required=True,
                        help="output directory; created if absent")
    parser.add_argument("--eval-pointer", action="append", default=[],
                        dest="eval_pointers", help="free-form; repeatable")
    return parser


def _release(args: argparse.Namespace) -> Path:
    if CAMPAIGN_RE.fullmatch(args.campaign) is None:
        raise ReleaseError(f"--campaign {args.campaign!r} does not match [a-z0-9]+")
    source: Path = args.ema_checkpoint
    if not source.is_file():
        raise ReleaseError(f"--ema-checkpoint {source.name!r} is not a file")
    out: Path = args.out
    if out.exists() and not out.is_dir():
        raise ReleaseError("--out exists and is not a directory")

    with tempfile.TemporaryDirectory(prefix="release-stage-") as staging_str:
        staging = Path(staging_str)
        staged = stage_checkpoint(source, staging)
        source_sha256 = sha256_hex(staged.read_bytes())
        loaded = load_staged_ema(staged)
        iteration = loaded.meta.get("iteration")
        if not isinstance(iteration, int) or isinstance(iteration, bool) or iteration < 0:
            raise ReleaseError(f"EMA meta iteration is {iteration!r}")

        config = match_config(loaded.model_state)
        net = KomugiNet(config)
        checkpoint.require_matching_plane_count(loaded, config.trunk.in_planes)
        net.load_state_dict(loaded.model_state)
        net.eval()

        staged_onnx = staging / "export.onnx"
        onnx_export.export_net(net, staged_onnx, batch="dynamic")
        parity = check_export_parity(staged_onnx, net)
        if not parity.passed:
            raise ReleaseError(f"export parity self-check failed: {parity.per_head}")

        onnx_bytes = staged_onnx.read_bytes()
        onnx_sha256 = sha256_hex(onnx_bytes)
        release_id = model_id(args.campaign, loaded.step, onnx_sha256)
        pt_bytes = release_payload_bytes(
            step=loaded.step, model_state=loaded.model_state,
            in_planes=config.trunk.in_planes, source_name=staged.name,
            source_sha256=source_sha256,
        )
        pt_path = out / f"{release_id}.pt"
        onnx_path = out / f"{release_id}.onnx"
        card_path = out / f"{release_id}.card.json"
        card = build_card(
            release_id=release_id, campaign=args.campaign, iteration=iteration,
            step=loaded.step, config=config, rules_profile=args.rules_profile,
            modes_trained=args.modes_trained, pt_path=pt_path,
            pt_sha256=sha256_hex(pt_bytes), pt_bytes=len(pt_bytes),
            onnx_path=onnx_path, onnx_sha256=onnx_sha256,
            onnx_bytes=len(onnx_bytes), source_name=staged.name,
            source_sha256=source_sha256, parity=parity,
            eval_pointers=args.eval_pointers,
            created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )
        _validate(card, CARD_SCHEMA, "card")

        index_path = out / MODELS_INDEX
        if index_path.is_file():
            index = json.loads(index_path.read_text(encoding="utf-8"))
        else:
            index = {"schemaVersion": 1, "recommended": {}, "models": {}}
        if not isinstance(index, dict) or release_id in index.get("models", {}):
            raise ReleaseError(f"{MODELS_INDEX} already records {release_id}")
        index["models"][release_id] = {"card": card_path.name, "onnxSha256": onnx_sha256}
        _validate(index, INDEX_SCHEMA, MODELS_INDEX)

        existing = [p.name for p in (pt_path, onnx_path, card_path) if p.exists()]
        if existing:
            raise ReleaseError(f"refusing to overwrite {existing}")

        out.mkdir(parents=True, exist_ok=True)
        pt_path.write_bytes(pt_bytes)
        shutil.copyfile(staged_onnx, onnx_path)
        card_path.write_text(_dump(card), encoding="utf-8")
        index_path.write_text(_dump(index), encoding="utf-8")
    return card_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) and exc.code else 2
    try:
        card_path = _release(args)
    except (ReleaseError, ValueError) as exc:
        print(f"release_model: refused: {exc}", file=sys.stderr)
        return 1
    print(f"release_model: wrote {card_path.name}", file=sys.stderr)
    return 0


__all__: Final[tuple[str, ...]] = (
    "ReleaseError", "ReleaseWeights", "ParityResult", "build_card", "build_net",
    "check_export_parity", "load_release_weights", "main", "match_config",
    "model_id", "parity_rtol",
)

if __name__ == "__main__":
    raise SystemExit(main())
