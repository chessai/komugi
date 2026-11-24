#!/usr/bin/env python3
"""verify-release.py -- check one release card against its onnx file
(KOM-1221 design docs/design/kom1221-release-ci.md section 5.3). Run by
build-release.sh (section 3.2 step 6) and e2e-negative.sh (case c, against a
mutated card copy and the unmutated onnx).

Three checks, each exiting 1 and naming both values on failure:
1. card.assets.onnx.sha256 == sha256(the onnx file).
2. card.id's trailing 8 hex chars prefix that same sha256 (section 2.1's
   ``komugi-<campaign>-s<step>-<sha8>`` id grammar).
3. card.compatibilityKey.planeMapVersion == the onnx's own
   metadata_props["komugi.plane_map_version"]
   (onnx_export.PLANE_MAP_VERSION_METADATA_KEY, stamped by export_net) -- the
   only check that can tell planeMapVersion 3 from 4 on an Advanced card; the
   engine itself can't (F4).

Usage: verify-release.py CARD_PATH ONNX_PATH
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Final

_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
_SRC: Final[Path] = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import onnx  # noqa: E402
from komugi.net import onnx_export  # noqa: E402


class VerifyError(Exception):
    """A card/onnx pair failed one of the three checks."""


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify(card_path: Path, onnx_path: Path) -> None:
    card: dict[str, Any] = json.loads(card_path.read_text(encoding="utf-8"))
    actual_sha256 = sha256_hex(onnx_path.read_bytes())

    card_sha256 = card["assets"]["onnx"]["sha256"]
    if actual_sha256 != card_sha256:
        raise VerifyError(
            f"assets.onnx.sha256: card says {card_sha256}, "
            f"{onnx_path} hashes to {actual_sha256}"
        )

    release_id = card["id"]
    id_suffix = release_id[-8:]
    if id_suffix != actual_sha256[:8]:
        raise VerifyError(
            f"id: {release_id!r}'s trailing 8 hex chars are {id_suffix!r}, "
            f"expected the onnx sha256's first 8 chars {actual_sha256[:8]!r}"
        )

    model = onnx.load(str(onnx_path))
    props = {entry.key: entry.value for entry in model.metadata_props}
    raw_version = props.get(onnx_export.PLANE_MAP_VERSION_METADATA_KEY)
    if raw_version is None:
        raise VerifyError(
            "compatibilityKey.planeMapVersion: onnx metadata_props has no "
            f"{onnx_export.PLANE_MAP_VERSION_METADATA_KEY!r} key"
        )
    onnx_version = int(raw_version)
    card_version = card["compatibilityKey"]["planeMapVersion"]
    if card_version != onnx_version:
        raise VerifyError(
            f"compatibilityKey.planeMapVersion: card says {card_version}, "
            f"onnx metadata says {onnx_version}"
        )


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        print("usage: verify-release.py CARD_PATH ONNX_PATH", file=sys.stderr)
        return 2
    card_path, onnx_path = Path(argv[0]), Path(argv[1])
    try:
        verify(card_path, onnx_path)
    except VerifyError as exc:
        print(f"verify-release: refused: {exc}", file=sys.stderr)
        return 1
    print(f"verify-release: ok {card_path.name}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
