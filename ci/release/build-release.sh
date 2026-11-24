#!/usr/bin/env bash
# komugi release CI: build and verify one release from a staged candidate
# (KOM-1221 design docs/design/kom1221-release-ci.md section 3.2, section 9
# unit 2). Run from the repo root, inside the dev shell:
#   CANDIDATE_DIR=... RELEASE_DIR=... MOCK_PUBLIC_DIR=... EXPECTED_ID=... \
#     nix develop .#ci -c bash ci/release/build-release.sh
#   RELEASE_DIR=... MOCK_PUBLIC_DIR=... \
#     nix develop .#ci -c bash ci/release/build-release.sh --fixture
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=ci/release/lib.sh
source "$SCRIPT_DIR/lib.sh"

FIXTURE=0
if [ "${1:-}" = "--fixture" ]; then
  FIXTURE=1
fi

for var in RELEASE_DIR MOCK_PUBLIC_DIR; do
  if [ -z "${!var:-}" ]; then
    echo "==> build-release.sh: $var is unset (expected from the caller's env)" >&2
    exit 1
  fi
done

ensure_ext
ensure_ort_web_linked

if [ "$FIXTURE" = 1 ]; then
  CANDIDATE_DIR="$(mktemp -d)"
  export CANDIDATE_DIR
  echo "==> build-release.sh --fixture: writing a fixture candidate into $CANDIDATE_DIR"
  python "$SCRIPT_DIR/make-fixture.py" "$CANDIDATE_DIR"
  EXPECTED_ID="$(jq -r .expectedId "$CANDIDATE_DIR/candidate.json")"
  export EXPECTED_ID
else
  for var in CANDIDATE_DIR EXPECTED_ID; do
    if [ -z "${!var:-}" ]; then
      echo "==> build-release.sh: $var is unset (expected from the caller's env)" >&2
      exit 1
    fi
  done
fi

CANDIDATE_JSON="$CANDIDATE_DIR/candidate.json"
CHECKPOINT_ASSET="$(jq -r .checkpointAsset "$CANDIDATE_JSON")"
CANDIDATE_SOURCE_SHA256="$(jq -r .sourceCheckpointSha256 "$CANDIDATE_JSON")"
CAMPAIGN="$(jq -r .campaign "$CANDIDATE_JSON")"
RULES_PROFILE="$(jq -r .rulesProfile "$CANDIDATE_JSON")"
MODES_TRAINED="$(jq -r '.modesTrained | join(" ")' "$CANDIDATE_JSON")"
CANDIDATE_EXPECTED_ID="$(jq -r .expectedId "$CANDIDATE_JSON")"
CANDIDATE_ONNX_SHA256="$(jq -r .onnxSha256 "$CANDIDATE_JSON")"

# Step 2 (section 3.2): checkpoint sha256, before anything is built -- on a
# mismatch RELEASE_DIR stays empty.
echo "==> build-release.sh: checking staged checkpoint sha256"
ACTUAL_CHECKPOINT_SHA256="$(sha256sum "$CANDIDATE_DIR/$CHECKPOINT_ASSET" | cut -d' ' -f1)"
if [ "$ACTUAL_CHECKPOINT_SHA256" != "$CANDIDATE_SOURCE_SHA256" ]; then
  echo "==> build-release.sh: sourceCheckpointSha256 mismatch: candidate.json says $CANDIDATE_SOURCE_SHA256, $CHECKPOINT_ASSET hashes to $ACTUAL_CHECKPOINT_SHA256" >&2
  exit 1
fi

# Step 3: build the release.
echo "==> build-release.sh: running release_model.py"
python tools/release_model.py \
  --ema-checkpoint "$CANDIDATE_DIR/$CHECKPOINT_ASSET" \
  --campaign "$CAMPAIGN" \
  --rules-profile "$RULES_PROFILE" \
  --modes-trained $MODES_TRAINED \
  --out "$RELEASE_DIR"

# Step 4: the built id and onnx sha256 must agree with EXPECTED_ID and
# candidate.json's own expectedId/onnxSha256 (G8a -- never satisfied by
# comparing only EXPECTED_ID).
BUILT_ID="$(jq -r '.models | keys[0]' "$RELEASE_DIR/models.json")"
BUILT_ONNX_SHA256="$(jq -r .assets.onnx.sha256 "$RELEASE_DIR/$BUILT_ID.card.json")"

echo "==> build-release.sh: checking id/sha agreement"
if [ "$BUILT_ID" != "$EXPECTED_ID" ]; then
  echo "==> build-release.sh: EXPECTED_ID mismatch: EXPECTED_ID=$EXPECTED_ID, built $BUILT_ID" >&2
  exit 1
fi
if [ "$BUILT_ID" != "$CANDIDATE_EXPECTED_ID" ]; then
  echo "==> build-release.sh: expectedId mismatch: candidate.json's expectedId=$CANDIDATE_EXPECTED_ID, built $BUILT_ID" >&2
  exit 1
fi
if [ "$BUILT_ONNX_SHA256" != "$CANDIDATE_ONNX_SHA256" ]; then
  echo "==> build-release.sh: onnxSha256 mismatch: candidate.json's onnxSha256=$CANDIDATE_ONNX_SHA256, built $BUILT_ONNX_SHA256" >&2
  exit 1
fi

# Step 5: "latest ships" applied once, here -- komugi-assets.mjs only copies
# a recommended entry whose value equals the id it is packaging, so an empty
# `recommended` (release_model.py always writes `{}`) ships nothing
# selectable.
case " $MODES_TRAINED " in
  *" advanced "*) ;;
  *)
    echo "==> build-release.sh: refusing to set recommended.advanced: modesTrained ($MODES_TRAINED) has no 'advanced'" >&2
    exit 1
    ;;
esac

echo "==> build-release.sh: setting recommended.advanced = $EXPECTED_ID"
python - "$RELEASE_DIR/models.json" "schemas/models-index.schema.json" "$EXPECTED_ID" <<'PY'
import json
import sys
from pathlib import Path

sys.path.insert(0, "tools")
import schema_check

models_json_path = Path(sys.argv[1])
schema_path = Path(sys.argv[2])
expected_id = sys.argv[3]

index = json.loads(models_json_path.read_text(encoding="utf-8"))
index["recommended"]["advanced"] = expected_id
schema_check.validate(index, schema_check.load_schema(schema_path))
models_json_path.write_text(
    json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

# Step 6 (section 5.3): sha256 + planeMapVersion verified against the onnx
# metadata, independent of what komugi-assets.mjs itself checks.
echo "==> build-release.sh: verify-release.py"
python ci/release/verify-release.py \
  "$RELEASE_DIR/$EXPECTED_ID.card.json" \
  "$RELEASE_DIR/$EXPECTED_ID.onnx"

# Step 7: pack the released model (and the ORT glue) into the mock site's
# public dir. Wasm-only (e2e never exercises WebGPU) -- no --webgpu.
echo "==> build-release.sh: packing assets into $MOCK_PUBLIC_DIR"
node packages/engine/bin/komugi-assets.mjs assets "$MOCK_PUBLIC_DIR" \
  --release-dir "$RELEASE_DIR"

echo "==> build-release.sh: ok, released $EXPECTED_ID"
