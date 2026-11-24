#!/usr/bin/env bash
# komugi release CI: build the mock site's esbuild bundles (KOM-1221 design
# docs/design/kom1221-release-ci.md section 4, section 9 unit 3). Run from
# the repo root:
#   MOCK_PUBLIC_DIR=... nix develop .#ci -c bash ci/release/mock-site.sh
#
# Does not start server.mjs -- the unit 3 gate and e2e.sh do that themselves,
# against a fresh port file.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=ci/release/lib.sh
source "$SCRIPT_DIR/lib.sh"

if [ -z "${MOCK_PUBLIC_DIR:-}" ]; then
  echo "==> mock-site.sh: MOCK_PUBLIC_DIR is unset (expected from the caller's env)" >&2
  exit 1
fi

ensure_ort_web_linked

echo "==> mock-site.sh: bundling page.bundle.js"
esbuild ci/release/mock-site/src/page.ts --bundle --format=esm \
  --platform=browser --conditions=onnxruntime-web-use-extern-wasm \
  --outfile="$MOCK_PUBLIC_DIR/page.bundle.js"

echo "==> mock-site.sh: bundling worker.js + crash_worker.js"
esbuild worker=packages/engine/dist/worker.js \
  crash_worker=ci/release/mock-site/src/crash_worker_entry.ts \
  --bundle --format=esm --platform=browser \
  --conditions=onnxruntime-web-use-extern-wasm --outdir="$MOCK_PUBLIC_DIR"

echo "==> mock-site.sh: copying index.html"
cp ci/release/mock-site/index.html "$MOCK_PUBLIC_DIR/index.html"

echo "==> mock-site.sh: ok"
