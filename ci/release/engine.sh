#!/usr/bin/env bash
# komugi release CI: build and gate the TS engine package under `.#ci`
# (KOM-1221 section 9, unit 1b). Run as:
#   nix develop .#ci -c bash ci/release/engine.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=ci/release/lib.sh
source "$SCRIPT_DIR/lib.sh"

for var in KOM_ORT_WEB_DIR ORTWEB_DIR KOM_TSC; do
  if [ -z "${!var:-}" ]; then
    echo "==> engine.sh: $var is unset (expected from \`nix develop .#ci\`)" >&2
    exit 1
  fi
done

ensure_ort_web_linked

cd packages/engine
tsc -p tsconfig.json
npm run build
npm test
npm run test:pack
npm run typecheck:strict
