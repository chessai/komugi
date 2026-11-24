#!/usr/bin/env bash
# komugi release CI: Python suite (KOM-1221 section 9, unit 1a).
#
# Run from the repo root, inside the .#test shell:
#     nix develop .#test -c bash ci/release/python.sh
set -euo pipefail

source "$(dirname "$0")/lib.sh"
ensure_ext

if [[ -n "${CI:-}" ]]; then
  export KOMUGI_GATE_SANDBOX=1
fi

bash ci/check.sh
