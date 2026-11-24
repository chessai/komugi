#!/usr/bin/env bash
set -euo pipefail

# KOM-1221 section 6: publish the release. `--dry-run` (or PUBLISH_DRY_RUN=1,
# the act path under section 8) prints the gh invocation and exits 0 without
# calling gh. Exactly one of --verify-tag / --target is ever appended
# (G11b): on dispatch the tag does not exist yet, so --verify-tag would fail.

: "${EXPECTED_ID:?}"
: "${RELEASE_DIR:?}"

dry_run=0
if [ "${1:-}" = "--dry-run" ] || [ "${PUBLISH_DRY_RUN:-}" = "1" ]; then
  dry_run=1
fi

assets=(
  "$RELEASE_DIR/$EXPECTED_ID.pt"
  "$RELEASE_DIR/$EXPECTED_ID.onnx"
  "$RELEASE_DIR/$EXPECTED_ID.card.json"
  "$RELEASE_DIR/models.json"
)

args=(release create "model-$EXPECTED_ID" "${assets[@]}" --title "$EXPECTED_ID")

if [[ "${GITHUB_REF_NAME:-}" == model-* ]]; then
  args+=(--verify-tag)
else
  args+=(--target "${GITHUB_SHA:?}")
fi

if [ "$dry_run" = 1 ]; then
  printf '==> publish-release: dry run, would run:\n'
  printf 'gh'
  for a in "${args[@]}"; do
    printf ' %q' "$a"
  done
  printf '\n'
  exit 0
fi

echo "==> publish-release: creating release model-$EXPECTED_ID"
gh "${args[@]}"
