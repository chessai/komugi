#!/usr/bin/env bash
set -euo pipefail

# KOM-1221 section 3.1: resolve EXPECTED_ID without circularity. Runs as its
# own step, before python.sh, so the $GITHUB_ENV write here is visible to
# every later step (a same-step write/read can never share a step, G1).

: "${CANDIDATE_DIR:?}"

if [ -f "$CANDIDATE_DIR/candidate.json" ]; then
  echo "==> resolve-candidate: candidate.json already staged at $CANDIDATE_DIR, skipping fetch"
else
  if [[ "${GITHUB_REF_NAME:-}" == model-* ]]; then
    release="candidate-${GITHUB_REF_NAME#model-}"
  else
    release="candidate-pending"
  fi
  echo "==> resolve-candidate: fetching release $release"
  gh release download "$release" --repo "${GITHUB_REPOSITORY:-}" --dir "$CANDIDATE_DIR" --pattern '*'
fi

id=$(jq -r .expectedId "$CANDIDATE_DIR/candidate.json")

if [[ "${GITHUB_REF_NAME:-}" == model-* ]]; then
  tag_id="${GITHUB_REF_NAME#model-}"
  if [ "$tag_id" != "$id" ]; then
    echo "==> resolve-candidate: tag id ($tag_id) != candidate.json expectedId ($id)" >&2
    exit 1
  fi
fi

echo "EXPECTED_ID=$id" >> "${GITHUB_ENV:?}"
