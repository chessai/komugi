# komugi release CI shared helpers (KOM-1221 section 3). Sourced, never run
# directly -- callers are bash scripts under ci/release/ that need komugi's
# Python package importable.

# ensure_ext -- build the Cython extension in place and put src/ on
# PYTHONPATH. A separate --build-temp from ci/check.sh's own src/build keeps
# this from colliding with a concurrent `bash ci/check.sh` run. Idempotent:
# build_ext --inplace re-runs safely, and PYTHONPATH is extended only once
# per shell since every caller sources this file exactly once.
ensure_ext() {
  echo "==> ensure_ext: cython build (in place)"
  python src/komugi/build_ext.py build_ext --inplace --build-temp build/release-ext
  # setuptools skips creating --build-temp when every extension is already
  # up to date (nothing to compile, so no object-file scratch dir is made);
  # guarantee the directory exists regardless of that cache state.
  mkdir -p build/release-ext
  export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
}

# ensure_ort_web_linked -- symlink `.#ci`'s onnxruntime-web FOD
# (KOM_ORT_WEB_DIR) into packages/engine/node_modules, so the engine's build
# and the mock site worker can resolve the bare `onnxruntime-web` import
# without a real `npm install` (section 4). Paths are relative to the repo
# root, where every caller runs.
ensure_ort_web_linked() {
  if [ -z "${KOM_ORT_WEB_DIR:-}" ] || [ ! -d "$KOM_ORT_WEB_DIR" ]; then
    echo "==> ensure_ort_web_linked: KOM_ORT_WEB_DIR is unset or not a directory: ${KOM_ORT_WEB_DIR:-<unset>}" >&2
    exit 1
  fi
  echo "==> ensure_ort_web_linked: linking onnxruntime-web"
  mkdir -p packages/engine/node_modules
  ln -sfn "$KOM_ORT_WEB_DIR" packages/engine/node_modules/onnxruntime-web
}
