#!/usr/bin/env bash
# komugi CI baseline (KOM-024).
#
# Run from the repo root, inside the dev shell:
#     nix develop -c bash ci/check.sh
set -euo pipefail

# src layout: nothing is pip-installed, so the compiled extension and the
# package itself are found through PYTHONPATH. Absolute, because the
# spawn-based tests hand their environment to fresh interpreters.
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"

echo "==> cython build (in place)"
# --build-temp keeps the object-file scratch inside src/ (ignored there)
# instead of dropping an untracked build/ at the repo root.
python src/komugi/build_ext.py build_ext --inplace --build-temp src/build

echo "==> mypy (strict; config in pyproject.toml)"
mypy

# KOM-236: the gate sandbox sets KOMUGI_GATE_SANDBOX=1 to say it is running
# inside the sandbox -- no CUDA, single-threaded BLAS, no room for the
# bridge's spawned subprocesses. The tests that cannot pass there carry the
# sandbox_unsafe marker (registered in pyproject.toml) and are deselected only
# in that case; with the variable unset the invocation below is unchanged.
if [[ "${KOMUGI_GATE_SANDBOX:-}" == "1" ]]; then
  echo "==> pytest (gate sandbox: deselecting sandbox_unsafe)"
  python -m pytest -q -m "not sandbox_unsafe"
else
  echo "==> pytest"
  python -m pytest -q
fi

echo "==> stubtest: hand-written stubs vs. the compiled modules"
# MYPYPATH=src so stubtest finds the .pyi files that sit next to their .pyx;
# the allowlist covers Cython's generated module attributes, and
# --ignore-unused-allowlist keeps entries for artifacts a given .pyx happens
# not to emit from failing the run.
MYPYPATH=src python -m mypy.stubtest \
  komugi._probe \
  komugi.engine.cyboard \
  komugi.engine.cymovegen \
  komugi.engine.cyengine \
  --allowlist ci/stubtest_allowlist.txt \
  --ignore-unused-allowlist

echo "==> all checks passed"
