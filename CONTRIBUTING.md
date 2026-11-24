# Contributing to Komugi

## Setup

```sh
nix develop .#test
python src/komugi/build_ext.py build_ext --inplace
```

`nix develop .#test` provides Python, Cython, mypy, pytest, and Node/npm.
The build step compiles the Cython extensions in place; rebuild it after
pulling changes to `src/komugi/engine/`.

## Python tests

```sh
python -m pytest
```

[`pyproject.toml`](pyproject.toml)'s `[tool.pytest.ini_options]` sets
`testpaths = ["tests", "tools"]`. One marker is registered,
`sandbox_unsafe`, for tests that cannot pass in a sandbox with no CUDA,
single-threaded BLAS, and no room for the training bridge's subprocesses;
`bash ci/check.sh` deselects it only when `KOMUGI_GATE_SANDBOX=1` is set, and
runs the full suite otherwise. Run `bash ci/check.sh` before opening a PR —
it also runs `mypy --strict` and the stub-consistency check
(`mypy.stubtest`) against the compiled engine modules.

## TypeScript checks

From `packages/engine`:

```sh
npm test                 # default suite against committed fixtures
npm run typecheck:strict # stricter, consumer-parity typecheck
npm run test:pack        # checks what actually ships in the published tarball
```

`npm test` skips the handful of suites backed by a real ONNX model, since
no model is committed to the package.

## Code style

Python is `mypy --strict` (see `[tool.mypy]` in `pyproject.toml`); new code
must type-check cleanly. A few modules carry per-module overrides for
verbatim-ported legacy code (`komugi.engine.*` and its test suites) — new
code does not get that exemption. TypeScript is checked with `tsc` in
strict mode (`npm run typecheck:strict`).

## Pull requests

- Any behaviour change needs a test. Bug fixes should add a test that fails
  without the fix.
- Several suites compare output against recorded fixtures checked into the
  repository — for example `packages/engine/test/fixtures/` and the
  `tests/**/goldens` directories (`tests/engine/goldens/`,
  `tests/encoding/goldens/`, and similar). Keep these green; a PR that
  changes behaviour covered by a fixture should update the fixture in the
  same PR, with an explanation of why the old one no longer applies.
- Regenerating some fixtures needs tooling that is not part of this
  repository. If you need a fixture changed and cannot regenerate it
  yourself, open an issue describing what should change and why.
