# Komugi (コムギ)

Komugi is an AlphaZero-style engine for Gungi, the board game from *Hunter ×
Hunter* (the name comes from a character in that story who plays it). The
rules engine follows [gungi.js](https://github.com/jwyce/gungi.js).

The project has two parts:

- **Training and search**, in Python, under [`src/komugi`](src/komugi):
  the board/move representation, the MCTS-based search, the self-play and
  training pipeline, and the ONNX export used to produce a release.
- **The client engine**, [`@komugi-ai/engine`](packages/engine), a
  TypeScript package that runs a trained network in the browser — board/move
  rules, search, and ONNX Runtime Web evaluation inside a dedicated worker.
  See [`packages/engine/README.md`](packages/engine/README.md) for its API
  and [`packages/engine/docs/integration.md`](packages/engine/docs/integration.md)
  for hosting a release (headers, bundler setup, a checklist).

## Development setup

```sh
nix develop .#test
python src/komugi/build_ext.py build_ext --inplace
python -m pytest
bash ci/check.sh
```

`nix develop .#test` provides Python, the Cython toolchain, and Node/npm.
`ci/check.sh` runs the full baseline: the Cython build, `mypy --strict`,
`pytest`, and a stub-consistency check (`mypy.stubtest`) against the
compiled engine modules.

The TypeScript package has its own checks, run from `packages/engine`
(scripts defined in [`packages/engine/package.json`](packages/engine/package.json)):

```sh
npm test               # default suite against committed fixtures
npm run typecheck:strict
npm run test:pack      # checks what actually ships in the published tarball
```

## Layout

| Path | Contents |
| --- | --- |
| `src/komugi/` | Python training and search engine |
| `packages/engine/` | TypeScript client engine (`@komugi-ai/engine`) |
| `tests/` | Python test suite |
| `schemas/` | JSON Schemas for model releases (`model-card`, `models-index`) |
| `stubs/` | Hand-written type stubs for the compiled Cython modules |
| `scripts/analysis/` | Offline analysis tooling over recorded games |
| `scripts/goldens/` | Generators for committed test fixtures |
| `scripts/oracle/` | Regenerates the golden corpus under `tests/engine/goldens/` against gungi.js |
| `tools/release_model.py`, `tools/schema_check.py` | Turn a checkpoint into a model release, and validate its schemas |
| `ci/release/`, `ci/check.sh` | Release packaging and the CI baseline above |

## Comparing against an earlier engine

`komugi.eval.v1runner` can compare the current engine against an earlier one
that is not included in this repository. It reads that engine's location
from the `KOMUGI_V1_REFERENCE_ROOT` environment variable; without it (or
against a tree that does not look like that engine), it reports a specific
reason it cannot run rather than failing silently.

## License

MIT — see [LICENSE](./LICENSE). See [NOTICE](./NOTICE) for the MIT notices
of alphazero-general and gungi.js, which parts of Komugi derive from.
