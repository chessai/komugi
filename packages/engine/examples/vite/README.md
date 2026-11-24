# @komugi-ai/engine — Vite example

A minimal Vite page that serves the files written by the engine's assets
CLI, loads a released model with `loadModel`, and plays one move from the
Advanced starting position at the default strength level.

## Run it

1. Build the engine package (this example depends on it through
   `"file:../.."`):

   ```sh
   cd packages/engine
   npm run build
   ```

2. Install the example's dependencies:

   ```sh
   cd examples/vite
   npm install --install-links
   ```

3. Copy the model and the onnxruntime-web runtime files into `public/`,
   from a model release directory you already have on disk:

   ```sh
   npx @komugi-ai/engine assets public --release-dir <dir>
   ```

4. Start the dev server:

   ```sh
   npm run dev
   ```

## Why `--install-links`

A `file:` dependency resolves through a symlink by default: npm links
`node_modules/@komugi-ai/engine` to `packages/engine` itself. The assets
CLI resolves `onnxruntime-web` from the engine's realpath, so through that
symlink it would search above `packages/engine`, where no
`onnxruntime-web` is installed, and fail with "onnxruntime-web not found".
A plain `npm install` would reproduce this failure. `--install-links`
makes npm install a real copy of the engine under this example's
`node_modules/`, beside the example's own `onnxruntime-web`, so the
lookup lands on that install.

## Wiring the paths

`src/main.ts` fetches `ort-assets.json` from an absolute URL and resolves
every path in it, and the model card URL, against the URL it was served
from, with `new URL(path, ortAssetsUrl).href`. Never pass a relative path
into `ortAssets` or `manifestUrl`.

## Checks

The example is type-checked against the built package types (run from
`packages/engine`, after `npm run build`):

```sh
tsc -p examples/vite/tsconfig.json
```

A production build is a manual check: `npm run build` in this directory
must finish with no errors.

`node_modules/`, `public/` and `dist/` are git-ignored.
