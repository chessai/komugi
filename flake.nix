{
  description = "komugi dev shells: python 3.12 + cython/pytest/mypy/numpy + torch -- default shell is cuda, `.#rocm` is the AMD reference shell, `.#v1` is the frozen v1 reference runtime (KOM-023b)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/867dcbc30bafe3c862ef88620f2e7a109d7d3be5";
    # KOM-1215: pinned separately from `nixpkgs` above. Hydra only caches the
    # DEFAULT interpreter's package set at the commits it actually evaluates;
    # this flake's main `nixpkgs` is a raw commit used with python312, so its
    # torch (CPU or CUDA) was never in any public cache (KOM-1215 findings,
    # 2026-11-21 dry-runs). `nixpkgs-test` is a nixos-26.05 channel revision
    # used below with that channel's DEFAULT `python3` (3.13) specifically
    # because that combination is the one Hydra actually built and cached --
    # swapping back to python312, or to this flake's own `nixpkgs` pin, would
    # silently reintroduce local CUDA/torch/triton builds.
    nixpkgs-test.url = "github:NixOS/nixpkgs/94c3211b0ee8ba969a3f12eb4aff8ba0f72971dd";
  };

  outputs = { self, nixpkgs, nixpkgs-test }:
    let
      system = "x86_64-linux";
      # allowUnfree is needed at *eval* time by the cuda shell only: the cuda
      # toolkit/cudnn closure torch pulls in is unfree. Scoped to this flake's
      # own nixpkgs instantiation; it does not change what the rocm shell
      # builds.
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };

      # KOM-024 baseline toolchain, extended in KOM-020a with torch so the
      # trainer has something to run against, and switched in KOM-020b to the
      # ROCm build: this machine has an AMD Radeon PRO W6400 (gfx1034), so the
      # GPU path is ROCm/HIP rather than CUDA proper. An NVIDIA card is on the
      # way, so the torch derivation is a parameter -- the two shells share one
      # package list and cannot drift.
      mkPythonEnv = torchFor: pkgs.python312.withPackages (ps: [
        ps.cython
        ps.pytest
        ps.mypy
        ps.numpy
        (torchFor ps) # rocm or cuda build; see tests/env/test_torch.py
        ps.setuptools
        ps.beartype # runtime typechecker behind the shapecheck seam
        ps.jaxtyping # shape/dtype annotations, only enforced when KOMUGI_TYPECHECK=1
      ]);

      # extraHook is prepended to the banner echo (empty for shells that need
      # no GPU env fixups).
      mkKomugiShell = { torchFor, banner, extraHook }:
        pkgs.mkShell {
          # pythonEnv first so its `python`/`python3` win on PATH.
          packages = [ (mkPythonEnv torchFor) pkgs.python312 ];

          shellHook = extraHook + ''
            echo "komugi dev shell | python 3.12 + cython pytest mypy numpy beartype jaxtyping | ${banner} | ci: bash ci/check.sh"
          '';
        };

      # ---------------------------------------------------------------------
      # KOM-023b: the v1 reference runtime, `nix develop .#v1`.
      #
      # This shell exists to run ONE program -- src/komugi/eval/v1runner, the
      # far end of kom-023 §D1's subprocess protocol -- against the frozen v1
      # sources under docs/reference-v1/. It is a SIBLING of the two shells
      # above and shares nothing with them on purpose: §D1 chose a subprocess
      # boundary precisely so that "one interpreter forces one torch" stops
      # being true (kom-023 §D1, the subprocess-vs-in-process table, row
      # "torch/numpy pins"), and a shell that reused mkPythonEnv would put the
      # pin back.
      #
      # Every pin below is derived from a file in docs/reference-v1 that
      # demands it. Nothing here is a guess dressed as a requirement; where the
      # tree does not demand a version, that is said rather than papered over.
      #
      #   python311        -- deliberately NOT komugi's 3.12. Nothing in the v1
      #                       tree pins an interpreter (its newest syntax is
      #                       `from __future__ import annotations` plus PEP-604
      #                       unions in annotations, e.g. `board: Board | None`
      #                       at docs/reference-v1/alphazero-env/gungi.py:200,
      #                       which needs only 3.7 under that future import).
      #                       A DIFFERENT generation from the default shell is
      #                       the point: ci/check.sh builds komugi's engine
      #                       in place as .cpython-312-*.so, so a 3.11 v1
      #                       process cannot import komugi's compiled engine
      #                       even by accident. §D1's crash/ABI isolation
      #                       becomes a property of the build rather than a
      #                       convention the runner is trusted to keep.
      #   numpy            -- docs/reference-v1/alphazero-env/observation.py:14
      #                       (`import numpy as np`), action_encoding.py:22,
      #                       gungi.py:22, and the C API via `cimport numpy as
      #                       np` in cyboard.pyx:9, cymovegen.pyx:15 and
      #                       cyengine.pyx:45. Generation: the tree uses only
      #                       API that survived numpy 2 -- a full sweep of its
      #                       `np.*` uses is {all, any, ndarray, nonzero,
      #                       float32, zeros, zeros_like, ones, uint8, int8,
      #                       int32, array, array_equal, asarray, where,
      #                       frombuffer, count_nonzero, get_include, random,
      #                       testing} and none of the removed aliases
      #                       (np.float_, np.bool8, np.object, np.int, np.NaN,
      #                       np.product, np.alltrue) appears anywhere. So this
      #                       shell takes nixpkgs' numpy unpinned, and the
      #                       claim is falsifiable rather than asserted: the
      #                       reference tree ships its own suite
      #                       (docs/reference-v1/tests/, 13 files) and KOM-023c
      #                       runs it here.
      #   cython           -- docs/reference-v1/alphazero-env/{cyboard,
      #                       cymovegen,cyengine}.pyx and their three .pxd
      #                       headers must be compiled before
      #                       gungi.py:18-20 can import them.
      #   setuptools       -- the build driver for those three extensions. The
      #                       reference tree ships NO setup.py and NO
      #                       pyproject.toml (docs/reference-v1 holds only
      #                       README.md, alphazero-env/, gungi-py/, gungi-js/
      #                       and tests/), so assembling and building them is
      #                       the operator's step; this is what it needs.
      #   pytest           -- docs/reference-v1/tests/ is the reference's own
      #                       suite and is the only evidence available that a
      #                       locally built extension IS the frozen v1.
      #   torch            -- NOT demanded by any file in the tree: a sweep of
      #                       every import in alphazero-env/ and gungi-py/
      #                       finds no `import torch`. It is demanded by v1's
      #                       NETWORK and MCTS, which are kom-023 §0.2's U-A --
      #                       "there is no NNet.py, no MCTS.py, no model
      #                       definition" under docs/reference-v1. So this pin
      #                       is PROVISIONAL and U-C stays open: it is the CPU
      #                       build of the same nixpkgs, chosen so that `.#v1`
      #                       drags in neither the ROCm nor the CUDA closure
      #                       the other two shells carry. That is not thrift --
      #                       §D1's `hello` device block reports the v1
      #                       process's device and cuDNN state (§D5, review
      #                       F19), and a CPU-only shell makes that block a
      #                       constant instead of a function of which
      #                       accelerator happened to be visible.
      mkV1PythonEnv = pkgs.python311.withPackages (ps: [
        ps.numpy
        ps.cython
        ps.setuptools
        ps.pytest
        ps.torch
      ]);
      cudaShell = mkKomugiShell {
        torchFor = ps: ps.torch.override { cudaSupport = true; };
        banner = "torch (cuda)";
        extraHook = "";
      };

      # `nix develop .#onnx` -- KOM-1054: the cuda shell's python env plus
      # `ps.onnx`, nothing else. A SEPARATE `mkShell`/`withPackages` call
      # rather than a parameter threaded through `mkPythonEnv`/`mkKomugiShell`,
      # so the default/cuda and rocm shells' derivations cannot be perturbed
      # by this one existing -- their expressions above are untouched. The
      # torch expression is copy-identical to `cudaShell`'s (same
      # `cudaSupport = true` override), so this shell assembles an env around
      # the same already-built torch derivation rather than asking for a
      # different one.
      onnxPythonEnv = pkgs.python312.withPackages (ps: [
        ps.cython
        ps.pytest
        ps.mypy
        ps.numpy
        (ps.torch.override { cudaSupport = true; })
        ps.setuptools
        ps.beartype
        ps.jaxtyping
        ps.onnx
      ]);
      onnxShell = pkgs.mkShell {
        packages = [ onnxPythonEnv pkgs.python312 ];
        shellHook = ''
          echo "komugi onnx dev shell | python 3.12 + cython pytest mypy numpy beartype jaxtyping onnx | torch (cuda) | ci: bash ci/check.sh"
        '';
      };

      # `nix develop .#test` -- KOM-1215: a CPU-only shell for gates, fixture
      # exports and CI, so they never compile CUDA (or anything else) locally.
      # It is a SEPARATE `mkShell`/`withPackages`/nixpkgs instantiation, built
      # off `nixpkgs-test` rather than this flake's own `pkgs`, so the
      # default/cuda/rocm/onnx/v1 shells' expressions and derivations above
      # are untouched by this one existing (same reason onnxShell is its own
      # `mkShell` rather than a parameter threaded through `mkKomugiShell`).
      # Uses that channel's DEFAULT `python3` (3.13 on nixos-26.05), NOT
      # python312 and NOT a `cudaSupport`/other torch override -- see the
      # `nixpkgs-test` input comment above for why only this exact
      # interpreter/channel/package combination is cached.
      #
      # KOM-1185 unit 0: `testPkgs.nodejs_24` is added alongside `testPython`
      # so this shell also carries node/npm, for the TS engine package's
      # build/pack gates (design docs/design/ts-engine-library.md section 9,
      # Unit 0) -- fully cached apart from its small wrapper derivation.
      testPkgs = nixpkgs-test.legacyPackages.${system};
      # Shared with `.#ci`'s own python env (KOM-1221 unit 1b, section 4) so
      # the two package lists cannot drift -- `.#ci` is that same list plus
      # `ps.playwright`.
      testPyPkgs = ps: [
        ps.cython
        ps.pytest
        ps.mypy
        ps.numpy
        ps.torch
        ps.setuptools
        ps.beartype
        ps.jaxtyping
        ps.onnx
      ];
      testPython = testPkgs.python3.withPackages testPyPkgs;
      testShell = testPkgs.mkShell {
        packages = [ testPython testPkgs.nodejs_24 ];
        # KOM-1221 D15: this shell's torch is a CPU-only build, never
        # hip/cuda, so tests/env/test_torch.py is told both the version and
        # the backend to expect instead of enforcing the training shells'
        # hip-xor-cuda pin against a build that has neither.
        KOMUGI_EXPECTED_TORCH_VERSION = testPkgs.python3Packages.torch.version;
        KOMUGI_EXPECTED_TORCH_BACKEND = "cpu";
        shellHook = ''
          echo "komugi test dev shell | nixos-26.05 python3 (3.13) + cython pytest mypy numpy beartype jaxtyping onnx | torch (cpu, fully substituted) | node/npm (nodejs_24, for the TS engine package's build/pack gates, KOM-1185) | ci: bash ci/check.sh"
        '';
      };

      # `nix develop .#ci` -- KOM-1221 unit 1b, section 4: the release-CI
      # toolchain (mock site, e2e, release packaging). A SEPARATE `mkShell`
      # off `nixpkgs-test`/`testPkgs` (same pattern as `onnxShell`/`testShell`
      # above), so `.#test`'s drvPath may move with it but
      # `default`/`cuda`/`rocm`/`onnx` cannot be perturbed by this shell
      # existing.
      ciPython = testPkgs.python3.withPackages (ps: testPyPkgs ps ++ [ ps.playwright ]);

      # onnxruntime-web 1.31.1, fetched once as a fixed-output derivation so
      # the engine package's build/test and the mock site never need a real
      # `npm install` to resolve it (section 4, G10a).
      ortWebTgz = testPkgs.fetchurl {
        url = "https://registry.npmjs.org/onnxruntime-web/-/onnxruntime-web-1.31.1.tgz";
        hash = "sha512-5GeH+bWQ+HxM10z7erGlsWqryrwMbSvm/1tkIsKTT48g0fI19VXVVC435t4t7VsPdSQeS7NmuZPXrQ73k82URw==";
      };
      ortWeb = testPkgs.runCommand "onnxruntime-web-1.31.1" { } ''
        mkdir -p $out && tar xzf ${ortWebTgz} -C $out --strip-components=1
      '';

      ciShell = testPkgs.mkShell {
        packages = [
          ciPython
          testPkgs.nodejs_24
          testPkgs.typescript
          testPkgs.esbuild
          testPkgs.jq
          testPkgs.gh
        ];
        KOM_TSC = "${testPkgs.typescript}/lib/node_modules/typescript/bin/tsc";
        KOM_ORT_WEB_DIR = ortWeb;
        ORTWEB_DIR = testPkgs.linkFarm "ortweb" [
          { name = "node_modules/onnxruntime-web"; path = ortWeb; }
        ];
        PLAYWRIGHT_BROWSERS_PATH = testPkgs.playwright-driver.browsers-chromium;
        PLAYWRIGHT_SKIP_VALIDATE_HOST_REQUIREMENTS = "true";
        KOMUGI_EXPECTED_TORCH_BACKEND = "cpu";
        KOMUGI_EXPECTED_TORCH_VERSION = testPkgs.python3Packages.torch.version;
        shellHook = ''
          echo "komugi release-ci dev shell | nixos-26.05 python3 (3.13) + playwright | node 24 + typescript esbuild jq gh | onnxruntime-web 1.31.1 (KOM_ORT_WEB_DIR/ORTWEB_DIR) | ci: bash ci/release/engine.sh"
        '';
      };
    in
    {
      devShells.${system} = {
        # `nix develop` -- CUDA is the default on this box (RTX 5070). No
        # HSA_OVERRIDE_GFX_VERSION here; that is a ROCm-only knob.
        default = cudaShell;
        cuda = cudaShell;

        # `nix develop .#rocm` -- the AMD reference shell, kept for the
        # pod/AMD path. Not built by default on this box.
        rocm = mkKomugiShell {
          torchFor = ps: ps.torchWithRocm;
          banner = "torch (rocm)";
          extraHook = ''
            # gfx1034 spoofed as gfx1030 -- RDNA2 workaround, verified on the W6400.
            export HSA_OVERRIDE_GFX_VERSION=10.3.0
          '';
        };

        # `nix develop .#onnx` -- see `onnxShell` above.
        onnx = onnxShell;

        # `nix develop .#test` -- see `testShell` above (KOM-1215).
        test = testShell;

        # `nix develop .#ci` -- see `ciShell` above (KOM-1221 unit 1b).
        ci = ciShell;


        # `nix develop .#v1` -- the frozen v1 reference runtime. See
        # mkV1PythonEnv above for the derivation of every pin.
        #
        # Two exports, and both are load-bearing (impl review v2, F7 and F9):
        #
        #   PYTHONPATH -- this repository is a src layout with nothing
        #     pip-installed, and ci/check.sh:11 is the only other place $PWD/src
        #     reaches PYTHONPATH -- in the DEFAULT shell, not this one. Without
        #     this line the very first command the banner prints, and the first
        #     command KOM-023c runs, dies with `ModuleNotFoundError: No module
        #     named komugi` before any of reference.py's careful refusal
        #     machinery gets to say anything useful. A shell that cannot run the
        #     one program it exists for is not a shell.
        #
        #   KOMUGI_V1_REFERENCE_ROOT -- the ASSEMBLED tree, never
        #     docs/reference-v1. The copy's own README says "NEVER edit these;
        #     NEVER import them into komugi code" and §D2.2 pre-registers that
        #     v1 is run from a real checkout and never from the copy; pointing
        #     the default at the copy is harmless only until someone assembles a
        #     package layout underneath it, at which point it becomes a
        #     pre-registration violation that nothing would detect. build/ is
        #     this repo's throwaway build directory, so the assembled tree is a
        #     build artefact rather than an edit to the frozen copy. The runner
        #     reads this only as a default; an explicit --reference-root always
        #     wins, so a stale export can never silently redirect a rated run.
        v1 = pkgs.mkShell {
          packages = [ mkV1PythonEnv pkgs.python311 ];

          shellHook = ''
            export PYTHONPATH="$PWD/src''${PYTHONPATH:+:$PYTHONPATH}"
            export KOMUGI_V1_REFERENCE_ROOT="$PWD/build/v1-reference"
            echo "komugi v1 reference shell | python 3.11 + numpy cython setuptools pytest torch (cpu)"
            echo "  komugi on path : $PWD/src (src layout, nothing pip-installed)"
            echo "  reference root : $KOMUGI_V1_REFERENCE_ROOT (the ASSEMBLED tree; --reference-root overrides)"
            echo "  runner         : python -m komugi.eval.v1runner --check-imports"
            echo "        --check-imports names exactly what is missing, in the order it is missing it."
          '';
        };
      };
    };
}
