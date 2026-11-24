"""komugi: a typed, Cython-accelerated rewrite of the v1 engine.

The package is import-light on purpose: heavy dependencies (compiled
extensions, inference backends) are pulled in by the submodules that need
them, so ``import komugi`` stays cheap for tooling and subprocess bootstraps.
"""

from __future__ import annotations

__version__: str = "0.0.1"
