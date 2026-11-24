"""Build komugi's Cython extension modules in place.

Run from the repo root::

    python src/komugi/build_ext.py build_ext --inplace

Each extension is named explicitly (rather than letting cythonize derive
``src.komugi._probe`` from the path) so that ``--inplace`` drops the shared
object next to the .pyx, where ``PYTHONPATH=src`` finds it.
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

# Per-file directives in the .pyx sources take precedence over these.
COMPILER_DIRECTIVES: dict[str, Any] = {"language_level": "3"}

# cyboard.pxd and the two modules that ``cimport`` it declare numpy types, so
# every engine extension needs numpy's C headers on the include path.
NUMPY_INCLUDE = np.get_include()

EXTENSIONS = [
    Extension("komugi._probe", ["src/komugi/_probe.pyx"]),
    Extension(
        "komugi.engine.cyboard",
        ["src/komugi/engine/cyboard.pyx"],
        include_dirs=[NUMPY_INCLUDE],
    ),
    Extension(
        "komugi.engine.cymovegen",
        ["src/komugi/engine/cymovegen.pyx"],
        include_dirs=[NUMPY_INCLUDE],
    ),
    Extension(
        "komugi.engine.cyengine",
        ["src/komugi/engine/cyengine.pyx"],
        include_dirs=[NUMPY_INCLUDE],
    ),
]


def main() -> None:
    """Invoke setuptools with komugi's extension list."""
    setup(
        packages=["komugi", "komugi.engine"],
        package_dir={"": "src"},
        ext_modules=cythonize(
            EXTENSIONS,
            compiler_directives=COMPILER_DIRECTIVES,
            # ``from komugi.engine.cyboard cimport Board`` resolves the .pxd
            # through this path, the same way PYTHONPATH=src resolves the
            # python-level package.
            include_path=["src"],
        ),
        script_args=sys.argv[1:] or ["build_ext", "--inplace"],
    )


if __name__ == "__main__":
    main()
