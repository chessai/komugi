"""Fixtures for the ported v1 engine tests.

The v1 suite loads oracle goldens (generated from gungi.js@3.4.11) through a
``load_golden`` helper that lived in its conftest. The golden JSON corpus was
not part of the KOM-001a copy-over, so this port resolves it from
``tests/engine/goldens/`` and skips the golden-driven tests when the corpus is
absent. Dropping the corpus in place re-arms those tests with no code change.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

GOLDENS_DIR = Path(__file__).parent / "goldens"


def load_golden(name: str) -> Any:
    """Return the parsed golden JSON *name*, skipping if the corpus is absent."""
    path = GOLDENS_DIR / name
    if not path.is_file():
        pytest.skip(f"oracle golden {name} not available (goldens corpus not vendored)")
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)
