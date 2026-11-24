"""The ``search_mode`` column's vocabulary. A leaf: this module imports nothing.

``0`` full, ``1`` fast (design F1) -- the same codes
:data:`komugi.selfplay.records.SCHEMA`'s docstring documents as the key the P0b
down-weight-versus-discard A/B is stratified on. It lives at the package root,
not under ``komugi/selfplay/``, so that a trainer-side import creates no edge
into the self-play *package*; see ``docs/design/kom-036-fast-mode-ab.md`` §1.2a.

Python draws its import edge at **package** granularity, which is what makes
the location rather than the file contents load-bearing: ``from
komugi.selfplay.modes import ...`` would execute ``komugi/selfplay/__init__.py``
in full -- adapter, inference, memgrid, orchestrate, worker -- three of which
import back into :mod:`komugi.train`, and the resulting package cycle does not
raise only because ``train/__init__.py`` happens to load
:mod:`komugi.train.shards` before :mod:`komugi.train.wiring`. Here the edge is
``komugi.train -> komugi``, and ``komugi/__init__.py`` imports nothing at all.
"""

from __future__ import annotations

from typing import Final

MODE_FULL: Final[int] = 0
"""``search_mode`` for a full search (design D4/F1)."""

MODE_FAST: Final[int] = 1
"""``search_mode`` for a fast search."""
