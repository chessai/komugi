"""The ``termination`` column's vocabulary. A leaf: this module imports nothing.

``0`` decisive, ``1`` draw, ``2`` resign (reserved) -- the three codes
:func:`komugi.selfplay.worker.finalize_rows` stamps on every row of a finished
game, and the stratum KOM-160's ``cap_row_value_weight`` keys the WDL loss on.

It lives at the package root, beside :mod:`komugi.modes`, and for exactly that
module's reason: the trainer needs these codes, and importing them from
:mod:`komugi.selfplay.worker` closes a genuine cycle. Python draws its import
edge at **package** granularity, so ``from komugi.selfplay.worker import
TERMINATION_DRAW`` executes ``komugi/selfplay/__init__.py`` in full -- adapter,
inference, memgrid, orchestrate, worker -- and ``worker`` imports
:mod:`komugi.train.shards`, which loads ``komugi/train/__init__.py``, which
loads :mod:`komugi.train.step` and :mod:`komugi.train.wiring`, which would
import back into the self-play package with ``records`` frozen at its own
import line. Here the edge is ``komugi.train -> komugi``, and
``komugi/__init__.py`` imports nothing at all.

A **second** leaf rather than three more constants in :mod:`komugi.modes`
because that module is the ``search_mode`` column's vocabulary and says so in
its first line: one module per column keeps "which codes does this column
take" answerable by opening one file, and keeps a reader of either from having
to know that the two columns are unrelated. Both are re-exported from
:mod:`komugi.selfplay.worker`, which is where every existing importer spells
them.
"""

from __future__ import annotations

from typing import Final

TERMINATION_DECISIVE: Final[int] = 0
"""``termination`` for a game the engine adjudicated as a win for one side."""

TERMINATION_DRAW: Final[int] = 1
"""``termination`` for an adjudicated draw, the engine's turn cap included."""

TERMINATION_RESIGN: Final[int] = 2
"""``termination`` for a resigned game -- **reserved for resign v2**.

Defined so the value set is stated in one place and a reader of a shard can
name what a ``2`` would mean, not because anything writes it: P0b has no
resign, so :func:`komugi.selfplay.worker.finalize_rows` produces only ``0`` and
``1``. The resigned fraction of every P0b window is therefore identically zero,
and code that computes it is armed for KOM-009 rather than reporting a live
measurement.
"""
