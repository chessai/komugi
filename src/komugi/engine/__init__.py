"""Pure-Python port of gungi.js@3.4.11 — the Gungi (Hunter x Hunter) rules engine.

Copied over verbatim from the v1 engine (KOM-001a); behaviour is frozen and the
module map mirrors gungi.js/src/gungi/:

    utils.py     <- utils.ts      (types, piece tables, board helpers)
    fen.py       <- fen.ts        (parse / encode / validate, start positions)
    move_gen.py  <- move_gen.ts   (movement, tier-blocking, leap, betray, check)
    api.py       <- gungi.ts      (Gungi class: moves/move, check/mate/draw, game-over)

Descoped for the AlphaZero v1 (not needed by training): piece-ids.ts, pgn.ts.
Import submodules directly (e.g. ``from komugi.engine import fen``); this
``__init__`` stays minimal on purpose to avoid shared-file merge conflicts
across parallel tasks.
"""

from __future__ import annotations

__gungi_js_version__: str = "3.4.11"
