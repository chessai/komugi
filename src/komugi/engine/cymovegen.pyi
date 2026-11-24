"""Stub for the compiled komugi.engine.cymovegen extension (see cymovegen.pyx).

The two python-visible entry points plus the action-layout constants the .pyx
duplicates from action_encoding (exported so a test can assert the duplication
has not drifted); every generator helper in the .pyx is ``cdef`` and has no
python-level name. See cyboard.pyi for why this file lives beside the .pyx
instead of under stubs/.
"""

from typing import Any

from komugi.engine.cyboard import Board

# Block starts and total width — must equal action_encoding's (KOM-128 T4).
A0_START: int
A1_START: int
B_START: int
C_START: int
ACTION_SIZE: int

def generate_all_moves(b: Board) -> list[dict[str, Any]]: ...
def fill_legal_mask(b: Board, mask: Any) -> None: ...
