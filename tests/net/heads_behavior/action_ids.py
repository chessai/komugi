"""Action-id arithmetic for the heads behaviour suite (KOM-105).

THE CONTRACT THIS FILE EXISTS TO PIN.  The policy head is a 1x1 convolution to
461 planes over the 9x9 board, flattened PLANE-MAJOR, and the resulting flat
index *is* the engine's action id:

    policy_logits[n, plane * 81 + sq] == the logit of engine action id
                                         ``plane * 81 + sq``

with the 461 planes laid out as the four blocks of
``komugi.engine.action_encoding`` concatenated in block order:

    planes   0.. 80  block A0  (81 planes)   ids [    0,  6561)
    planes  81..404  block A1  (324 planes)  ids [ 6561, 32805)
    planes 405..432  block B   (28 planes)   ids [32805, 35073)
    planes 433..460  block C   (28 planes)   ids [35073, 37341)

and ``sq`` the engine's own square index (``action_encoding.sq_index``).

Everything below is TRANSCRIBED from the documented layout in
``src/komugi/engine/action_encoding.py`` -- the module docstring's block table
and the ``block_start + plane * 81 + to_sq`` sentence -- rather than derived
from, or imported out of, the head under test.  A suite that asked the head for
its own arithmetic and then checked the answer against itself would assert
nothing.  ``test_action_id_helpers.py`` closes the loop the other way: it runs
UNCONDITIONALLY (the engine exists on this branch even though the heads do not)
and pins every constant here against the engine's own.

NAMING.  The deliverable list calls this file ``helpers.py``.  It is
``action_ids.py`` because ``tests/encoding/helpers.py`` already exists and the
test tree has no packages: two modules named ``helpers`` collide both in
pytest's sys.path-based collection and in mypy's module resolution
("Duplicate module named 'helpers'"), which is a hard CI failure rather than a
matter of taste.

SQUARES AND SPACE.  ``sq_index`` runs rank 1->9 and, within a rank, file 9->1
(``action_encoding._RF_TO_SQ``: ``r * 9 + (8 - f)`` on 0-based coordinates).
The observation encoder writes the very same square to ``obs[plane, r, 8 - f]``
(``observation.observation_from_cyboard``: ``j = 8 - f``), so in the (9, 9)
spatial layout the trunk and the heads work over,

    sq_index == row * 9 + col

i.e. the square index is exactly the C-order (row-major) flatten of the board
plane.  That identity is what makes the head's ``.flatten(1)`` land on the
engine's ids at all, so it is stated here as its own function pair and pinned
against the engine over all 81 squares.
"""

from __future__ import annotations

from typing import Final, NamedTuple, Sequence

from komugi.engine.observation import observation_size

# ---------------------------------------------------------------------------
# The layout, transcribed
# ---------------------------------------------------------------------------

#: The board the whole net is defined over.
BOARD: Final = 9

#: Squares per policy plane -- the plane-major stride of every block.
SQUARES: Final = 81

#: Policy planes and the flat action space they span.
POLICY_PLANES: Final = 461
NUM_ACTIONS: Final = 37341

#: Blocks in id order, with the plane count each contributes (action_encoding
#: docstring: 81 from-squares, 324 = 81 * 4 selector-widened from-squares,
#: 28 = 14 * 2 arata planes, and the same 28 again for the 終 block).
BLOCK_NAMES: Final = ("A0", "A1", "B", "C")
BLOCK_PLANES: Final = (81, 324, 28, 28)


def _running_sums(counts: Sequence[int], *, scale: int) -> tuple[int, ...]:
    """Exclusive running sums of ``counts * scale`` -- the block offsets.

    The block starts are the running sums *by construction* here rather than
    four hand-typed numbers, so a transcription slip cannot make the four
    constants mutually consistent and jointly wrong.
    """
    out: list[int] = []
    running = 0
    for count in counts:
        out.append(running)
        running += count * scale
    return tuple(out)


#: Index of each block's first PLANE in the 461-plane stack: 0, 81, 405, 433.
BLOCK_PLANE_OFFSETS: Final = _running_sums(BLOCK_PLANES, scale=1)

#: Index of each block's first ACTION ID: 0, 6561, 32805, 35073 -- the
#: ``A0_START`` / ``A1_START`` / ``B_START`` / ``C_START`` of the engine.
BLOCK_STARTS: Final = _running_sums(BLOCK_PLANES, scale=SQUARES)

#: Planes the observation encoder emits, as the encoder itself reports them --
#: never a constant of ours.  The heads suite feeds real observations, so the
#: net's ``in_planes`` has to follow the encoder rather than a literal that can
#: rot; ``test_action_id_helpers`` pins today's value (128) so that a change
#: over there is visible here as one failing assertion and not as a shape error
#: three modules away.
def observation_planes() -> int:
    """``observation.observation_size()[0]`` -- 128 today."""
    return int(observation_size()[0])


# ---------------------------------------------------------------------------
# id <-> (plane, square)
# ---------------------------------------------------------------------------


def id_to_plane_and_square(action_id: int) -> tuple[int, int]:
    """Split an engine action id into its ``(policy plane, square)``.

    Routed deliberately through the BLOCK table -- locate the block, take the
    plane-major ``divmod`` inside it, then add the block's plane offset --
    because that is the statement the layout actually makes.  That this equals
    the flat ``divmod(action_id, 81)`` the head's flatten performs is a
    consequence of the four blocks being contiguous and identically strided,
    and it is a *test* (``test_action_id_helpers``), not an assumption.
    """
    if not 0 <= action_id < NUM_ACTIONS:
        raise ValueError(f"not an action id: {action_id}")
    for block in reversed(range(len(BLOCK_NAMES))):
        start = BLOCK_STARTS[block]
        if action_id >= start:
            local_plane, square = divmod(action_id - start, SQUARES)
            return BLOCK_PLANE_OFFSETS[block] + local_plane, square
    raise AssertionError("unreachable: block A0 starts at id 0")


def plane_and_square_to_id(plane: int, square: int) -> int:
    """Inverse of :func:`id_to_plane_and_square`."""
    if not 0 <= plane < POLICY_PLANES:
        raise ValueError(f"not a policy plane: {plane}")
    if not 0 <= square < SQUARES:
        raise ValueError(f"not a square index: {square}")
    for block in reversed(range(len(BLOCK_NAMES))):
        offset = BLOCK_PLANE_OFFSETS[block]
        if plane >= offset:
            return BLOCK_STARTS[block] + (plane - offset) * SQUARES + square
    raise AssertionError("unreachable: block A0 starts at plane 0")


def block_of_plane(plane: int) -> str:
    """Which of ``A0 / A1 / B / C`` a policy plane belongs to."""
    if not 0 <= plane < POLICY_PLANES:
        raise ValueError(f"not a policy plane: {plane}")
    for block in reversed(range(len(BLOCK_NAMES))):
        if plane >= BLOCK_PLANE_OFFSETS[block]:
            return BLOCK_NAMES[block]
    raise AssertionError("unreachable: block A0 starts at plane 0")


def plane_ids(plane: int) -> range:
    """The 81 contiguous ids a policy plane owns: ``[p * 81, (p + 1) * 81)``."""
    if not 0 <= plane < POLICY_PLANES:
        raise ValueError(f"not a policy plane: {plane}")
    return range(plane * SQUARES, (plane + 1) * SQUARES)


# ---------------------------------------------------------------------------
# Squares <-> board space
# ---------------------------------------------------------------------------


def sq_index(rank: int, file: int) -> int:
    """1-based ``(rank, file)`` -> the 0..80 index the encoding uses."""
    if not (1 <= rank <= BOARD and 1 <= file <= BOARD):
        raise ValueError(f"square out of range: ({rank}, {file})")
    return (rank - 1) * BOARD + (BOARD - file)


def rank_file(square: int) -> tuple[int, int]:
    """Inverse of :func:`sq_index` -- 1-based ``(rank, file)``."""
    if not 0 <= square < SQUARES:
        raise ValueError(f"not a square index: {square}")
    return square // BOARD + 1, BOARD - square % BOARD


def spatial_of_sq(square: int) -> tuple[int, int]:
    """Square index -> the ``(row, col)`` it occupies in a ``(9, 9)`` plane.

    The identity from the module docstring, in one place: row-major flatten.
    """
    if not 0 <= square < SQUARES:
        raise ValueError(f"not a square index: {square}")
    return divmod(square, BOARD)


def sq_of_spatial(row: int, col: int) -> int:
    """Inverse of :func:`spatial_of_sq`."""
    if not (0 <= row < BOARD and 0 <= col < BOARD):
        raise ValueError(f"position out of range: ({row}, {col})")
    return row * BOARD + col


# ---------------------------------------------------------------------------
# The hand-derived pins
# ---------------------------------------------------------------------------


class Pin(NamedTuple):
    """One action id whose ``(plane, square)`` was derived by hand.

    ``meaning`` records the derivation in the layout's own vocabulary, so a
    failure names which corner of which block moved.
    """

    action_id: int
    plane: int
    square: int
    block: str
    meaning: str


#: Every block boundary, plus the design's worked example.  Each entry is
#: arithmetic done on paper against the ``action_encoding`` docstring:
#:
#:     0     = A0 + 0*81 + 0        -> plane 0,   sq 0    (from_sq 0 -> to_sq 0)
#:     6560  = A0 + 80*81 + 80      -> plane 80,  sq 80   (from_sq 80 -> to_sq 80)
#:     6561  = A1 + (0*4+0)*81 + 0  -> plane 81,  sq 0    (from_sq 0, sel 0)
#:     19553 = A1 + (40*4+0)*81 + 32
#:             19553 - 6561 = 12992; 12992 = 160*81 + 32, and 160 = 40*4 + 0,
#:             so from_sq 40 (= 5-5), sel 0 (the DECLINED betrayal -- a plain
#:             tsuke), to_sq 32 (= 4-4).  Plane 81 + 160 = 241, square 32.
#:             This is the accepted id of the design's worked example as the
#:             KOM-128 suite pins it (tests/encoding/test_negative_selectors.py);
#:             the design's own prose quotes 19552, the same move onto 4-5.
#:     32804 = A1 + (80*4+3)*81 + 80 -> plane 404, sq 80  (from_sq 80, sel 3)
#:     32805 = B  + (0*2+0)*81 + 0   -> plane 405, sq 0   (piece 0, bsel 0)
#:     35072 = B  + (13*2+1)*81 + 80 -> plane 432, sq 80  (piece 13, bsel 1)
#:     35073 = C  + (0*2+0)*81 + 0   -> plane 433, sq 0   (piece 0, bsel 0, 終)
#:     37340 = C  + (13*2+1)*81 + 80 -> plane 460, sq 80  (piece 13, bsel 1, 終)
PINS: Final = (
    Pin(0, 0, 0, "A0", "first A0: route/capture from_sq 0 to to_sq 0"),
    Pin(6560, 80, 80, "A0", "last A0: route/capture from_sq 80 to to_sq 80"),
    Pin(6561, 81, 0, "A1", "first A1: from_sq 0, sel 0, to_sq 0"),
    Pin(19553, 241, 32, "A1", "worked example: from_sq 40 (5-5), sel 0, to_sq 32 (4-4)"),
    Pin(32804, 404, 80, "A1", "last A1: from_sq 80, sel 3, to_sq 80"),
    Pin(32805, 405, 0, "B", "first B: arata piece 0, bsel 0, to_sq 0"),
    Pin(35072, 432, 80, "B", "last B: arata piece 13, bsel 1, to_sq 80"),
    Pin(35073, 433, 0, "C", "first C: arata+終 piece 0, bsel 0, to_sq 0"),
    Pin(37340, 460, 80, "C", "last C: arata+終 piece 13, bsel 1, to_sq 80"),
)
