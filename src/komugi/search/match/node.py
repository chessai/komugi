"""One expanded position, and the read-time arithmetic over it.

The whole of design §S lives here. The node schema's single organising rule is
in the design's own words: *no stored field is a function of contempt*. Note
what is absent from :data:`S1_FIELDS` -- there is no ``value``, no
``value_sum``, no ``q`` and no ``utility`` field anywhere. The scalar does not
exist in storage, and it is impossible to poison what is never written.

Two consequences shape everything below.

* **Utility is derived, never stored.** :func:`utility` collapses a stored WDL
  triple under a contempt the *caller* supplies, in float64, at every call
  site, every time.
* **The second moment is stored as a WDL outer product, not as a sum of
  squared utilities.** ``u = q.x`` is linear, so ``u**2 = q'(x x')q`` and the
  sum of weighted squares is an exact quadratic form in a contempt-free
  matrix (design §S4). Six unique entries suffice, packed by
  :func:`pack_outer` and read by :func:`quadratic_form`.

Arrays are **sparse over the node's own legal set**, not over the 37341-wide
action space: ``actions`` holds the legal ids ascending and every per-child
array is indexed by position in ``actions``. At ``L = 229`` that is about
22 KB per node against the ~2.7 MB a full-width ``moment`` would cost, which
is the difference between a reusable tree and an out-of-memory one
(design §S3).
"""

from __future__ import annotations

from typing import Final

import numpy as np
import numpy.typing as npt

from komugi.selfplay.inference import wdl_utility

#: Every field of design §S1's table, in its order. Declared **once**, here,
#: because two acceptances read it: A10a snapshots exactly these buffers and
#: asserts they do not move under a second contempt, and A10b's static taint
#: check fails on a contempt-derived expression written into any of them. A
#: new field joins both checks by being declared in this tuple.
S1_FIELDS: Final[tuple[str, ...]] = (
    "to_play",
    "is_terminal",
    "is_root",
    "actions",
    "legal_hash",
    "prior",
    "logits",
    "wdl_net",
    "visits",
    "weight",
    "wdl_sum",
    "moment",
    "bias_key",
    "bias_contribution",
    "bias_weight",
    "bias_gen",
    "children",
)

#: Length of a WDL triple and of the packed second moment's diagonal.
WDL: Final[int] = 3

#: Number of unique entries in a symmetric 3x3 matrix: ``(ww, wd, wl, dd, dl,
#: ll)``. The order is part of the storage contract and is pinned by a test
#: against a dense reference, because a transposed index here produces a
#: variance that is merely *wrong* rather than obviously broken.
MOMENT_SIZE: Final[int] = 6

#: The "no previous move" slot of design §A2's bias key, at the root of a
#: fresh tree where no grandparent edge exists.
NO_PREVIOUS_ACTION: Final[int] = -1

#: "No mover is known for the edge that reached this node", used by the tree
#: while no move has been played into its root. It is **not** a key component:
#: design §A2's mover slot is the mover at the *parent* of the edge, so a tree
#: that has been rehomed knows one and a fresh tree does not, and the two cases
#: have to be distinguishable from a legal mover id (which is 0 or 1).
NO_MOVER: Final[int] = -1

_FNV_OFFSET_BASIS: Final[int] = 0xCBF29CE484222325
_FNV_PRIME: Final[int] = 0x100000001B3
_UINT64_SPAN: Final[int] = 1 << 64
_INT64_LIMIT: Final[int] = 1 << 63


def legal_hash_of(actions: npt.NDArray[np.int32]) -> int:
    """Return the FNV-1a 64 hash of `actions`, as a signed int64.

    **Not** :func:`hash`. Python's hash of a ``bytes`` is randomised by
    ``PYTHONHASHSEED``, so a stored hash computed with it would be
    incomparable across processes and would silently break any golden that
    carries one -- and design §A4's rehome check stores exactly this number on
    the node. The arithmetic is the standard FNV-1a: XOR each octet into the
    digest, then multiply by the prime, modulo ``2**64``.

    Args:
        actions: The node's legal action ids, ascending. Cast to int32 before
            hashing so the digest does not depend on the caller's dtype.

    Returns:
        The digest reinterpreted as a signed int64, which is the dtype
        design §S1 declares for ``legal_hash``.
    """
    octets = np.frombuffer(
        np.ascontiguousarray(actions, dtype=np.int32).tobytes(), dtype=np.uint8
    )
    digest: np.uint64 = np.uint64(_FNV_OFFSET_BASIS)
    prime: np.uint64 = np.uint64(_FNV_PRIME)
    # Wrapping unsigned arithmetic is the algorithm, not an accident, so the
    # overflow it relies on is silenced rather than left to warn per octet.
    with np.errstate(over="ignore"):
        for octet in octets:
            digest = np.uint64((digest ^ np.uint64(octet)) * prime)
    unsigned = int(digest) % _UINT64_SPAN
    return unsigned - _UINT64_SPAN if unsigned >= _INT64_LIMIT else unsigned


def bias_key_of(
    to_play: int, action: int, previous_action: int, num_actions: int
) -> int:
    """Return design §A2's subtree-value-bias key, packed into one int64.

    The key is ``(mover, the move being made, the move that preceded it)``.
    KataGo's key additionally hashes a *local board pattern*; nothing in
    ``komugi.engine`` exposes such a hash and this design does not invent one
    (U-A3). The consequence is booked rather than hidden: this key is coarser,
    so it pools more edges per entry, so entries are better estimated but less
    specific and the correction is blunter than KataGo's.

    Args:
        to_play: The mover at the *parent* of the edge.
        action: The action the edge plays.
        previous_action: The action into the parent, or
            :data:`NO_PREVIOUS_ACTION` at the root of a fresh tree.
        num_actions: The action space width, which fixes the packing radix.

    Returns:
        A mixed-radix packing, injective over the three components.
    """
    radix = num_actions + 1
    return (to_play * radix + (action + 1)) * radix + (previous_action + 1)


def edge_action_of(bias_key: int, num_actions: int) -> int:
    """Return the action slot of `bias_key`: the move the keyed edge plays.

    The inverse of :func:`bias_key_of` on its middle component, and it exists
    for one caller: selection has to name the key of every edge *out of* a
    node, whose "previous move" slot is the move played *into* that node --
    which is exactly this node's own :attr:`MatchNode.bias_key` action slot.
    Nothing on the node stores that action separately, and adding a field for
    it would be a design §S1 schema change to hold a number the key already
    contains.

    It recovers the same value the playout loop threads down as
    ``previous_action``, at the root as well: a fresh root's key is
    ``(mover, root_action, -1)``, so this returns ``root_action`` -- or
    :data:`NO_PREVIOUS_ACTION` when no move has been played into the tree at
    all.

    Args:
        bias_key: A key produced by :func:`bias_key_of`.
        num_actions: The action space width the key was packed with. A
            different width decodes a different action, silently, so this must
            be the same ``num_actions`` the key was built under.

    Returns:
        The action the keyed edge plays, or :data:`NO_PREVIOUS_ACTION`.
    """
    radix = num_actions + 1
    return (bias_key // radix) % radix - 1


def utility(wdl: npt.NDArray[np.float64], contempt: float) -> float:
    """Collapse one stored WDL triple to a utility in ``[-1, 1]``.

    ``w - l + contempt * d``: design §S2's read-time collapse, and the same
    arithmetic ``komugi.selfplay.inference.wdl_utility`` implements. It is
    **re-derived here in float64 rather than delegated** on the hot path:
    that function is float32-in/float32-out and requires a ``(N, 3)`` input,
    so routing every child read of every playout through it would downcast
    every utility and force a reshape allocation in the hottest loop in the
    search. The two can therefore drift, and :func:`utility_reference` plus
    acceptance A9 are what stop them.

    Args:
        wdl: A ``(3,)`` win/draw/loss triple -- a mean, a sum, or a net
            output; the collapse is linear, so it commutes with the division
            by a weight.
        contempt: What a draw is worth to the side to move.

    Returns:
        The utility, in the perspective the triple is stored in.
    """
    return float(wdl[0]) - float(wdl[2]) + contempt * float(wdl[1])


def utility_reference(wdl: npt.NDArray[np.float64], contempt: float) -> float:
    """Return :func:`utility` computed through production's ``wdl_utility``.

    The cross-check helper of acceptance A9, and the **only** use of
    ``wdl_utility`` in this package: it exists so a test can pin the hot-path
    collapse against ``inference.py``'s within ``1e-6``, and it is never
    called by the search itself.

    Args:
        wdl: A ``(3,)`` win/draw/loss triple.
        contempt: What a draw is worth to the side to move.

    Returns:
        The production collapse of the same triple, as a Python float.
    """
    row = np.asarray(wdl, dtype=np.float32).reshape(1, WDL)
    return float(wdl_utility(row, contempt)[0])


def pack_outer(wdl: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return ``x x'`` packed as ``(ww, wd, wl, dd, dl, ll)``.

    The six unique entries of a symmetric 3x3 matrix of pure WDL products,
    with no contempt anywhere in it -- which is the whole reason the variance
    of a *contempt-dependent* utility can be recovered at read time from
    contempt-free storage (design §S4).

    Args:
        wdl: A ``(3,)`` win/draw/loss triple.

    Returns:
        A ``(6,)`` float64 array in the packing order above.
    """
    win = float(wdl[0])
    draw = float(wdl[1])
    loss = float(wdl[2])
    return np.asarray(
        [
            win * win,
            win * draw,
            win * loss,
            draw * draw,
            draw * loss,
            loss * loss,
        ],
        dtype=np.float64,
    )


def quadratic_form(
    moment: npt.NDArray[np.float64], contempt: float
) -> npt.NDArray[np.float64]:
    """Return ``q' M q`` for each packed row of `moment`, with ``q = (1, c, -1)``.

    Expanding the symmetric form over the packing of :func:`pack_outer`::

        q'Mq = M0 + c**2 * M3 + M5 + 2c * M1 - 2 * M2 - 2c * M4

    The off-diagonal factors of 2 and the sign on the win/loss cross term are
    the two places a careless expansion goes wrong, so both are pinned by a
    test against a dense ``q @ M @ q`` reference (design §S4, T-MOMENT).

    Args:
        moment: ``(L, 6)`` packed second moments, one row per edge.
        contempt: What a draw is worth to the side to move.

    Returns:
        ``(L,)`` float64 values of the quadratic form.
    """
    packed = np.asarray(moment, dtype=np.float64)
    return np.asarray(
        packed[:, 0]
        + contempt * contempt * packed[:, 3]
        + packed[:, 5]
        + 2.0 * contempt * packed[:, 1]
        - 2.0 * packed[:, 2]
        - 2.0 * contempt * packed[:, 4],
        dtype=np.float64,
    )


def masked_prior(logits: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Return the softmax of `logits` over a node's legal actions.

    The same construction Gumbel uses, restricted to the legal subset because
    that is the only axis a match node has: the max is subtracted before
    exponentiating, in float64, and the float32 result is what design §S1's
    ``prior`` field stores.

    Args:
        logits: ``(L,)`` raw network logits at the node's legal actions.

    Returns:
        ``(L,)`` float32 probabilities summing to 1, or an all-zero array
        when there are no legal actions (a terminal node).
    """
    wide = np.asarray(logits, dtype=np.float64)
    if wide.shape[0] == 0:
        return np.zeros(0, dtype=np.float32)
    exp = np.exp(wide - float(wide.max()))
    return np.asarray(exp / float(exp.sum()), dtype=np.float32)


class MatchNode:
    """One expanded position: design §S1's field list and nothing else.

    Statistics live on the **parent**, keyed by position in :attr:`actions`,
    which is what lets a node report a weight and a visit count for an edge
    whose child was never created. :attr:`wdl_sum` accumulates in *this*
    node's perspective, so the negamax swap happens once per edge on the way
    up (design §A6) rather than at every read.

    There is deliberately **no parent back-pointer**: a rehomed tree must be
    able to drop every sibling subtree, and a back-pointer would keep the
    whole game's tree alive through the retained child (design §A4, T11).

    Args:
        to_play: The mover at this position.
        is_terminal: Whether the engine has adjudicated here. A terminal node
            has no legal actions and is never expanded further.
        actions: ``(L,)`` int32 legal action ids, **ascending**. The ascending
            order is load-bearing twice: ``np.argmax`` ties then fall to the
            lowest action id, and design §A4's rehome check compares this
            array against ``np.flatnonzero`` of a mask.
        logits: ``(L,)`` float32 raw network logits at `actions`.
        wdl_net: ``(3,)`` float64 network WDL for this position in its own
            mover's perspective -- or the one-hot result at a terminal node.
        bias_key: Design §A2's key for the edge that reaches this node.
        bias_gen: The generation of that key's table entry at expansion time.
        is_root: Whether this node is the tree's root, which selects the FPU
            constant (design §A1) and nothing else.
        moment_dtype: ``"float64"`` or ``"float32"`` (design §S3's memory
            escape hatch).
    """

    # Exactly design §S1's fields, plus the weak-reference slot a test needs:
    # T11 holds `weakref`s to two dropped sibling subtrees and asserts they
    # die, and a slotted class without `__weakref__` cannot be referred to
    # weakly at all.
    __slots__ = (*S1_FIELDS, "__weakref__")

    def __init__(
        self,
        *,
        to_play: int,
        is_terminal: bool,
        actions: npt.NDArray[np.int32],
        logits: npt.NDArray[np.float32],
        wdl_net: npt.NDArray[np.float64],
        bias_key: int,
        bias_gen: int = 0,
        is_root: bool = False,
        moment_dtype: str = "float64",
    ) -> None:
        ids = np.ascontiguousarray(actions, dtype=np.int32)
        width = int(ids.shape[0])
        self.to_play: int = int(to_play)
        self.is_terminal: bool = bool(is_terminal)
        self.is_root: bool = bool(is_root)
        self.actions: npt.NDArray[np.int32] = ids
        self.legal_hash: int = legal_hash_of(ids)
        self.logits: npt.NDArray[np.float32] = np.asarray(logits, dtype=np.float32)
        self.prior: npt.NDArray[np.float32] = masked_prior(self.logits)
        self.wdl_net: npt.NDArray[np.float64] = np.asarray(wdl_net, dtype=np.float64)
        self.visits: npt.NDArray[np.int32] = np.zeros(width, dtype=np.int32)
        self.weight: npt.NDArray[np.float64] = np.zeros(width, dtype=np.float64)
        self.wdl_sum: npt.NDArray[np.float64] = np.zeros((width, WDL), dtype=np.float64)
        self.moment: npt.NDArray[np.float64] = np.zeros(
            (width, MOMENT_SIZE), dtype=np.dtype(moment_dtype)
        )
        self.bias_key: int = int(bias_key)
        self.bias_contribution: npt.NDArray[np.float64] = np.zeros(WDL, dtype=np.float64)
        self.bias_weight: float = 0.0
        self.bias_gen: int = int(bias_gen)
        self.children: dict[int, MatchNode] = {}

    @property
    def width(self) -> int:
        """Return ``L``, the number of legal actions at this node."""
        return int(self.actions.shape[0])

    def total_weight(self) -> float:
        """Return ``W_tot(n)``, the summed weight of every edge below."""
        return float(self.weight.sum())

    def child_index(self, action: int) -> int | None:
        """Return the position of `action` in :attr:`actions`, or ``None``.

        A binary search rather than a scan: `actions` is ascending by
        construction, and this is on the rehoming path once per played move.

        Args:
            action: An action id, legal here or not.

        Returns:
            The index into every per-child array, or ``None`` if `action` is
            not one of this node's legal actions.
        """
        position = int(np.searchsorted(self.actions, np.int32(action)))
        if position >= self.width or int(self.actions[position]) != int(action):
            return None
        return position

    def subtree_wdl_sum(self) -> npt.NDArray[np.float64]:
        """Return ``sum_a S(a)``, this node's total backed-up WDL mass."""
        return np.asarray(self.wdl_sum.sum(axis=0), dtype=np.float64)

    def __repr__(self) -> str:
        return (
            f"MatchNode(to_play={self.to_play}, L={self.width}, "
            f"is_terminal={self.is_terminal}, is_root={self.is_root}, "
            f"W_tot={self.total_weight():.3f})"
        )
